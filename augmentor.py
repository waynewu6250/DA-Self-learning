import torch
import torch.nn as nn
from torch.optim import Adam
import os
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import json

from model import DAE, VAE, AAE
from utils import *

class Augmentor:

    def __init__(self, args, opt, device, vocab, new_vocab, checkpoint_path, output_path):

        self.args = args
        self.opt = opt
        self.device = device
        self.vocab = vocab
        self.new_vocab = new_vocab
        self.checkpoint_path = checkpoint_path
        self.output_path = output_path
        self.dictionary = {'domain': len(vocab['domain']['stoi']),
                           'intent': len(vocab['intent']['stoi']),
                           'slotKey': len(vocab['slotKey']['stoi'])}
        self.model = self.create_model(new_vocab)

    def create_model(self, new_vocab):

        model = None

        if self.args.model == 'autoencoder':
            model = VAE(self.dictionary, self.args, self.opt, self.device, new_vocab)
        elif self.args.model == 'adv_autoencoder':
            model = AAE(self.dictionary, self.args, self.opt, self.device, new_vocab)

        ckpt = torch.load(self.checkpoint_path)
        model.load_state_dict(ckpt['model'])
        print("Pretrained model has been loaded.")
        model = model.to(self.device)

        return model

    def generate(self, test_dataloader):
        """Generate data based on the chosen trained model"""
        self.model.eval()

        if self.args.model == 'cvae':

            for (batch_data, batch_length) in test_dataloader:
                for key in batch_data.keys():
                    if key != 'uid':
                        batch_data[key] = batch_data[key].to(self.device)
                        batch_length[key] = batch_length[key].to(self.device)

                samples, z = self.model.inference(batch_data, batch_length, n=self.args.num_samples)
                print('----------SAMPLES----------')
                print(samples)
                # print(*idx2word(samples,
                #                 i2w=self.vocab['transcription']['itos'],
                #                 pad_idx=self.vocab['transcription']['stoi']['<pad>']),
                #                 sep='\n')

            # z1 = torch.randn([self.opt.latent_size]).numpy()
            # z2 = torch.randn([self.opt.latent_size]).numpy()
            # z = torch.from_numpy(interpolate(start=z1, end=z2, steps=8)).float().to(self.device)
            # samples, _ = self.model.inference(z=z)
            # print('-------INTERPOLATION-------')
            # print(*idx2word(samples,
            #                 i2w=self.vocab['transcription']['itos'],
            #                 pad_idx=self.vocab['transcription']['stoi']['<pad>']),
            #       sep='\n')

        elif self.args.model == 'autoencoder' or self.args.model == 'adv_autoencoder':
            # random sample
            # z = np.random.normal(size=(self.args.num_samples, self.opt.dim_z)).astype('f')
            # sents = self.decode(z)
            # unique_sents = set([' '.join(s) for s in sents])
            # print('Unique rate: ', len(unique_sents) / len(sents))
            # write_sent(sents, self.output_path)

            # interpolate
            # new_vocab = test_dataloader.dataset.new_vocab
            # f1, f2 = self.args.data.split(',')
            # s1, s2 = load_sent(f1), load_sent(f2)
            # z1, z2 = self.encode(s1, new_vocab), self.encode(s2, new_vocab)
            # zi = [interpolate(z1_, z2_, self.args.num_samples) for z1_, z2_ in zip(z1, z2)]
            # zi = np.concatenate(zi, axis=0)
            # si = self.decode(zi)
            # si = list(zip(*[iter(si)] * (self.args.num_samples)))
            # write_doc(si, os.path.join(self.args.load_checkpoint, self.args.output))

            # conditional from data
            if self.args.generate_type == 'data':
                original_sents = []
                domains = []
                intents = []
                sents = []
                for (batch_data, batch_length) in test_dataloader:
                    for key in batch_data.keys():
                        if key != 'uid':
                            batch_data[key] = batch_data[key].to(self.device)
                            batch_length[key] = batch_length[key].to(self.device)

                    zi = np.random.normal(size=(batch_data['transcription'].size(0), self.opt.dim_z)).astype('f')
                    zi = torch.tensor(zi, device=self.device)
                    outputs = self.model.generate(zi, self.opt.max_len, self.opt.dec, batch_data).t()
                    for s in outputs:
                        sents.append([self.new_vocab.idx2word[id] for id in s[1:]])  # skip <go>

                    # Retrieve original sentences/domains/intents/slot keys
                    decoded_sents = idx2sent(batch_data['target'], self.new_vocab.idx2word)
                    original_sents.extend(decoded_sents)
                    for i in range(len(batch_data['domain'])):
                        domains.append(self.vocab['domain']['itos'][batch_data['domain'][i]])
                        intents.append(self.vocab['intent']['itos'][batch_data['intent'][i]])

                sents = [sent[:sent.index('<eos>')] if '<eos>' in sent else sent for sent in sents]
                unique_sents = set([' '.join(s) for s in sents])
                print('Unique rate: ', len(unique_sents) / len(sents))
                write_sent(self.output_path, original_sents, domains, intents, sents)

            # conditional from file
            elif self.args.generate_type == 'file':
                domains, intents = load_NLU('/home/ec2-user/checkpoints/NLU_{}.txt'.format(self.args.data_folder[:-1]))
                with open(self.output_path+'_file', 'w') as f:
                    for domain, intent in zip(domains, intents):
                        z = np.random.normal(size=(self.args.num_samples, self.opt.dim_z)).astype('f')
                        domain_id = self.vocab['domain']['stoi'][domain]
                        intent_id = self.vocab['intent']['stoi'][intent]
                        sents = self.decode_condition(z, domain_id, intent_id)
                        f.write('{:<20} | {:<20}'.format('Domain: '+domain, 'Intent: '+intent) + '\n')
                        f.write('-' * 50 + '\n')
                        for s in sents:
                            f.write(' '.join(s) + '\n')
                        f.write('\n\n')

    def decode_condition(self, z, domain, intent):
        sents = []
        i = 0
        while i < len(z):
            zi = torch.tensor(z[i: i + self.opt.test_batch_size], device=self.device)
            batch_data = {'domain': torch.LongTensor(len(zi)).fill_(domain).to(self.device),
                          'intent': torch.LongTensor(len(zi)).fill_(intent).to(self.device)}
            outputs = self.model.generate(zi, self.opt.max_len, self.opt.dec, batch_data).t()
            for s in outputs:
                sents.append([self.new_vocab.idx2word[id] for id in s[1:]])  # skip <go>
            i += self.opt.test_batch_size
        return [sent[:sent.index('<eos>')] if '<eos>' in sent else sent
                for sent in sents]

    def decode(self, z):
        sents = []
        i = 0
        while i < len(z):
            zi = torch.tensor(z[i: i + self.opt.test_batch_size], device=self.device)
            outputs = self.model.generate(zi, self.opt.max_len, self.opt.dec).t()
            for s in outputs:
                sents.append([self.new_vocab.idx2word[id] for id in s[1:]])  # skip <go>
            i += self.opt.test_batch_size
        return [sent[:sent.index('<eos>')] if '<eos>' in sent else sent
                for sent in sents]

    def encode(self, sents, vocab):
        batches, order = get_batches(sents, vocab, self.opt.test_batch_size, self.device)
        z = []
        for inputs, _ in batches:
            mu, logvar = self.model.encode(inputs)
            if self.opt.enc == 'mu':
                zi = mu
            else:
                zi = self.reparameterize(mu, logvar)
            z.append(zi.detach().cpu().numpy())
        z = np.concatenate(z, axis=0)
        z_ = np.zeros_like(z)
        z_[np.array(order)] = z
        return z_

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

