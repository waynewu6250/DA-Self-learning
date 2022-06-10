import torch
import torch.nn as nn
from torch.optim import Adam
import os
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import json

from model import EncoderRNN, DecoderRNN, VAE, CVAE
from utils import idx2word

class CVAETrainer:

    def __init__(self, args, opt, device, vocab):

        self.args = args
        self.opt = opt
        self.device = device
        self.vocab = vocab
        self.checkpoint_path = '/home/ec2-user/checkpoints/{}/best_model.pth'.format(args.model)
        self.vocab_size = {'transcription': len(vocab['transcription']['stoi']),
                           'domain': len(vocab['domain']['stoi']),
                           'intent': len(vocab['intent']['stoi']),
                           'slotKey': len(vocab['slotKey']['stoi'])}
        self.model = self.create_model(vocab)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.opt.lr)
        self.criterion = nn.NLLLoss(ignore_index=1, reduction='sum').to(self.device)
        self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
        self.step = 0

    def create_model(self, vocab):

        if self.args.model == 'cvae':
            # e = EncoderRNN(self.vocab_size['transcription'], self.opt, self.device).to(self.device)
            # d = DecoderRNN(self.opt, dataset.trn_split.n_conditions, condition_size, decoder_hidden_size, n_words, 1,
            #                word_dropout=word_dropout).to(self.device)
            # model = VAE(e, d, n_steps).to(self.device)

            model = CVAE(self.vocab_size, self.opt, self.device, vocab)

            # if os.path.exists(self.checkpoint_path):
            #     model.load_state_dict(torch.load(self.checkpoint_path))
            #     print("Pretrained model has been loaded.")
            # else:
            print("Train from scratch...")

        model = model.to(self.device)

        return model

    def loss_fn(self, logp, target, length, mean, logv, anneal_function, step, k, x0):

        def kl_anneal_function(anneal_function, step, k, x0):
            if anneal_function == 'logistic':
                return float(1 / (1 + np.exp(-k * (step - x0))))
            elif anneal_function == 'linear':
                return min(1, step / x0)

        # cut-off unnecessary padding from target, and flatten
        target = target[:, :torch.max(length).item()].contiguous().view(-1)
        logp = logp.view(-1, logp.size(2))

        # Negative Log Likelihood
        NLL_loss = self.criterion(logp, target)

        # KL Divergence
        KL_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
        KL_weight = kl_anneal_function(anneal_function, step, k, x0)

        return NLL_loss, KL_loss, KL_weight

    def train_one_batch(self, dataloader, type):

        tracker = defaultdict(self.tensor)
        if type == 'train':
            self.model.train()
        else:
            self.model.eval()
        for i, (batch_data, batch_length) in enumerate(tqdm(dataloader,
                                                            total=len(dataloader),
                                                            desc='Batches',
                                                            unit=' batches',
                                                            ncols=80)):
            for key in batch_data.keys():
                if key != 'uid':
                    batch_data[key] = batch_data[key].to(self.device)
                    batch_length[key] = batch_length[key].to(self.device)
            batch_size = batch_data['transcription'].size(0)

            self.optimizer.zero_grad()

            # Forward pass
            logp, mean, logv, z = self.model(batch_data, batch_length)

            # loss calculation
            NLL_loss, KL_loss, KL_weight = self.loss_fn(logp, batch_data['target'],
                                                        batch_length['transcription'],
                                                        mean, logv, self.opt.anneal_function, self.step, self.opt.k,
                                                        self.opt.x0)
            loss = (NLL_loss + KL_weight * KL_loss) / batch_size

            if type == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.step += 1

            tracker['ELBO'] = torch.cat((tracker['ELBO'], loss.data.view(1, -1)), dim=0)

            if i % self.opt.print_every == 0 or i + 1 == len(dataloader):
                print("%s Batch %04d/%i, Loss %9.4f, NLL-Loss %9.4f, KL-Loss %9.4f, KL-Weight %6.3f"
                      % (type.upper(), i,
                         len(dataloader) - 1, loss.item(),
                         NLL_loss.item() / batch_size,
                         KL_loss.item() / batch_size,
                         KL_weight))

            if type == 'valid':
                if 'target_sents' not in tracker:
                    tracker['target_sents'] = list()
                tracker['target_sents'] += idx2word(batch_data['target'],
                                                    i2w=self.vocab['transcription']['itos'],
                                                    pad_idx=self.vocab['transcription']['stoi']['<pad>'])
                tracker['z'] = torch.cat((tracker['z'], z.data), dim=0)
        return tracker


    def train(self, train_dataloader, valid_dataloader):

        best_loss = float('inf')
        for epoch in range(self.opt.epochs):
            print("====== epoch %d / %d: ======" % (epoch + 1, self.opt.epochs))

            train_tracker = self.train_one_batch(train_dataloader, 'train')
            valid_tracker = self.train_one_batch(valid_dataloader, 'valid')
            train_epoch_ELBO = train_tracker['ELBO'].mean()
            print("Train Epoch %02d/%i, Mean ELBO %9.4f" % (epoch, self.opt.epochs, train_epoch_ELBO))
            valid_epoch_ELBO = valid_tracker['ELBO'].mean()
            print("Valid Epoch %02d/%i, Mean ELBO %9.4f" % (epoch, self.opt.epochs, valid_epoch_ELBO))

            # save a dump of all sentences and the encoded latent space
            dump = {'target_sents': valid_tracker['target_sents'], 'z': valid_tracker['z'].tolist()}
            with open(os.path.join('/home/ec2-user/checkpoints/{}/meta_data/'.format(self.args.model), 'valid_E%i.json' % epoch), 'w') as dump_file:
                json.dump(dump, dump_file)


            if valid_epoch_ELBO <= best_loss:
                print('saving with loss of {}'.format(valid_epoch_ELBO),
                      'improved over previous {}'.format(best_loss))
                best_loss = valid_epoch_ELBO

                # Save model
                torch.save(self.model.state_dict(), self.checkpoint_path)
                print("Model saved at %s" % self.checkpoint_path)
