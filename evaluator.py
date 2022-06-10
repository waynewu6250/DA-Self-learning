import torch
import torch.nn as nn
from torch.optim import Adam
import os
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import json
from nltk.translate.bleu_score import sentence_bleu

from model import DAE, VAE, AAE
from utils import *
from meter import AverageMeter

class Evaluator:

    def __init__(self, args, opt, device, vocab, new_vocab, checkpoint_path):

        self.args = args
        self.opt = opt
        self.device = device
        self.vocab = vocab
        self.new_vocab = new_vocab
        self.checkpoint_path = checkpoint_path
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

    def evaluate(self, test_dataloader):
        """Evaluate test data based on the chosen trained model"""
        self.model.eval()

        # accuracy metrics
        meters, nll, ppl, bleu_score1, bleu_score2 = self.calculate_accuracy(test_dataloader)
        print(' '.join(['{}: {:.3f},'.format(k, meter.avg)
                        for k, meter in meters.items()]))
        print('NLL: {:.3f}, PPL: {:.3f}'.format(nll, ppl))
        print('BLEU-1: {:.3f}, BLEU-2: {:.3f}'.format(bleu_score1, bleu_score2))


    def calculate_accuracy(self, dataloader):

        meters = defaultdict(lambda: AverageMeter())
        total_nll = 0
        total_num = 0
        n_words = 0
        bleu_score1 = 0
        bleu_score2 = 0

        with torch.no_grad():
            for i, (batch_data, batch_length) in enumerate(tqdm(dataloader,
                                                                total=len(dataloader),
                                                                desc='Batches',
                                                                unit=' batches',
                                                                ncols=80)):
                for key in batch_data.keys():
                    if key != 'uid':
                        batch_data[key] = batch_data[key].to(self.device)
                        batch_length[key] = batch_length[key].to(self.device)
                batch_data['transcription'] = batch_data['transcription'].t().contiguous() # transpose txb
                batch_data['target'] = batch_data['target'].t().contiguous()  # transpose txb

                # calculate loss
                losses = self.model.autoenc(batch_data['transcription'], batch_data['target'], batch_data)
                for k, v in losses.items():
                    meters[k].update(v.item(), batch_data['transcription'].size(1))

                # calculate ppl
                total_nll += self.model.nll_is(batch_data['transcription'], batch_data['target'], batch_data,
                                               self.opt.m_importance).sum().item()
                total_num += batch_data['transcription'].size(1)
                n_words += (torch.sum(batch_length['transcription']).item())

                # calculate bleu
                decoded_sents = idx2sent(batch_data['target'].t().contiguous(), self.new_vocab.idx2word)

                mu, logvar, z, logits = self.model(batch_data['transcription'], batch_data)
                decoded_idx = logits.argmax(dim=-1)
                reconstruct_sents = idx2sent(decoded_idx.t().contiguous(), self.new_vocab.idx2word)

                for t, r in zip(decoded_sents, reconstruct_sents):
                    bleu_score1 += sentence_bleu([t], r, weights=(1.0, 0.0, 0.0, 0.0))
                    bleu_score2 += sentence_bleu([t], r, weights=(0.0, 1.0, 0.0, 0.0))

        loss = self.model.loss({k: meter.avg for k, meter in meters.items()})
        meters['loss'].update(loss)
        print('Total words: ', n_words)

        return meters, total_nll / total_num, np.exp(total_nll / n_words), bleu_score1 / total_num, bleu_score2 / total_num







