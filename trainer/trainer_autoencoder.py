import torch
import torch.nn as nn
from torch.optim import Adam
import os
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import json
import time

from model import DAE, VAE, AAE, RegVAE
from utils import idx2word
from meter import AverageMeter

class AETrainer:

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

        if self.args.model == 'autoencoder':
            model = VAE(self.dictionary, self.args, self.opt, self.device, new_vocab)
        elif self.args.model == 'adv_autoencoder':
            model = AAE(self.dictionary, self.args, self.opt, self.device, new_vocab)
        elif self.args.model == 'reg_autoencoder':
            model = RegVAE(self.dictionary, self.args, self.opt, self.device, new_vocab)

            # if os.path.exists(self.checkpoint_path):
            #     model.load_state_dict(torch.load(self.checkpoint_path))
            #     print("Pretrained model has been loaded.")
            # else:
        print("Train from scratch...")
        print('# model parameters: {}'.format(sum(x.data.nelement() for x in model.parameters())))
        model = model.to(self.device)

        return model

    def evaluate(self, valid_dataloader):

        self.model.eval()
        meters = defaultdict(lambda: AverageMeter())
        with torch.no_grad():
            for i, (batch_data, batch_length) in enumerate(tqdm(valid_dataloader,
                                                                total=len(valid_dataloader),
                                                                desc='Batches',
                                                                unit=' batches',
                                                                ncols=80)):
                for key in batch_data.keys():
                    if key != 'uid':
                        batch_data[key] = batch_data[key].to(self.device)
                        batch_length[key] = batch_length[key].to(self.device)
                batch_data['transcription'] = batch_data['transcription'].t().contiguous() # transpose txb
                batch_data['target'] = batch_data['target'].t().contiguous()  # transpose txb

                losses = self.model.autoenc(batch_data['transcription'], batch_data['target'], batch_data)
                for k, v in losses.items():
                    meters[k].update(v.item(), batch_data['transcription'].size(1))
        loss = self.model.loss({k: meter.avg for k, meter in meters.items()})
        meters['loss'].update(loss)

        return meters


    def train(self, train_dataloader, valid_dataloader):

        best_val_loss = None
        for epoch in range(self.opt.epochs):
            start_time = time.time()
            print("====== epoch %d / %d: ======" % (epoch + 1, self.opt.epochs))
            self.model.train()
            meters = defaultdict(lambda: AverageMeter())
            for i, (batch_data, batch_length) in enumerate(tqdm(train_dataloader,
                                                                total=len(train_dataloader),
                                                                desc='Batches',
                                                                unit=' batches',
                                                                ncols=80)):
                for key in batch_data.keys():
                    if key != 'uid':
                        batch_data[key] = batch_data[key].to(self.device)
                        batch_length[key] = batch_length[key].to(self.device)
                batch_data['transcription'] = batch_data['transcription'].t().contiguous()  # transpose txb
                batch_data['target'] = batch_data['target'].t().contiguous()  # transpose txb

                losses = self.model.autoenc(batch_data['transcription'], batch_data['target'], batch_data, is_train=True)
                losses['loss'] = self.model.loss(losses)
                self.model.step(losses)
                for k, v in losses.items():
                    meters[k].update(v.item())

                if (i + 1) % self.opt.log_interval == 0:
                    log_output = '| epoch {:3d} | {:5d}/{:5d} batches |'.format(
                        epoch + 1, i + 1, len(train_dataloader))
                    for k, meter in meters.items():
                        log_output += ' {} {:.2f},'.format(k, meter.avg)
                        meter.clear()
                    print(log_output)

            # evalaute
            valid_meters = self.evaluate(valid_dataloader)
            log_output = '| end of epoch {:3d} | time {:5.0f}s | valid'.format(
                epoch + 1, time.time() - start_time)
            for k, meter in valid_meters.items():
                log_output += ' {} {:.2f},'.format(k, meter.avg)
            if not best_val_loss or valid_meters['loss'].avg < best_val_loss:
                log_output += ' | saving model'
                ckpt = {'args': self.args, 'model': self.model.state_dict()}
                torch.save(ckpt, self.checkpoint_path)
                best_val_loss = valid_meters['loss'].avg
            print(log_output)
        print('Done training')
