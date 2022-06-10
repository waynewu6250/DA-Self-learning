import numpy as np
import pandas as pd
from collections import defaultdict, Counter

import json
import csv
import pickle
import os
import gzip
import glob
from functools import partial

import torch
from torch.utils.data import DataLoader, Dataset

class Vocab(object):
    def __init__(self, path):
        self.word2idx = {}
        self.idx2word = []

        with open(path) as f:
            for line in f:
                w = line.split()[0]
                self.word2idx[w] = len(self.word2idx)
                self.idx2word.append(w)
        self.size = len(self.word2idx)

        self.pad = self.word2idx['<pad>']
        self.go = self.word2idx['<go>']
        self.eos = self.word2idx['<eos>']
        self.unk = self.word2idx['<unk>']
        self.blank = self.word2idx['<blank>']
        self.nspecial = 5

    @staticmethod
    def build(sents, path, size, old_vocab):
        v = ['<pad>', '<go>', '<eos>', '<unk>', '<blank>']
        sents = [sent['transcription'] for sent in sents]
        words = [old_vocab['transcription']['itos'][idx] for s in sents for idx in s]
        cnt = Counter(words)
        n_unk = len(words)
        for w, c in cnt.most_common(size):
            v.append(w)
            n_unk -= c
        cnt['<unk>'] = n_unk

        with open(path, 'w') as f:
            for w in v:
                f.write('{}\t{}\n'.format(w, cnt[w]))


class HypDataset(Dataset):

    def __init__(self, fpath, args, opt, device, vocab, num_data=float('inf')):
        """
        Store in meta docstring (like previous pipeline)
        dictionary: {'ID', 'text', 'NLU', 'Other features'}
        @param fpath: raw data path
        @param args: arguments
        @param opt: configurations
        @param device: torch device
        """
        self.args = args
        self.opt = opt
        self.device = device
        self.vocab = vocab
        self.num_data = num_data
        self.read_data(fpath)
        # torch.save(self.raw_data, '/home/ec2-user/raw_data/tmp_raw.pth')
        # torch.save(self.filter_data, '/home/ec2-user/raw_data/tmp_filter.pth')
        if not os.path.isfile(opt.vocab_file):
            Vocab.build(self.filter_data, opt.vocab_file, opt.vocab_size, self.vocab)
            print('Process new vocab file at: ', opt.vocab_file)
        self.new_vocab = Vocab(opt.vocab_file)

    def __getitem__(self, index):
        item_info = self.filter_data[index]
        return item_info

    def __len__(self):
        return len(self.filter_data)

    def read_single_file(self, file_path):
        with gzip.open(file_path, 'rb') as f:
            counter = 0
            domain_counter = defaultdict(int)
            for line in f:
                data_detail = json.loads(line)
                # self.raw_data[data_detail['uid']] = data_detail

                # uncomment here to filter domain to use in train/valid/test
                # if data_detail['domain'][0] == self.args.domainuse:

                # uncomment here to balance the data
                # domain_id = data_detail['domain'][0]  # select the top NLU interpretation domain
                # if domain_counter[domain_id] <= self.num_data / 5:
                #     self.filter_data.append({k: v for k, v in data_detail.items() if k in self.opt.use_fields})
                # if sum(list(domain_counter.values())) > self.num_data:
                #     break

                # normal case
                self.filter_data.append({k: v for k, v in data_detail.items() if k in self.opt.use_fields})
                counter += 1
                if counter == self.num_data:
                    break

    def read_data(self, fpath):
        self.raw_data = {}
        self.filter_data = []

        if os.path.isdir(fpath):
            for file_path in sorted(os.listdir(fpath)):
                file_path = os.path.join(fpath, file_path)
                print('Process: ', file_path)
                self.read_single_file(file_path)
        else:
            print('Process: ', fpath)
            self.read_single_file(fpath)


##############################################################################################################################

def compute_shape(nested_list):
    element_max_dims = []
    overall_max_dims = [len(nested_list)]
    lengths = []

    if len(nested_list) == 0:
        return 0, [0]
    if not isinstance(nested_list[0], list):
        return len(nested_list), overall_max_dims

    for inner_list in nested_list:
        inner_lengths, inner_max_dims = compute_shape(inner_list)
        lengths.append(inner_lengths)
        element_max_dims.append(inner_max_dims)

    for d in range(len(element_max_dims[0])):
        overall_max_dims.append(max([inner_max_dims[d] for inner_max_dims in element_max_dims]))

    return lengths, overall_max_dims


def set_values(nested_list, tensor):
    if len(nested_list) == 0:
        return
    if not isinstance(nested_list[0], list):
        tensor[: len(nested_list)] = torch.Tensor(nested_list)
        return
    for i, inner_list in enumerate(nested_list):
        set_values(inner_list, tensor[i])


def pad(minibatch, new_vocab):
    """Pad a batch of examples using this field.
    # TODO make sure proper padding is being done
    """
    minibatch = list(minibatch)
    lengths, max_dims = compute_shape(minibatch)
    max_dims = [max(1, dim) for dim in max_dims]  # Pad if all examples are empty.

    pad_value = new_vocab.pad
    padded_minibatch = np.full(shape=max_dims, fill_value=pad_value)
    padded_lengths = np.full(shape=max_dims[:-1], fill_value=0, dtype=np.int64)

    set_values(minibatch, padded_minibatch)
    if len(max_dims) > 1:
        set_values(lengths, padded_lengths)
    else:
        padded_lengths[()] = lengths

    return (padded_minibatch, padded_lengths)

def collate_fn(batch, opt, vocab, new_vocab):
    """collate function for batching the inputs"""

    # Each field will have a batch of items, ex. Intent: batch x hyp
    # utterance: batch x T
    # labels: batch x hyp
    # slot tags: batch x hyp x slot
    batch_data = {}
    batch_length = {}
    for key in batch[0]:
        # Do some preprocessing on transcription
        if key == 'transcription':
            raw_input = []
            raw_target = []
            for b in batch:
                raw_text = [vocab['transcription']['itos'][idx] for idx in b[key]]  # map into raw texts
                new_ids = [new_vocab.word2idx[w] if w in new_vocab.word2idx else new_vocab.unk for w in
                           raw_text]  # turn into new ids

                input = [new_vocab.go] + new_ids
                input = input[:opt.max_len]
                target = new_ids[:opt.max_len - 1]
                target = target + [new_vocab.eos]
                assert len(input) == len(target), "%i, %i" % (len(input), len(target))
                raw_input.append(input)
                raw_target.append(target)

            data_input, length_input = pad(raw_input, new_vocab)
            data_output, length_output = pad(raw_target, new_vocab)
            batch_data[key] = torch.tensor(data_input)
            batch_length[key] = torch.tensor(length_input)
            batch_data['target'] = torch.tensor(data_output)
            batch_length['target'] = torch.tensor(length_output)
        # uids are strings
        elif key == 'uid':
            raw_batch = [b[key] for b in batch]
            batch_data[key] = raw_batch
        # other id fields
        else:
            raw_batch = [b[key] for b in batch]
            data, length = pad(raw_batch, new_vocab)
            batch_data[key] = torch.tensor(data)
            batch_length[key] = torch.tensor(length)

    # Augment top NLU interpretations first (single hypothesis)
    # TODO: Take top NLU only, filter data
    for key, value in batch_data.items():
        if key not in ['uid', 'transcription', 'target']:
            batch_data[key] = value[:, 0]

    return batch_data, batch_length

def get_vocab(opt):
    vocab = json.load(open(opt.vocab_path, 'r'))
    print('Loading vocab...')
    # Add <sos> and <eos> token:
    vocab['transcription']['itos'].append('<sos>')
    vocab['transcription']['stoi']['<sos>'] = len(vocab['transcription']['stoi'])
    vocab['transcription']['itos'].append('<eos>')
    vocab['transcription']['stoi']['<eos>'] = len(vocab['transcription']['stoi'])
    return vocab

def get_dataloader(fpath, args, opt, device, type, batch_size, vocab, num_data=float('inf')):
    """Get the torch dataloader given a file path"""
    dataset = HypDataset(fpath, args, opt, device, vocab, num_data)
    print('# {} utterances: {}'.format(type, len(dataset)))

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True if type == 'train' else False,
                            collate_fn=partial(collate_fn, opt=opt, vocab=vocab, new_vocab=dataset.new_vocab), num_workers=16)
    return dataloader

def prepare_data(args, opt, device):

    # get vocab
    vocab = get_vocab(opt)

    # get dataloader
    train_dataloader = get_dataloader(opt.sub_train_path, args, opt, device, 'train', opt.train_batch_size, vocab, args.num_data)
    valid_dataloader = get_dataloader(opt.sub_valid_path, args, opt, device, 'valid', opt.valid_batch_size, vocab, 10000)
    test_dataloader = get_dataloader(opt.sub_test_path, args, opt, device, 'test', opt.test_batch_size, vocab, 10000)

    return train_dataloader, valid_dataloader, test_dataloader, vocab, train_dataloader.dataset.new_vocab








