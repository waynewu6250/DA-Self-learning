from time import time
import torch
import os
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam, RMSprop

from dataset import get_vocab, get_dataloader, prepare_data
from config import get_args
from trainer import CVAETrainer, AETrainer
from augmentor import Augmentor
from evaluator import Evaluator
from utils import *

def train_model(args, opt, checkpoint_path):

    assert args.model in ['cvae', 'autoencoder', 'adv_autoencoder']
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.backends.cudnn.enabled = False

    # create folder
    create_folder(args)

    # get data
    train_dataloader, valid_dataloader, test_dataloader, vocab, new_vocab = prepare_data(args, opt, device)

    # sanity check
    # for batch_data, batch_length in train_dataloader:
    #     print(batch_data['transcription'].shape)
    #     print(batch_data['target'].shape)
    #     print(batch_data['domain'].shape)
    #     print(batch_data['intent'].shape)
    #     print(batch_data['slotKey'].shape)
    #     print(batch_data['interpretationRank'].shape)
    #     print(batch_data['proposedSpeechlet'].shape)
    #     print(batch_data['hypRankModelScores'].shape)
    #     dd
    
    print('Trainer used: ', args.model)
    trainer = eval(opt.trainer_mapping[args.model])(args, opt, device, vocab, new_vocab, checkpoint_path)
    trainer.train(train_dataloader, valid_dataloader)

def evaluate(args, opt, checkpoint_path):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.backends.cudnn.enabled = False

    # check if folder exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)

    # get test data
    vocab = get_vocab(opt)
    test_dataloader = get_dataloader(opt.sub_test_path, args, opt, device, 'test', opt.test_batch_size, vocab, 10000)

    print('Evaluator used: ', args.model)
    evaluator = Evaluator(args, opt, device, vocab, test_dataloader.dataset.new_vocab, checkpoint_path)
    evaluator.evaluate(test_dataloader)


def generate_data(args, opt, checkpoint_path, output_path):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.backends.cudnn.enabled = False

    # check if folder exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)

    # get test data
    vocab = get_vocab(opt)
    test_dataloader = get_dataloader(opt.sub_test_path, args, opt, device, 'test', opt.test_batch_size, vocab, 10000)

    print('Augmentor used: ', args.model)
    augmentor = Augmentor(args, opt, device, vocab, test_dataloader.dataset.new_vocab, checkpoint_path, output_path)
    augmentor.generate(test_dataloader)


if __name__ == '__main__':

    # Get arguments
    args, opt = get_args()
    print('Model to use: ', opt.checkpoint_path)

    if args.type == 'train':
        train_model(args, opt, opt.checkpoint_path)
    elif args.type == 'evaluate':
        evaluate(args, opt, opt.checkpoint_path)
    elif args.type == 'generate':
        generate_data(args, opt, opt.checkpoint_path, opt.output_path)



