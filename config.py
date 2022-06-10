import argparse
import os

class Config:

    # raw data
    train_path = '/home/ec2-user/raw_data/train/'
    valid_path = '/home/ec2-user/raw_data/valid/'

    # subsample data
    vocab_path = '/home/ec2-user/raw_data/vocab.json'

    use_fields = ['uid', 'transcription', 'domain', 'intent', 'slotKey', 'interpretationRank', 'proposedSpeechlet', 'hypRankModelScores']

    # trainer
    trainer_mapping = {'cvae': 'CVAETrainer',
                       'autoencoder': 'AETrainer',
                       'adv_autoencoder': 'AETrainer'}

    train_batch_size = 256
    valid_batch_size = 256
    test_batch_size = 256
    max_len = 35

    # cvae config
    # epochs = 10
    # lr = 1e-4
    # bidirectional = True
    # encoder_hidden_size = 512
    # z_size = 128
    # n_encoder_layers = 2
    # decoder_hidden_size = 512
    # habits_lambda = 0.2

    # anneal_function = 'logistic'
    # k = 0.0025
    # x0 = 2500
    # print_every = 200
    # epochs = 30
    # lr = 1e-3
    # embedding_size = 300
    # condition_size = 16
    # hidden_size = 256
    # word_dropout = 0
    # embedding_dropout = 0.5
    # latent_size = 16
    # num_layers = 1
    # bidirectional = True
    # rnn_type = 'gru'

    vocab_size = 10000
    dim_z = 128
    dim_condition = 128
    dim_emb = 512
    dim_h = 1024
    nlayers = 1
    dim_d = 512

    lambda_kl = 0.1
    lambda_adv = 10
    lambda_p = 0.01
    dropout = 0.5
    lr = 0.0005
    epochs = 30
    seed = 1111
    log_interval = 100
    enc = 'mu' # ['mu', 'z']
    dec = 'greedy' # ['greedy', 'sample']
    m_importance = 100






def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--type', default='train', dest='type', help='train | evaluate | generate | visualize')
    parser.add_argument('-m', '--model', default='autoencoder', dest='model', help='autoencoder')
    parser.add_argument('-g', '--generate_type', default='data', dest='generate_type', help='data | file')
    parser.add_argument('-p', '--postfix', default='', dest='postfix', help='postfix of checkpoint path')
    parser.add_argument('-n', '--num_samples', type=int, default=10000)
    parser.add_argument('--num_data', type=int, default=1000000)
    parser.add_argument('--data', metavar='FILE',
                        help='path to data file')
    parser.add_argument('--noise', default='0,0,0,0', metavar='P,P,P,K',
                        help='word drop prob, blank prob, substitute prob'
                             'max word shuffle distance')
    parser.add_argument('--data-folder', default='', help='data folder')
    # uncomment here to choose a single domain to use in train/valid/test
    # parser.add_argument('--domainuse', type=int , default=0)
    args = parser.parse_args()

    assert args.type in ['train', 'evaluate', 'generate', 'visualize']
    args.noise = [float(x) for x in args.noise.split(',')]

    opt = Config()
    opt.checkpoint_path = '/home/ec2-user/checkpoints/{}/{}best_model_{}{}.pth'.format(args.model, args.data_folder, args.num_data, args.postfix)
    opt.output_path = '/home/ec2-user/checkpoints/{}/{}sample_{}{}'.format(args.model, args.data_folder, args.num_data,args.postfix)

    opt.vocab_file = os.path.join('/home/ec2-user/checkpoints/{}/{}'.format(args.model, args.data_folder),'vocab.txt')  # 'vocab_d1.5M_v25k.txt'
    opt.sub_train_path = '/home/ec2-user/raw_data/wuti_data/{}train.gz'.format(args.data_folder)
    opt.sub_valid_path = '/home/ec2-user/raw_data/wuti_data/{}valid.gz'.format(args.data_folder)
    opt.sub_test_path = '/home/ec2-user/raw_data/wuti_data/{}test.gz'.format(args.data_folder)

    return args, opt
