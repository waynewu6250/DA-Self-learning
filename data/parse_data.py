"""Split the data for use"""
import torch
import json
import os
import gzip
import numpy as np
np.random.seed(42)

def get_vocab():

    model = torch.load('/home/ec2-user/raw_data/model.pth', map_location=torch.device('cpu'))
    use_fields = ['transcription', 'domain', 'intent', 'slotKey', 'interpretationRank', 'proposedSpeechlet']
    new_vocab = {key: model['vocab'][key] for key in use_fields}
    with open('/home/ec2-user/raw_data/vocab.json', 'w') as fp:
        json.dump(new_vocab, fp)

def split_data(folder_paths, save_dir):

    # Take all files
    all_files = []
    for folder_path in folder_paths:
        for file_path in sorted(os.listdir(folder_path)):
            file_path = os.path.join(folder_path, file_path)
            all_files.append(file_path)

    if not os.path.exists('/home/ec2-user/raw_data/wuti_data/{}/'.format(save_dir)):
        os.mkdir('/home/ec2-user/raw_data/wuti_data/{}/'.format(save_dir))

    counter = {}
    # domain_to_use = set([1, 2, 3, 4, 5])
    # domain_to_use = set([3, 5, 6, 15])
    domain_to_use = set([5, 6, 8, 10, 15])
    # domain_to_use = set([5]) # knowledge
    # domain_to_use = set([6]) # shopping
    # domain_to_use = set([15]) # books
    flag = False
    data_type = ['train', 'valid', 'test']
    data_dic = {i:t for i, t in enumerate(data_type)}
    data_file = [gzip.open('/home/ec2-user/raw_data/wuti_data/{}/{}.gz'.format(save_dir, d), 'wb') for d in data_type]

    for file_path in all_files:
        print(file_path)

        with gzip.open(file_path, 'rb') as f:
            for line in f:
                data_detail = json.loads(line)
                domain_id = data_detail['domain'][0] # select the top NLU interpretation domain
                if domain_id in domain_to_use: #and len(data_detail['transcription']) >= 4:
                    counter[domain_id] = counter.get(domain_id, 0) + 1
                    # Limit the domain number
                    if counter[domain_id] > 1e6:
                        continue
                    num = np.argmax(np.random.multinomial(1, [0.7, 0.1, 0.2], size=1))
                    data_file[num].write(line)
                    counter[data_dic[num]] = counter.get(data_dic[num], 0)+1
                # domain_set = set(data_detail['domain'])
                # if len(domain_set.intersection(domain_to_use)) != 0:
                #     for domain in domain_set:
                #         counter[domain] = counter.get(domain, 0)+1
                #         # if domain count is larger than a threshold, stop adding new utterance
                #         if counter[domain] > 500000:
                #             flag = True
                #     if flag:
                #         flag = False
                #         continue
                #     # sample the utterance into the file
                #     num = np.argmax(np.random.multinomial(1, [0.7, 0.1, 0.2], size=1))
                #     data_file[num].write(line)

    for file in data_file:
        file.close()

    for k, v in counter.items():
        print('{} Number: {}'.format(k, v))

def drop_to_text_file(fpath, vocab, opath, num_data):
    g = open(opath, 'w')
    with gzip.open(fpath, 'rb') as f:
        counter = 0
        for line in f:
            data_detail = json.loads(line)
            text = ' '.join([vocab['transcription']['itos'][word_id] for word_id in data_detail['transcription']])
            g.write(text + '\n')
            counter += 1
            if counter == num_data:
                break
    g.close()

if __name__ == '__main__':
    # Split data from raw gzip file
    folder_paths = ['/home/ec2-user/raw_data/train/', '/home/ec2-user/raw_data/valid/']
    save_dir = '5_6_8_10_15'
    print('Parsing data: ', save_dir)
    split_data(folder_paths, save_dir)

    # To drop data in a text file
    # vocab = json.load(open('/home/ec2-user/raw_data/vocab.json', 'r'))
    # drop_to_text_file('/home/ec2-user/raw_data/wuti_data/train.gz', vocab,
    #                   '/home/ec2-user/raw_data/wuti_data/train.txt', 200000)
    # drop_to_text_file('/home/ec2-user/raw_data/wuti_data/valid.gz', vocab,
    #                   '/home/ec2-user/raw_data/wuti_data/valid.txt', 10000)
    # drop_to_text_file('/home/ec2-user/raw_data/wuti_data/test.gz', vocab,
    #                   '/home/ec2-user/raw_data/wuti_data/test.txt', 10000)


