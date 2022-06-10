import os
import numpy as np
import torch

def create_folder(args):

    if not os.path.exists('/home/ec2-user/checkpoints/'):
        os.mkdir('/home/ec2-user/checkpoints/')
    if not os.path.exists('/home/ec2-user/checkpoints/' + args.model + '/'):
        os.mkdir('/home/ec2-user/checkpoints/' + args.model + '/')
    if not os.path.exists('/home/ec2-user/checkpoints/' + args.model  + '/' + args.data_folder):
        os.mkdir('/home/ec2-user/checkpoints/' + args.model  + '/' + args.data_folder)

def idx2word(idx, i2w, pad_idx):
    sent_str = [str()]*len(idx)
    for i, sent in enumerate(idx):
        for word_id in sent:
            if word_id == pad_idx:
                break
            sent_str[i] += i2w[word_id.item()] + " "
        sent_str[i] = sent_str[i].strip()
    return sent_str

def lerp(t, p, q):
    return (1-t) * p + t * q

def interpolate(z1, z2, n):
    z = []
    for i in range(n):
        zi = lerp(1.0*i/(n-1), z1, z2)
        z.append(np.expand_dims(zi, axis=0))
    return np.concatenate(z, axis=0)

##################################################

def idx2sent(decoded_index, idx2word):
    decoded_sents = []
    for s in decoded_index:
        decoded_sents.append([idx2word[id] for id in s])
    decoded_sents = [sent[:sent.index('<eos>')] if '<eos>' in sent else sent
                     for sent in decoded_sents]
    return decoded_sents

def load_sent(path):
    sents = []
    with open(path) as f:
        for line in f:
            sents.append(line.split())
    return sents

def load_NLU(path):
    domains = []
    intents = []
    with open(path) as f:
        for line in f:
            element = line.rstrip('\n').split(' ')
            domains.append(element[0])
            intents.append(element[1])
    return domains, intents

def write_sent(path, *args):
    with open(path, 'w') as f:
        f.write('{:<40} | {:<20} | {:<30} | {:<40}'.format('Test text', 'Domain', 'Intent', 'Gen Text') + '\n')
        f.write('-'*140+'\n')
        for org_s, d, i, s in zip(*args):
            f.write('{:<40} | {:<20} | {:<30} | {:<40}'.format(' '.join(org_s), str(d), str(i), ' '.join(s)) + '\n')

def write_doc(docs, path):
    with open(path, 'w') as f:
        for d in docs:
            for s in d:
                f.write(' '.join(s) + '\n')
            f.write('\n')

def get_batch(x, vocab, device):
    go_x, x_eos = [], []
    max_len = max([len(s) for s in x])
    for s in x:
        s_idx = [vocab.word2idx[w] if w in vocab.word2idx else vocab.unk for w in s]
        padding = [vocab.pad] * (max_len - len(s))
        go_x.append([vocab.go] + s_idx + padding)
        x_eos.append(s_idx + [vocab.eos] + padding)
    return torch.LongTensor(go_x).t().contiguous().to(device), \
           torch.LongTensor(x_eos).t().contiguous().to(device)  # time * batch

def get_batches(data, vocab, batch_size, device):
    order = range(len(data))
    z = sorted(zip(order, data), key=lambda i: len(i[1]))
    order, data = zip(*z)

    batches = []
    i = 0
    while i < len(data):
        j = i
        while j < min(len(data), i+batch_size) and len(data[j]) == len(data[i]):
            j += 1
        batches.append(get_batch(data[i: j], vocab, device))
        i = j
    return batches, order


# ==========================================================================
# Common querying helper utilities
# ==========================================================================
def get_dict_field(nested_dict, path, raise_exception=False, na_value=None):
    """
    Look for a field in a nested dictionary and if it is not found return which
    field was not found and where it was expected.
    Extremely useful for finding out why some fields are not in some jsons
    and for avoiding throwing exceptions every time a key is not found.

    :param nested_dict: A nested dictionary containing the path.
    :param path: A list of values such as ["value", "name"] representing nested_dict["value"]["name"]
    :param raise_exception: Throw and exception if set to True and a field in the path is not found.
    :param na_value: The text to return if the field is not found. By default this is the value None
    :return: The object from following the path if it is found, otherwise the na_value. KeyError will be raised
    if the field is not found and raise_exception=True
    """
    cur_object = nested_dict

    for index, next_field in enumerate(path):
        try:
            cur_object = cur_object[next_field]

        except (KeyError, TypeError, IndexError) as e:
            if na_value:
                exception_message = na_value
            elif index > 0:
                exception_message = "NONE: Could not find \"{}\" in \"{}\"".format(next_field, path[index - 1])
            else:
                exception_message = "NONE: Could not find \"{}\"".format(next_field)

            if raise_exception:
                raise KeyError("{}\n{}".format(exception_message, str(e)))
            else:
                return na_value

    return cur_object

# ==========================================================================
# Generate noises
# ==========================================================================

def word_shuffle(vocab, x, k):   # slight shuffle such that |sigma[i]-i| <= k
    base = torch.arange(x.size(0), dtype=torch.float).repeat(x.size(1), 1).t()
    inc = (k+1) * torch.rand(x.size())
    inc[x == vocab.go] = 0     # do not shuffle the start sentence symbol
    inc[x == vocab.pad] = k+1  # do not shuffle end paddings
    _, sigma = (base + inc).sort(dim=0)
    return x[sigma, torch.arange(x.size(1))]

def word_drop(vocab, x, p):     # drop words with probability p
    x_ = []
    for i in range(x.size(1)):
        words = x[:, i].tolist()
        keep = np.random.rand(len(words)) > p
        keep[0] = True  # do not drop the start sentence symbol
        sent = [w for j, w in enumerate(words) if keep[j]]
        sent += [vocab.pad] * (len(words)-len(sent))
        x_.append(sent)
    return torch.LongTensor(x_).t().contiguous().to(x.device)

def word_blank(vocab, x, p):     # blank words with probability p
    blank = (torch.rand(x.size(), device=x.device) < p) & \
        (x != vocab.go) & (x != vocab.pad)
    x_ = x.clone()
    x_[blank] = vocab.blank
    return x_

def word_substitute(vocab, x, p):     # substitute words with probability p
    keep = (torch.rand(x.size(), device=x.device) > p) | \
        (x == vocab.go) | (x == vocab.pad)
    x_ = x.clone()
    x_.random_(vocab.nspecial, vocab.size)
    x_[keep] = x[keep]
    return x_

def noisy(vocab, x, drop_prob, blank_prob, sub_prob, shuffle_dist):
    if shuffle_dist > 0:
        x = word_shuffle(vocab, x, shuffle_dist)
    if drop_prob > 0:
        x = word_drop(vocab, x, drop_prob)
    if blank_prob > 0:
        x = word_blank(vocab, x, blank_prob)
    if sub_prob > 0:
        x = word_substitute(vocab, x, sub_prob)
    return x