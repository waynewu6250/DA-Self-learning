import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter
from functools import wraps
import numpy as np

MAX_SAMPLE = False
TRUNCATED_SAMPLE = True
model_random_state = np.random.RandomState(1988)
torch.manual_seed(1999)

# Encoder
# ------------------------------------------------------------------------------

# Encode into Z with mu and log_var

class EncoderRNN:
    def __init__(self, input_size, opt, device):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = opt.encoder_hidden_size
        self.output_size = opt.z_size
        self.n_layers = opt.n_encoder_layers
        self.bidirectional = opt.bidirectional
        self.device = device

        self.embed = nn.Embedding(input_size, self.hidden_size)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, self.n_layers, dropout=0.1, bidirectional=self.bidirectional)
        self.o2p = nn.Linear(self.hidden_size, self.output_size * 2)

    def sample(self, mu, logvar):
        eps = Variable(torch.randn(mu.size())).to(self.device)
        std = torch.exp(logvar / 2.0)
        return mu + eps * std

    def forward(self, batch_data):
        embedded = self.embed(batch_data['transcription']).unsqueeze(1) # batch x T x e_hidden

        output, hidden = self.gru(embedded, None)
        #output = torch.mean(output, 0).squeeze(0) #output[-1]
        output = output[:, -1] # Take only the last value
        if self.bidirectional:
            output = output[:, :, :self.hidden_size] + output[:, :,self.hidden_size:] # Sum bidirectional outputs
        else:
            output = output[:, :, :self.hidden_size]

        ps = self.o2p(output)
        mu, logvar = torch.chunk(ps, 2, dim=1)
        z = self.sample(mu, logvar)
        return mu, logvar, z

# Decoder
# ------------------------------------------------------------------------------

# Decode from Z into sequence

class DecoderRNN(nn.Module):
    def __init__(self, z_size, n_conditions, condition_size, hidden_size, output_size, n_layers=1, word_dropout=1.):
        super(DecoderRNN, self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.word_dropout = word_dropout

        input_size = z_size + condition_size

        self.embed = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size + input_size, hidden_size, n_layers)
        self.i2h = nn.Linear(input_size, hidden_size)
        if n_conditions > 0 and condition_size > 0 and n_conditions != condition_size:
            self.c2h = nn.Linear(n_conditions, condition_size)
        #self.dropout = nn.Dropout()
        self.h2o = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size + input_size, output_size)

        print(f'MAX_SAMPLE: {MAX_SAMPLE}; TRUNCATED_SAMPLE: {TRUNCATED_SAMPLE}')

    def sample(self, output, temperature, device, max_sample=MAX_SAMPLE, trunc_sample=TRUNCATED_SAMPLE):
        if max_sample:
            # Sample top value only
            top_i = output.data.topk(1)[1].item()
        else:
            # Sample from the network as a multinomial distribution
            if trunc_sample:
                # Sample from top k values only
                k = 10
                new_output = torch.empty_like(output).fill_(float('-inf'))
                top_v, top_i = output.data.topk(k)
                new_output.data.scatter_(1, top_i, top_v)
                output = new_output

            output_dist = output.data.view(-1).div(temperature).exp()
            if len(torch.nonzero(output_dist)) > 0:
                top_i = torch.multinomial(output_dist, 1)[0]
            else:
                # TODO: how does this happen?
                print(f'[WARNING] output_dist is all zeroes')
                top_i = UNK_token

        input = Variable(torch.LongTensor([top_i])).to(device)
        return input, top_i

    def forward(self, z, condition, inputs, temperature, device):
        n_steps = inputs.size(0)
        outputs = Variable(torch.zeros(n_steps, 1, self.output_size)).to(device)

        input = Variable(torch.LongTensor([SOS_token])).to(device)
        if condition is None:
            decode_embed = z
        else:
            if hasattr(self, 'c2h'):
                #squashed_condition = self.c2h(self.dropout(condition))
                squashed_condition = self.c2h(condition)
                decode_embed = torch.cat([z, squashed_condition], 1)
            else:
                decode_embed = torch.cat([z, condition], 1)


        hidden = self.i2h(decode_embed).unsqueeze(0).repeat(self.n_layers, 1, 1)

        for i in range(n_steps):
            output, hidden = self.step(i, decode_embed, input, hidden)
            outputs[i] = output

            use_word_dropout = model_random_state.rand() < self.word_dropout
            if use_word_dropout and i < (n_steps - 1):
                unk_input = Variable(torch.LongTensor([UNK_token])).to(device)
                input = unk_input
                continue

            use_teacher_forcing = model_random_state.rand() < temperature
            if use_teacher_forcing:
                input = inputs[i]
            else:
                input, top_i = self.sample(output, temperature, device, max_sample=True)

            if input.dim() == 0:
                input = input.unsqueeze(0)

        return outputs.squeeze(1)

    def generate_with_embed(self, embed, n_steps, temperature, device, max_sample=MAX_SAMPLE, trunc_sample=TRUNCATED_SAMPLE):
        outputs = Variable(torch.zeros(n_steps, 1, self.output_size)).to(device)
        input = Variable(torch.LongTensor([SOS_token])).to(device)

        hidden = self.i2h(embed).unsqueeze(0).repeat(self.n_layers, 1, 1)

        for i in range(n_steps):
            output, hidden = self.step(i, embed, input, hidden)
            outputs[i] = output
            input, top_i = self.sample(output, temperature, device, max_sample=max_sample, trunc_sample=trunc_sample)
            #if top_i == EOS: break
        return outputs.squeeze(1)

    def generate(self, z, condition, n_steps, temperature, device, max_sample=MAX_SAMPLE, trunc_sample=TRUNCATED_SAMPLE):
        if condition is None:
            decode_embed = z
        else:
            if condition.dim() == 1:
                condition = condition.unsqueeze(0)

            if hasattr(self, 'c2h'):
                #squashed_condition = self.c2h(self.dropout(condition))
                squashed_condition = self.c2h(condition)
                decode_embed = torch.cat([z, squashed_condition], 1)
            else:
                decode_embed = torch.cat([z, condition], 1)

        return self.generate_with_embed(decode_embed, n_steps, temperature, device, max_sample, trunc_sample)

    def step(self, s, decode_embed, input, hidden):
        # print('[DecoderRNN.step] s =', s, 'decode_embed =', decode_embed.size(), 'i =', input.size(), 'h =', hidden.size())
        input = F.relu(self.embed(input))
        input = torch.cat((input, decode_embed), 1)
        input = input.unsqueeze(0)
        output, hidden = self.gru(input, hidden)
        output = output.squeeze(0)
        output = torch.cat((output, decode_embed), 1)
        output = self.out(output)
        return output, hidden

# Container
# ------------------------------------------------------------------------------

class VAE(nn.Module):
    def __init__(self, encoder, decoder, n_steps=None):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.register_buffer('steps_seen', torch.tensor(0, dtype=torch.long))
        self.register_buffer('kld_max', torch.tensor(1.0, dtype=torch.float))
        self.register_buffer('kld_weight', torch.tensor(0.0, dtype=torch.float))
        if n_steps is not None:
            self.register_buffer('kld_inc', torch.tensor((self.kld_max - self.kld_weight) / (n_steps // 2), dtype=torch.float))
        else:
            self.register_buffer('kld_inc', torch.tensor(0, dtype=torch.float))

    def encode(self, batch_data):
        mu, logvar, z = self.encoder(batch_data)
        return mu, logvar, z

    def forward(self, batch_data, batch_length, targets, condition, device, temperature=1.0):
        mu, logvar, z = self.encoder(batch_data)
        decoded = self.decoder(z, condition, targets, temperature, device)
        return mu, logvar, z, decoded