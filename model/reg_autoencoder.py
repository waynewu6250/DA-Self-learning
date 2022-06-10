import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import noisy

def reparameterize(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mu)

def log_prob(z, mu, logvar):
    var = torch.exp(logvar)
    logp = - (z-mu)**2 / (2*var) - torch.log(2*np.pi*var) / 2
    return logp.sum(dim=1)

def loss_kl(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / len(mu)


class TextModel(nn.Module):
    """Container module with word embedding and projection layers"""

    def __init__(self, dictionary, args, opt, device, new_vocab, initrange=0.1):
        super().__init__()
        self.dictionary = dictionary
        self.args = args
        self.opt = opt
        self.device = device
        self.new_vocab = new_vocab

        self.embed = nn.Embedding(new_vocab.size, opt.dim_emb)
        self.proj = nn.Linear(opt.dim_h, new_vocab.size)

        self.embed.weight.data.uniform_(-initrange, initrange)
        self.proj.bias.data.zero_()
        self.proj.weight.data.uniform_(-initrange, initrange)


class DAE(TextModel):
    """Denoising Auto-Encoder"""

    def __init__(self, dictionary, args, opt, device, new_vocab):
        super().__init__(dictionary, args, opt, device, new_vocab)
        self.drop = nn.Dropout(opt.dropout)
        self.E = nn.LSTM(opt.dim_emb, opt.dim_h, opt.nlayers,
            dropout=opt.dropout if opt.nlayers > 1 else 0, bidirectional=True)
        self.G = nn.LSTM(opt.dim_emb, opt.dim_h, opt.nlayers,
            dropout=opt.dropout if opt.nlayers > 1 else 0)
        self.h2mu = nn.Linear(opt.dim_h*2, opt.dim_z)
        self.h2logvar = nn.Linear(opt.dim_h*2, opt.dim_z)
        self.z2emb = nn.Linear(opt.dim_z, opt.dim_emb)
        self.optimizer = optim.Adam(self.parameters(), lr=opt.lr, betas=(0.5, 0.999))

        self.domain_embedding = nn.Embedding(dictionary['domain'], self.opt.dim_condition)
        self.intent_embedding = nn.Embedding(dictionary['intent'], self.opt.dim_condition)
        # no slot
        self.emb2h = nn.Linear(2 * self.opt.dim_condition, opt.dim_h)
        # add slot
        # self.slotkey_embedding = nn.Embedding(dictionary['slotKey'], self.opt.dim_condition)
        # self.emb2h = nn.Linear(3*self.opt.dim_condition, opt.dim_h)

    def flatten(self):
        self.E.flatten_parameters()
        self.G.flatten_parameters()

    def encode(self, input):
        input = self.drop(self.embed(input))
        _, (h, _) = self.E(input)
        h = torch.cat([h[-2], h[-1]], 1)
        return self.h2mu(h), self.h2logvar(h)

    def decode(self, z, input, batch_data, hidden=None):
        # Encode NLU interpretations
        domain_embed = self.domain_embedding(batch_data['domain'])  # batch x h_e
        intent_embed = self.intent_embedding(batch_data['intent'])  # batch x h_e

        # add slot
        # slot_embed = self.slotkey_embedding(batch_data['slotKey']).mean(dim=1)  # batch x h_e
        # hidden = torch.cat([domain_embed, intent_embed, slot_embed], dim=1)
        # no slot
        hidden = torch.cat([domain_embed, intent_embed], dim=1)

        h0 = self.emb2h(hidden).unsqueeze(0)
        c0 = torch.zeros_like(h0).to(self.device)
        hidden = (h0, c0)

        input = self.drop(self.embed(input)) + self.z2emb(z)
        output, hidden = self.G(input, hidden)
        output = self.drop(output)
        logits = self.proj(output.view(-1, output.size(-1)))
        return logits.view(output.size(0), output.size(1), -1), hidden

    def decode_one_step(self, z, input, hidden=None):
        input = self.drop(self.embed(input)) + self.z2emb(z)
        output, hidden = self.G(input, hidden)
        output = self.drop(output)
        logits = self.proj(output.view(-1, output.size(-1)))
        return logits.view(output.size(0), output.size(1), -1), hidden

    def generate(self, z, max_len, alg, batch_data):
        assert alg in ['greedy' , 'sample' , 'top5']
        sents = []
        input = torch.zeros(1, len(z), dtype=torch.long, device=z.device).fill_(self.new_vocab.go)
        # Encode NLU interpretations
        domain_embed = self.domain_embedding(batch_data['domain'])  # batch x h_e
        intent_embed = self.intent_embedding(batch_data['intent'])  # batch x h_e

        # add slot
        # slot_embed = self.slotkey_embedding(batch_data['slotKey']).mean(dim=1)  # batch x h_e
        # hidden = torch.cat([domain_embed, intent_embed, slot_embed], dim=1)
        # no slot
        hidden = torch.cat([domain_embed, intent_embed], dim=1)

        h0 = self.emb2h(hidden).unsqueeze(0)
        c0 = torch.zeros_like(h0).to(self.device)
        hidden = (h0, c0)

        for l in range(max_len):
            sents.append(input)
            logits, hidden = self.decode_one_step(z, input, hidden)
            if alg == 'greedy':
                input = logits.argmax(dim=-1)
            elif alg == 'sample':
                input = torch.multinomial(logits.squeeze(dim=0).exp(), num_samples=1).t()
            elif alg == 'top5':
                not_top5_indices=logits.topk(logits.shape[-1]-5,dim=2,largest=False).indices
                logits_exp=logits.exp()
                logits_exp[:,:,not_top5_indices]=0.
                input = torch.multinomial(logits_exp.squeeze(dim=0), num_samples=1).t()
        return torch.cat(sents)

    def forward(self, input, batch_data, is_train=False):
        _input = noisy(self.new_vocab, input, *self.args.noise) if is_train else input
        mu, logvar = self.encode(_input)
        z = reparameterize(mu, logvar)
        logits, _ = self.decode(z, input, batch_data)
        return mu, logvar, z, logits

    def loss_rec(self, logits, targets):
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
            ignore_index=self.new_vocab.pad, reduction='none').view(targets.size())
        return loss.sum(dim=0)

    def loss(self, losses):
        return losses['rec']

    def autoenc(self, inputs, targets, batch_data, is_train=False):
        _, _, _, logits = self(inputs, batch_data, is_train)
        return {'rec': self.loss_rec(logits, targets).mean()}

    def step(self, losses):
        self.optimizer.zero_grad()
        losses['loss'].backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        #nn.utils.clip_grad_norm_(self.parameters(), clip)
        self.optimizer.step()

    def nll_is(self, inputs, targets, batch_data, m):
        """compute negative log-likelihood by importance sampling:
           p(x;theta) = E_{q(z|x;phi)}[p(z)p(x|z;theta)/q(z|x;phi)]
        """
        mu, logvar = self.encode(inputs)
        tmp = []
        for _ in range(m):
            z = reparameterize(mu, logvar)
            logits, _ = self.decode(z, inputs, batch_data)
            v = log_prob(z, torch.zeros_like(z), torch.zeros_like(z)) - \
                self.loss_rec(logits, targets) - log_prob(z, mu, logvar)
            tmp.append(v.unsqueeze(-1))
        ll_is = torch.logsumexp(torch.cat(tmp, 1), 1) - np.log(m)
        return -ll_is


class RegVAE(DAE):
    """Variational Auto-Encoder"""

    def __init__(self, dictionary, args, opt, device, new_vocab):
        super().__init__(dictionary, args, opt, device, new_vocab)

    def loss(self, losses):
        return losses['rec'] + self.opt.lambda_kl * losses['kl']

    def autoenc(self, inputs, targets, batch_data, is_train=False):
        mu, logvar, _, logits = self(inputs, batch_data, is_train)
        return {'rec': self.loss_rec(logits, targets).mean(),
                'kl': loss_kl(mu, logvar)}


class AAE(DAE):
    """Adversarial Auto-Encoder"""

    def __init__(self, dictionary, args, opt, device, new_vocab):
        super().__init__(dictionary, args, opt, device, new_vocab)
        self.D = nn.Sequential(nn.Linear(opt.dim_z, opt.dim_d), nn.ReLU(),
            nn.Linear(opt.dim_d, 1), nn.Sigmoid())
        self.optD = optim.Adam(self.D.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    def loss_adv(self, z):
        zn = torch.randn_like(z)
        zeros = torch.zeros(len(z), 1, device=z.device)
        ones = torch.ones(len(z), 1, device=z.device)
        loss_d = F.binary_cross_entropy(self.D(z.detach()), zeros) + \
            F.binary_cross_entropy(self.D(zn), ones)
        loss_g = F.binary_cross_entropy(self.D(z), ones)
        return loss_d, loss_g

    def loss(self, losses):
        return losses['rec'] + self.opt.lambda_adv * losses['adv'] + \
            self.opt.lambda_p * losses['|lvar|']

    def autoenc(self, inputs, targets, batch_data, is_train=False):
        _, logvar, z, logits = self(inputs, batch_data, is_train)
        loss_d, adv = self.loss_adv(z)
        return {'rec': self.loss_rec(logits, targets).mean(),
                'adv': adv,
                '|lvar|': logvar.abs().sum(dim=1).mean(),
                'loss_d': loss_d}

    def step(self, losses):
        super().step(losses)

        self.optD.zero_grad()
        losses['loss_d'].backward()
        self.optD.step()