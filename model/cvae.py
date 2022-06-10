import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch.autograd import Variable
MAX_SAMPLE = True
TRUNCATED_SAMPLE = True

class CVAE(nn.Module):
    def __init__(self, vocab_size, opt, device, vocab):

        super().__init__()
        self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
        self.opt = opt
        self.device = device
        self.vocab = vocab
        self.n_words = vocab_size['transcription']

        self.sos_idx = vocab['transcription']['stoi']['<sos>']
        self.eos_idx = vocab['transcription']['stoi']['<eos>']
        self.pad_idx = vocab['transcription']['stoi']['<pad>']
        self.unk_idx = vocab['transcription']['stoi']['<unk>']

        self.latent_size = self.opt.latent_size

        self.rnn_type = self.opt.rnn_type
        self.bidirectional = self.opt.bidirectional
        self.num_layers = self.opt.num_layers
        self.hidden_size = self.opt.hidden_size

        self.embedding = nn.Embedding(vocab_size['transcription'], self.opt.embedding_size)
        self.word_dropout_rate = self.opt.word_dropout
        self.embedding_dropout = nn.Dropout(p=self.opt.embedding_dropout)

        self.domain_embedding = nn.Embedding(vocab_size['domain'], self.opt.condition_size)
        self.intent_embedding = nn.Embedding(vocab_size['intent'], self.opt.condition_size)
        self.slotkey_embedding = nn.Embedding(vocab_size['slotKey'], self.opt.condition_size)

        if self.opt.rnn_type == 'rnn':
            rnn = nn.RNN
        elif self.opt.rnn_type == 'gru':
            rnn = nn.GRU
        elif self.opt.rnn_type == 'lstm':
            rnn = nn.LSTM
        else:
            raise ValueError()

        self.encoder_rnn = rnn(self.opt.embedding_size, self.hidden_size, num_layers=self.num_layers, bidirectional=self.bidirectional,
                               batch_first=True)
        self.decoder_rnn = rnn(self.opt.embedding_size, self.hidden_size, num_layers=self.num_layers, bidirectional=self.bidirectional,
                               batch_first=True)

        self.hidden_factor = (2 if self.bidirectional else 1) * self.num_layers

        self.hidden2mean = nn.Linear(self.hidden_size * self.hidden_factor, self.latent_size)
        self.hidden2logv = nn.Linear(self.hidden_size * self.hidden_factor, self.latent_size)
        self.latent2hidden = nn.Linear(self.latent_size + 3 * self.opt.condition_size, self.hidden_size * self.hidden_factor)
        self.outputs2vocab = nn.Linear(self.hidden_size * (2 if self.bidirectional else 1), vocab_size['transcription'])

    def forward(self, batch_data, batch_length):

        batch_size = batch_data['transcription'].size(0)
        sorted_lengths, sorted_idx = torch.sort(batch_length['transcription'], descending=True)
        input_sequence = batch_data['transcription'][sorted_idx]

        # ENCODER
        input_embedding = self.embedding(input_sequence) # b x t x h

        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)

        _, hidden = self.encoder_rnn(packed_input)

        if self.bidirectional or self.num_layers > 1:
            # flatten hidden state
            hidden = hidden.view(batch_size, self.hidden_size*self.hidden_factor)
        else:
            hidden = hidden.squeeze()

        # REPARAMETERIZATION
        mean = self.hidden2mean(hidden)
        logv = self.hidden2logv(hidden)
        std = torch.exp(0.5 * logv)

        z = torch.randn([batch_size, self.latent_size]).to(self.device)
        z = z * std + mean

        # DECODER
        # Encode NLU interpretations
        domain_embed = self.domain_embedding(batch_data['domain']) # batch x h_e
        intent_embed = self.intent_embedding(batch_data['intent']) # batch x h_e
        slot_embed = self.slotkey_embedding(batch_data['slotKey']).mean(dim=1)  # batch x h_e
        z = torch.cat([z, domain_embed, intent_embed, slot_embed], dim=1)

        hidden = self.latent2hidden(z)

        if self.bidirectional or self.num_layers > 1:
            # unflatten hidden state
            hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)
        else:
            hidden = hidden.unsqueeze(0)

        # decoder input
        if self.word_dropout_rate > 0:
            # randomly replace decoder input with <unk>
            prob = torch.rand(input_sequence.size())
            if torch.cuda.is_available():
                prob=prob.cuda()
            prob[(input_sequence.data - self.sos_idx) * (input_sequence.data - self.pad_idx) == 0] = 1
            decoder_input_sequence = input_sequence.clone()
            decoder_input_sequence[prob < self.word_dropout_rate] = self.unk_idx
            input_embedding = self.embedding(decoder_input_sequence)
        input_embedding = self.embedding_dropout(input_embedding)
        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)

        # decoder forward pass
        outputs, _ = self.decoder_rnn(packed_input, hidden)

        # process outputs
        padded_outputs = rnn_utils.pad_packed_sequence(outputs, batch_first=True)[0]
        padded_outputs = padded_outputs.contiguous()
        _,reversed_idx = torch.sort(sorted_idx)
        padded_outputs = padded_outputs[reversed_idx]
        b,s,_ = padded_outputs.size()

        # project outputs to vocab
        logp = nn.functional.log_softmax(self.outputs2vocab(padded_outputs.view(-1, padded_outputs.size(2))), dim=-1)
        logp = logp.view(b, s, self.embedding.num_embeddings)

        return logp, mean, logv, z

    def inference(self, batch_data, batch_length, n=4, z=None):

        # batch_size = batch_data['transcription'].size(0)
        # sorted_lengths, sorted_idx = torch.sort(batch_length['transcription'], descending=True)
        # input_sequence = batch_data['transcription'][sorted_idx]
        #
        # # ENCODER
        # input_embedding = self.embedding(input_sequence)  # b x t x h
        #
        # packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)
        #
        # _, hidden = self.encoder_rnn(packed_input)
        #
        # if self.bidirectional or self.num_layers > 1:
        #     # flatten hidden state
        #     hidden = hidden.view(batch_size, self.hidden_size * self.hidden_factor)
        # else:
        #     hidden = hidden.squeeze()
        #
        # # REPARAMETERIZATION
        # mean = self.hidden2mean(hidden)
        # logv = self.hidden2logv(hidden)
        # std = torch.exp(0.5 * logv)
        #
        # z = torch.randn([batch_size, self.latent_size]).to(self.device)
        # z = z * std + mean

        if z is None:
            batch_size = batch_data['transcription'].size(0)
            z = torch.randn([batch_size, self.latent_size]).to(self.device)
        else:
            batch_size = z.size(0)

        # Encode NLU interpretations
        domain_embed = self.domain_embedding(batch_data['domain'])  # batch x h_e
        intent_embed = self.intent_embedding(batch_data['intent'])  # batch x h_e
        slot_embed = self.slotkey_embedding(batch_data['slotKey']).mean(dim=1)  # batch x h_e
        z = torch.cat([z, domain_embed, intent_embed, slot_embed], dim=1)
        hidden = self.latent2hidden(z)

        if self.bidirectional or self.num_layers > 1:
            # unflatten hidden state
            hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)
        else:
            hidden = hidden.unsqueeze(0)

        generated = self.generate_with_embed(hidden, 1.0)
        generations = self.float_word_tensor_to_string(generated)


        # # required for dynamic stopping of sentence generation
        # sequence_idx = torch.arange(0, batch_size, out=self.tensor()).long()  # all idx of batch
        # # all idx of batch which are still generating
        # sequence_running = torch.arange(0, batch_size, out=self.tensor()).long()
        # sequence_mask = torch.ones(batch_size, out=self.tensor()).bool()
        # # idx of still generating sequences with respect to current loop
        # running_seqs = torch.arange(0, batch_size, out=self.tensor()).long()
        #
        # generations = self.tensor(batch_size, self.opt.max_len).fill_(self.pad_idx).long()
        #
        # t = 0
        # while t < self.opt.max_len and len(running_seqs) > 0:
        #
        #     if t == 0:
        #         input_sequence = torch.Tensor(batch_size).fill_(self.sos_idx).long().to(self.device)
        #
        #     input_sequence = input_sequence.unsqueeze(1)
        #
        #     input_embedding = self.embedding(input_sequence)
        #
        #     output, hidden = self.decoder_rnn(input_embedding, hidden)
        #
        #     logits = self.outputs2vocab(output)
        #
        #     input_sequence = self._sample(logits)
        #
        #     # save next input
        #     generations = self._save_sample(generations, input_sequence, sequence_running, t)
        #
        #     # update gloabl running sequence
        #     sequence_mask[sequence_running] = (input_sequence != self.eos_idx)
        #     sequence_running = sequence_idx.masked_select(sequence_mask)
        #
        #     # update local running sequences
        #     running_mask = (input_sequence != self.eos_idx).data
        #     running_seqs = running_seqs.masked_select(running_mask)
        #
        #     # prune input and hidden state according to local update
        #     if len(running_seqs) > 0:
        #         input_sequence = input_sequence[running_seqs]
        #         hidden = hidden[:, running_seqs]
        #
        #         running_seqs = torch.arange(0, len(running_seqs), out=self.tensor()).long()
        #
        #     t += 1

        return generations, z

    def _sample(self, dist, mode='greedy'):

        if mode == 'greedy':
            _, sample = torch.topk(dist, 1, dim=-1)
        sample = sample.reshape(-1)

        return sample

    def _save_sample(self, save_to, sample, running_seqs, t):
        # select only still running
        running_latest = save_to[running_seqs]
        # update token at position t
        running_latest[:,t] = sample.data
        # save back
        save_to[running_seqs] = running_latest

        return save_to

    def generate_with_embed(self, hidden, temperature, max_sample=MAX_SAMPLE,
                            trunc_sample=TRUNCATED_SAMPLE):
        outputs = Variable(torch.zeros(self.opt.max_len, 1, self.n_words)).to(self.device)
        input = Variable(torch.LongTensor([self.sos_idx])).to(self.device)

        for i in range(self.opt.max_len):
            input = input.unsqueeze(0)
            input = self.embedding(input)
            output, hidden = self.decoder_rnn(input, hidden)
            logits = self.outputs2vocab(output)
            outputs[i] = logits.squeeze(0)
            input, top_i = self.sample(output, temperature, self.device, max_sample=max_sample, trunc_sample=trunc_sample)
            # if top_i == EOS: break
        return outputs.squeeze(1)

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
            top_i = torch.multinomial(output_dist, 1)[0]
            # if len(torch.nonzero(output_dist)) > 0:
            #     top_i = torch.multinomial(output_dist, 1)[0]
            # else:
            #     # TODO: how does this happen?
            #     print(f'[WARNING] output_dist is all zeroes')
            #     top_i = self.unk_idx

        input = Variable(torch.LongTensor([top_i])).to(device)
        return input, top_i

    def float_word_tensor_to_string(self, t):
        s = ''
        for i in range(t.size(0)):
            ti = t[i]
            top_k = ti.data.topk(1)
            top_i = top_k[1][0]
            s += self.vocab['transcription']['itos'][top_i]+" "
            if top_i == self.eos_idx:
                break
        return s