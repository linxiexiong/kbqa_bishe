import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CharRNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(TypeRNN, self).__init__()

        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        #self.softmax = F.log_softmax(input_size)

    def forward(self, input, hidden):
        #print(type(input), type(hidden))
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = F.log_softmax(output)
        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(1, self.hidden_size))


class TypeRNN(nn.Module):

    def __init__(self, args):
        super(TypeRNN, self).__init__()
        self.word_emb = nn.Embedding(args.vocab_size, args.emb_dim,
                                     padding_idx=args.padding_idx)
        self.char_emb = nn.Embedding(args.char_vocab_size, args.char_dim)
        self.char_rnn = nn.GRU(args.char_dim, args.char_hidden,
                               batch_first=True, dropout=args.dropout)
        #self.word_emb.weight = nn.Parameter(args.word_embedding)
        self.sentence_rnn = nn.GRU(args.emb_dim + args.char_dim,
                                   args.sent_hidden,
                                   num_layers=args.num_layer,
                                   batch_first=True, bidirectional=True)
        self.fc = nn.Linear(args.sent_hidden * 2 * args.num_layer, args.num_label)
        self.word_rnn_init_h = nn.Parameter(torch.randn(2 * 3,
                                                        args.batch_size,
                                                        args.sent_hidden).type(torch.FloatTensor),
                                            )

    def forward(self, words, chars):
        #print (words)
        word_emb = self.word_emb(words)
        #print (word_emb.size())
        c_batch_size, c_seq_len, c_emb_dim = chars.size()
        #print (chars.size())
        char_emb = self.char_emb(chars.view(-1, c_emb_dim))
        #print (char_emb.size())
        char_emb = char_emb.view(c_batch_size * c_seq_len, c_emb_dim, -1)
        #print (char_emb.size())
        _, char_hidden = self.char_rnn(char_emb)
        char_out = torch.cat(list(char_hidden), dim=1)
        #print (char_out.size())
        char_out = char_out.view(c_batch_size, c_seq_len, -1)
        #print (char_out.size())
        out = torch.cat([word_emb, char_out], dim=2)
        #print (out.size())
        #print (out[0][0])
        _, sent_hidden = self.sentence_rnn(out, self.word_rnn_init_h)
        out = torch.cat(list(sent_hidden), dim=1)
        output = self.fc(out)
        if self.training:
            pred = F.log_softmax(output)
        else:
            pred = F.softmax(output)
        return pred