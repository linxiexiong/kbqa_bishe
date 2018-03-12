import torch
import torch.nn as nn
from . import layers


class QuestionEmb(nn.Module):

    RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}

    def __init__(self, args, normalize=True):
        super(QuestionEmb, self).__init__()

        self.args = args

        self.word_emb = nn.Embedding(args.vocab_size,
                                      args.embedding_dim,
                                      padding_idx=0)
        self.char_emb = nn.Embedding(args.char_vocab_size,
                                     args.char_dim)
        self.ent_emb = nn.Embedding(args.entity_vocab_size,
                                    args.entity_dim)

        self.char_rnn = nn.GRU(args.char_dim, args.char_hidden, batch_first=True)

        self.q_init_rnn = layers.StackedBRNN(input_size=args.embedding_dim + args.char_dim,
                                             hidden_size=args.hidden_size,
                                             num_layers=args.qinit_layers,
                                             dropout_rate=args.dropout_rnn,
                                             dropout_output=args.dropout_rnn_output,
                                             concat_layers=args.concat_layers,
                                             rnn_type=self.RNN_TYPES[args.rnn_type],
                                             padding=args.rnn_padding)
        # self.q_end_rnn = layers.StackedBRNN(input_size=args.embedding_dim+args.char_dim+args.entity_dim,
        #                                      hidden_size=args.hidden_size,
        #                                      num_layers=args.qinit_layers,
        #                                      dropout_rate=args.dropout_rnn,
        #                                      dropout_output=args.dropout_rnn_output,
        #                                      concat_layers=args.concat_layers,
        #                                      rnn_type=self.RNN_TYPES[args.rnn_type],
        #                                      padding=args.rnn_padding)
        self.q_end_rnn = layers.SimpleRNN(args)
        self.attention = layers.SeqAttnMatch(args.entity_dim)

    def forward(self, words, chars, entities, mask):
        word_emb = self.word_emb(words)
        c_batch_size, c_seq, c_word_size = chars.size()
        char_emb = self.char_emb(chars.view(-1, c_word_size))
        char_emb = char_emb.view(c_batch_size*c_seq, c_word_size, -1)
        _, char_hidden = self.char_rnn(char_emb)
        char_out = torch.cat(list(char_hidden), dim=1)
        char_w_emb = char_out.view(c_batch_size, c_seq, -1)
        char_word_emb = torch.cat([word_emb, char_w_emb], dim=2)
        print (char_word_emb.size())
        print (entities.size())
        #_, q_hidden = self.q_init_rnn(char_word_emb)
        #q_init_emb = torch.cat(list(q_hidden), dim=1)
        # size = batch_size * (word_dim + char_dim)
        #ent_emb = self.ent_emb(entities)
        #new_ent_emb = self.attention(q_init_emb, ent_emb)
        #x,y,z = entity_init.size()
        #for idx in change_idx:
        #    entity_init[:][idx][:] = new_ent_emb
        word_char_ent_emb = torch.cat([char_word_emb, entities], dim=2)
        qe_emb = self.q_end_rnn(word_char_ent_emb)
        #qe_emb = torch.cat(list(qe_hidden), dim=1)
        return qe_emb


class EntityEmb(nn.Module):
    def __init__(self, args):
        super(EntityEmb, self).__init__()
        self.word_emb = nn.Embedding(args.vocab_size, args.embedding_dim)
        self.char_emb = nn.Embedding(args.char_vocab_size,
                                     args.char_dim)
        self.char_rnn = nn.GRU(args.char_dim, args.char_hidden, batch_first=True)
        self.ent_rnn = nn.GRU(args.embedding_dim + args.char_dim,
                              args.entity_hidden,
                              num_layers=args.num_layers,
                              batch_first=True, bidirectional=True)
        # self.word_rnn_init_h = nn.Parameter(torch.randn(2 * args.num_layers,
        #                                                 args.batch_size,
        #                                                 args.entity_hidden).type(torch.FloatTensor),
        #                                     )

    def forward(self, words, chars, ent_emb):
        word_emb = self.word_emb(words)
        c_batch_size, c_seq, c_word_size = chars.size()
        char_emb = self.char_emb(chars.view(-1, c_word_size))
        char_emb = char_emb.view(c_batch_size*c_seq, c_word_size, -1)
        _, char_hidden = self.char_rnn(char_emb)
        char_out = torch.cat(list(char_hidden), dim=1)
        char_w_emb = char_out.view(c_batch_size, c_seq, -1)
        char_word_emb = torch.cat([word_emb, char_w_emb], dim=2)
        print (char_word_emb.size())
        _, ent_hidden = self.ent_rnn(char_word_emb)
        out = torch.cat(list(ent_hidden), dim=1)
        print (out.size())
        print (ent_emb.size())
        entity_emb = torch.cat([out, ent_emb], dim=1)
        return entity_emb


class RelationEmb(nn.Module):
    def __init__(self, args):
        super(RelationEmb, self).__init__()
        self.word_emb = nn.Embedding(args.vocab_size, args.embedding_dim)
        self.rel_rnn = nn.GRU(args.embedding_dim,
                              args.relation_hidden,
                              num_layers=args.num_layers,
                              bidirectional=True)

    def forward(self, words, rel_emb):
        word_emb = self.word_emb(words)
        _, rel_hidden = self.rel_rnn(word_emb)
        out = torch.cat(list(rel_hidden), dim=1)
        relation_emb = torch.cat([out, rel_emb], dim=1)
        return relation_emb