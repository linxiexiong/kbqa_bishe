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

        self.char_rnn = nn.GRU(args.char_dim, args.char_hidden, num_layers=args.num_layers,
                               bidirectional=True, batch_first=True)

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

    def forward(self, words, chars, entities, mask, method, predict):
        word_emb = self.word_emb(words)
        c_batch_size, c_seq, c_word_size = chars.size()
        char_emb = self.char_emb(chars.view(-1, c_word_size))
        char_emb = char_emb.view(c_batch_size*c_seq, c_word_size, -1)
        _, char_hidden = self.char_rnn(char_emb)
        char_out = torch.cat(list(char_hidden), dim=1)
        char_w_emb = char_out.view(c_batch_size, c_seq, -1)
        char_word_emb = torch.cat([word_emb, char_w_emb], dim=2)
        #print (char_word_emb.size())
        #print (entities.size())
        #_, q_hidden = self.q_init_rnn(char_word_emb)
        #q_init_emb = torch.cat(list(q_hidden), dim=1)
        # size = batch_size * (word_dim + char_dim)
        #ent_emb = self.ent_emb(entities)
        #new_ent_emb = self.attention(q_init_emb, ent_emb)
        #x,y,z = entity_init.size()
        #for idx in change_idx:
        #    entity_init[:][idx][:] = new_ent_emb
        if method == 'word':
            qe_emb = self.q_end_rnn(char_word_emb)
        elif method == 'ent_idx':
            ents_emb = self.ent_emb(entities)
            print (predict)
            print (predict.size())
            predict = predict.unsqueeze(2)
            pred = predict.transpose(2, 1)
            entity_emb = pred.bmm(ents_emb)
            # entity_emb = entity_emb.squeeze(1)
            entities_emb = entity_emb.expand(entity_emb.size(0), char_word_emb.size(1), entity_emb.size(2))
            word_char_ent_emb = torch.cat([char_word_emb, entities_emb], dim=2)
            qe_emb = self.q_end_rnn(word_char_ent_emb)
        else:
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
        self.entity_emb = nn.Embedding(args.entity_vocab_size, args.entity_dim)
        self.char_rnn = nn.GRU(args.char_dim, args.char_hidden, num_layers=args.num_layers,
                               bidirectional=True, batch_first=True)
        self.ent_rnn = nn.GRU(args.embedding_dim + args.char_hidden*args.num_layers*2 + args.entity_dim,
                              args.entity_hidden,
                              num_layers=args.num_layers,
                              batch_first=True, bidirectional=True)
        self.word_rnn = nn.GRU(args.embedding_dim + args.char_hidden*args.num_layers*2,
                              args.entity_hidden,
                              num_layers=args.num_layers,
                              batch_first=True, bidirectional=True)

        # self.word_rnn_init_h = nn.Parameter(torch.randn(2 * args.num_layers,
        #                                                 args.batch_size,
        #                                                 args.entity_hidden).type(torch.FloatTensor),
        #                                     )

    def forward(self, words, chars, ent_emb, method, types):
        #print (type(words.data), type(chars.data), type(ent_emb.data))
        word_emb = self.word_emb(words)
        c_batch_size, c_seq, c_word_size = chars.size()
        char_emb = self.char_emb(chars.view(-1, c_word_size))
        char_emb = char_emb.view(c_batch_size*c_seq, c_word_size, -1)
        _, char_hidden = self.char_rnn(char_emb)
        char_out = torch.cat(list(char_hidden), dim=1)
        char_w_emb = char_out.view(c_batch_size, c_seq, -1)
        char_word_emb = torch.cat([word_emb, char_w_emb], dim=2)
        #print (char_word_emb.size())
        #print (char_word_emb.size())
        #print (ent_emb.size())
        if method == 'ent_idx':
            ent_emb = self.entity_emb(ent_emb)
            print(ent_emb.size())
            print(char_word_emb.size())
            ent_emb = ent_emb.unsqueeze(1)
            ent_emb_exp = ent_emb.expand(ent_emb.size(0), char_word_emb.size(1), ent_emb.size(2))
            word_char_ent_emb = torch.cat([char_word_emb, ent_emb_exp], dim=2)
            _, ent_hidden = self.ent_rnn(word_char_ent_emb)
            out = torch.cat(list(ent_hidden), dim=1)
        elif method == 'ent_vec':
            word_char_ent_emb = torch.cat([char_word_emb, ent_emb], dim=2)
            _, ent_hidden = self.ent_rnn(word_char_ent_emb)
            out = torch.cat(list(ent_hidden), dim=1)
        else:
            _, ent_hidden = self.word_rnn(char_word_emb)
            out = torch.cat(list(ent_hidden), dim=1)
        # word_char_ent_emb = word_char_ent_emb.view(word_char_ent_emb.size(0),
        #                                            -1,
        #                                            word_char_ent_emb.size(1))
        # #print (char_word_emb.size())

        #print (out.size())
        #print (ent_emb.size())
        #entity_emb = torch.cat([out, ent_emb], dim=1)
        return out


class RelationEmb(nn.Module):
    def __init__(self, args):
        super(RelationEmb, self).__init__()
        self.word_emb = nn.Embedding(args.vocab_size, args.embedding_dim)
        self.relation_emb = nn.Embedding(args.rel_vocab_size, args.entity_dim)
        self.rel_rnn = nn.GRU(args.embedding_dim + args.entity_dim,
                              args.relation_hidden,
                              num_layers=args.num_layers,
                              bidirectional=True, batch_first=True)
        self.word_rnn = nn.GRU(args.embedding_dim,
                              args.relation_hidden,
                              num_layers=args.num_layers,
                              bidirectional=True, batch_first=True)

    def forward(self, words, rel_emb, method):
        #print ("rel emb size ===========")
        #print (rel_emb.size())
        word_emb = self.word_emb(words)
        #print (word_emb.size())
        if method == 'ent_idx':
            rel_emb = self.relation_emb(rel_emb)
            rel_emb = rel_emb.unsqueeze(1)
            rel_emb = rel_emb.expand(rel_emb.size(0), word_emb.size(1), rel_emb.size(2))
            word_rel_emb = torch.cat([word_emb, rel_emb], dim=2)
            _, rel_hidden = self.rel_rnn(word_rel_emb)
            out = torch.cat(list(rel_hidden), dim=1)
        elif method == 'ent_vec':
            word_rel_emb = torch.cat([word_emb, rel_emb], dim=2)
            _, rel_hidden = self.rel_rnn(word_rel_emb)
            out = torch.cat(list(rel_hidden), dim=1)
        else:
            _, rel_hidden = self.word_rnn(word_emb)
            out = torch.cat(list(rel_hidden), dim=1)
        # word_rel_emb = word_rel_emb.view(word_rel_emb.size(0), -1, word_rel_emb.size(1))

        #print ("out size ============")
        #print (out.size())
        #relation_emb = torch.cat([out, rel_emb], dim=1)
        return out