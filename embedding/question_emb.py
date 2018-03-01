import torch
import torch.nn as nn
from . import layers


class QuestionEmb(nn.Module):

    RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}

    def __init__(self, args, normalize=True):
        super(QuestionEmb, self).__init__()

        self.args = args

        self.embedding = nn.Embedding(args.vocab_size,
                                      args.embedding_dim,
                                      padding_idx=0)

        self.q_init_rnn = layers.StackedBRNN(input_size=args.embedding_dim,
                                             hidden_size=args.hidden_size,
                                             num_layers=args.qinit_layers,
                                             dropout_rate=args.dropout_rnn,
                                             dropout_output=args.dropout_rnn_output,
                                             concat_layers=args.concat_layers,
                                             rnn_type=self.RNN_TYPES[args.rnn_type],
                                             padding=args.rnn_padding)

        self.entity_rnn = layers.StackedBRNN(input_size=args.embedding_dim,
                                             hidden_size=args.hidden_size,
                                             num_layers=args.entity_layers,
                                             dropout_rate=args.dropout_rnn,
                                             dropout_output=args.dropout_rnn_output,
                                             concat_layers=args.concat_layers,
                                             rnn_type=self.RNN_TYPES[args.rnn_type],
                                             padding=args.rnn_padding)
