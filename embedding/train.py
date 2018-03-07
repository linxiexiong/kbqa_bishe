import torch
import torch.nn as nn
from torch.autograd import Variable
import sys
sys.path.append('..')
sys.path.append('../..')

from embedding.qa_model import *
from data_processing.mysql import MySQL
from embedding.vocab import *
import pandas as pd
import argparse
from embedding.gen_train_data import gen_train_data


def load_data(file_name):
    return pd.read_csv(file_name)

def gen_batch_train_data(args, train_data, word_dict, char_dict):
    words = Variable((torch.rand(args.batch_size, 10)* args.vocab_size).long())
    chars = Variable((torch.rand(args.batch_size, 10, 5) * 128).long())
    #print (chars)
    entities = Variable((torch.rand(args.batch_size, 10, args.entity_dim)))
    entities_p = Variable(torch.rand(args.batch_size, args.entity_dim))
    entities_n = Variable(torch.rand(args.batch_size, args.entity_dim))
    return words, chars, entities, entities_p, entities_n


def train(args, word_dict, train_data):

    qe_model = EntityModel(args)
    qr_model = RelationModel(args)

    print (qe_model.question_emb.word_emb.weight.data)
    words = index_embedding_words(args.embedding_file)

    pre_train_embedding = load_pretrain_embedding(words, word_dict, args.embedding_file, args.embedding_dim)
    print pre_train_embedding
    #load_embeddings(words, word_dict, args.embedding_file, qe_model.question_emb)
    #load_embeddings(words, word_dict, args.embedding_file, qe_model.entity_emb)
    #load_embeddings(words, word_dict, args.embedding_file, qr_model.relation_emb)
    qe_model.question_emb.word_emb.weight.data.copy_(pre_train_embedding)
    print (qe_model.question_emb.word_emb.weight.data)
    print (qe_model.entity_emb.word_emb.weight.data)
    optimizer = torch.optim.Adagrad(qe_model.parameters(),
                                    lr=args.learning_rate)
    margin = 1
    mask = Variable(torch.rand(args.batch_size, 10))

    def batch_train_entity(data_tensor):
        optimizer.zero_grad()
        score_ep = qe_model(data_tensor['qw'], data_tensor['qc'], data_tensor['entity'],
                            data_tensor['positive_entw'], data_tensor['positive_entc'],
                            data_tensor['positive_ente'], mask)
        score_en = qe_model(data_tensor['qw'], data_tensor['qc'], data_tensor['entity'],
                            data_tensor['negative_entw'], data_tensor['negative_entc'],
                            data_tensor['negative_ente'], mask)
        loss_e = torch.mean(torch.clamp(margin + score_ep - score_en, min=0.0))

        loss_e.backward()
        optimizer.step()
        return score_en, score_ep, loss_e
    for epoch in range(args.epoch):
        #train_data = train_data.sample(frac=1).reset_index(drop=True)
        # words, chars, entities, entities_p, entities_n = gen_batch_train_data(args, train_data,
        #                                                                       word_dict, char_dict)
        data_tensor = gen_train_data(train_data, word_dict, char_dict, args)
        score_n, score_p, loss = batch_train_entity(data_tensor)
        print (score_n, score_p, loss)

if __name__ == "__main__":
    file_name = '../entity_link/head_10.csv'
    train_data = load_data(file_name)
    train_data = train_data.fillna(value='NULL')
    question_datas = train_data['question'].tolist()
    entitty_datas = train_data['topic_words_names'].tolist()
    relation_datas = train_data['relation'].tolist()
    word_datas = question_datas + entitty_datas + relation_datas
    args = argparse.Namespace()
    args.batch_size = 32
    args.epoch = 10
    args.learning_rate = 0.05
    args.embedding_file = '../datas/glove.6B.50d.txt'
    args.file_name = '../entity_link/head_10.csv'

    word_dict = buil_word_dict_simple(True, args.embedding_file,
                                word_datas)

    args.vocab_size = len(word_dict)
    args.embedding_dim = 50

    args.char_dim = 128
    args.char_vocab_size = 128
    args.char_hidden = 128

    args.entity_dim = 50
    args.entity_vocab_size = 10000
    args.entity_hidden = 256
    args.relation_hidden = 256
    args.num_layers = 3


    args.hidden_size = 256
    args.qinit_layers = 3
    args.dropout_rnn = 0
    args.dropout_rnn_output = False
    args.concat_layers = True
    args.rnn_type = 'gru'
    args.rnn_padding = False

    char_dict = build_char_dict()
    train(args, word_dict, train_data)




