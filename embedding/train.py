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
from embedding.gen_train_data import gen_train_data, prepair_data, gen_simgle_test_data


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

    #print (qe_model.question_emb.word_emb.weight.data)
    words = index_embedding_words(args.embedding_file)

    pre_train_embedding = load_pretrain_embedding(words, word_dict, args.embedding_file, args.embedding_dim)
    #print (pre_train_embedding)
    #load_embeddings(words, word_dict, args.embedding_file, qe_model.question_emb)
    #load_embeddings(words, word_dict, args.embedding_file, qe_model.entity_emb)
    #load_embeddings(words, word_dict, args.embedding_file, qr_model.relation_emb)
    qe_model.question_emb.word_emb.weight.data.copy_(pre_train_embedding)
    qe_model.entity_emb.word_emb.weight.data.copy_(pre_train_embedding)
    #print (qe_model.question_emb.word_emb.weight.data)
    #print (qe_model.entity_emb.word_emb.weight.data)
    optimizer = torch.optim.Adagrad(qe_model.parameters(),
                                    lr=args.learning_rate)
    margin = 0.5
    mask = Variable(torch.rand(args.batch_size, 10))
    train_data = prepair_data(train_data, args.triples)
    #print (qe_model.state_dict())
    def batch_train_entity(data_tensor):
        optimizer.zero_grad()
        #print(optimizer.param_groups)
        score_ep = qe_model(data_tensor['qw'], data_tensor['qc'], data_tensor['entity'],
                            data_tensor['positive_entw'], data_tensor['positive_entc'],
                            data_tensor['positive_ente'], mask)
        score_en = qe_model(data_tensor['qw'], data_tensor['qc'], data_tensor['entity'],
                            data_tensor['negative_entw'], data_tensor['negative_entc'],
                            data_tensor['negative_ente'], mask)
        loss_e = torch.mean(torch.clamp(margin + score_ep - score_en, min=0.0))

        loss_e.backward()
        optimizer.step()
        print (score_ep.grad.data)
        #print (loss_e)
        #print (optimizer.param_groups)
        return score_en, score_ep, loss_e
    for epoch in range(args.epoch):
        #train_data = train_data.sample(frac=1).reset_index(drop=True)
        # words, chars, entities, entities_p, entities_n = gen_batch_train_data(args, train_data,
        #                                                                       word_dict, char_dict)

        data_tensor = gen_train_data(train_data, word_dict, char_dict, args)
        score_n, score_p, loss = batch_train_entity(data_tensor)
        print (score_n, score_p, loss)
    return qe_model


def predict(model, word_dict, char_dict, test_data):
    qid_max = max(test_data['qid'].tolist())
    mask = Variable(torch.rand(args.batch_size, 10))
    cnt = 0
    no_in = 0
    for i in range(qid_max):
        data_tensor = gen_simgle_test_data(test_data, word_dict, char_dict, i, entity_dim=50)
        qw = data_tensor['qw_tensor']
        qc = data_tensor['qc_tensor']
        entity = data_tensor['entity_tensor']
        candw_tensor = data_tensor['candw_tensor']
        candc_tensor = data_tensor['candc_tensor']
        cande_tensor = data_tensor['cande_tensor']
        cand_ents = data_tensor['cand_mid']
        #cand_vecs = data_tensor['cand_vecs']
        cand_score = dict()
        cand_gold = (data_tensor['golden'])
        max_score = -10000.0
        max_cand = 'None'
        for idx, cand in enumerate(cand_ents):

            scores = model(qw, qc, entity,
                            candw_tensor[idx], candc_tensor[idx],
                            cande_tensor[idx], mask)
            #print (cand, scores)
            s = float(scores.data.cpu().numpy()[0])
            if s > max_score:
                max_score = s
                max_cand = cand
            cand_score[cand] = scores
        #print (max_cand, cand_gold)
        if max_cand == cand_gold:
            cnt += 1
        if cand_gold  in cand_score:
            no_in += 1
            #print (cand_score[cand_gold])
    print (cnt, no_in, qid_max)
    #print (cand_score)
    print ("predict done")


if __name__ == "__main__":
    file_name = '../entity_link/train_head_100_10.csv'
    train_data = load_data(file_name)
    train_data = train_data.fillna(value='none')
    #train_data = train_data['topic_words_names'].fillna(value='none')
    train_data['question'] = train_data['question'].apply(lambda x: x.lower())
    train_data['topic_words_names'] = train_data['topic_words_names'].apply(lambda x: x.lower())
    train_data['relation'] = train_data['relation'].apply(lambda x: x.lower())
    question_datas = train_data['question'].tolist()
    entitty_datas = train_data['topic_words_names'].tolist()
    relation_datas = train_data['relation'].tolist()
    word_datas = question_datas + entitty_datas + relation_datas
    args = argparse.Namespace()
    args.batch_size = 50
    args.epoch = 2
    args.learning_rate = 0.2
    args.embedding_file = '../datas/glove.6B.50d.txt'
    args.file_name = '../entity_link/train_head_100_10.csv'

    word_dict = buil_word_dict_simple(True, args.embedding_file,
                                word_datas)

    args.vocab_size = len(word_dict)
    args.embedding_dim = 50

    args.char_dim = 128
    args.char_vocab_size = 128
    args.char_hidden = 128

    args.entity_dim = 50
    args.entity_vocab_size = len(word_dict)
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

    args.triples = pd.read_csv('../datas/FB5M-triples.txt', header=None, sep='\t')
    args.triples.columns = ['subject', 'relation', 'object']
    

    char_dict = build_char_dict()
    model = train(args, word_dict, train_data)
    print ("train done =============================")
    model.eval()
    test_data = pd.read_csv('../entity_link/train_head_100_10.csv')
    test_data = test_data.fillna(value='none')
    test_data['question'] = test_data['question'].apply(lambda x: x.lower())
    test_data['topic_words_names'] = test_data['topic_words_names'].apply(lambda x: x.lower())
    test_data['relation'] = test_data['relation'].apply(lambda x: x.lower())
    #test_data = test_data['topic_words_names'].fillna(value='none')
    predict(model, word_dict, char_dict, test_data)



