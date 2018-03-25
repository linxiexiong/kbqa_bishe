from __future__ import print_function, unicode_literals
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
from embedding.gen_train_data import *
from visdom import Visdom
import numpy as np
import time
import math
import os
# #
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# torch.cuda.set_device(device=1)
viz = Visdom()
viz.line(X=np.array([0]), Y=np.array([0]), win="loss_100",name='train_loss_10',
         opts={'title':'train loss'})


def load_data(fn):
    return pd.read_csv(fn)


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def gen_batch_train_data(args, train_data, word_dict, char_dict):
    words = Variable((torch.rand(args.batch_size, 10)* args.vocab_size).long())
    chars = Variable((torch.rand(args.batch_size, 10, 5) * 128).long())
    #print (chars)
    entities = Variable((torch.rand(args.batch_size, 10, args.entity_dim)))
    entities_p = Variable(torch.rand(args.batch_size, args.entity_dim))
    entities_n = Variable(torch.rand(args.batch_size, args.entity_dim))
    return words, chars, entities, entities_p, entities_n


def train(args, word_dict, train_data, entity_dict, relation_dict):
    time_start = time.time()
    wf = open('error_data.txt', 'w')
    qe_model = EntityModel(args).cuda()
    #qe_model = nn.DataParallel(qe_model, device_ids=[0,1,2,3])
    qr_model = RelationModel(args).cuda()
    #qr_model = nn.DataParallel(qr_model, device_ids=[0,1,2,3])
    #print (qe_model.question_emb.word_emb.weight.data)
    words = index_embedding_words(args.embedding_file)

    pre_train_embedding = load_pretrain_embedding(words, word_dict, args.embedding_file, args.embedding_dim)
    #print (pre_train_embedding)
    #load_embeddings(words, word_dict, args.embedding_file, qe_model.question_emb)
    #load_embeddings(words, word_dict, args.embedding_file, qe_model.entity_emb)
    #load_embeddings(words, word_dict, args.embedding_file, qr_model.relation_emb)
    # pre_train_ent_emb = load_entity_embedding(entity_dict, args.entity_dim)
    # pre_train_rel_emb = load_relation_embedding(relation_dict, args.entity_dim)

    qe_model.question_emb.word_emb.weight.data.copy_(pre_train_embedding)
    qe_model.entity_emb.word_emb.weight.data.copy_(pre_train_embedding)
    qr_model.question_emb.word_emb.weight.data.copy_(pre_train_embedding)
    qr_model.relation_emb.word_emb.weight.data.copy_(pre_train_embedding)

    # qe_model.question_emb.ent_emb.weight.data.copy_(pre_train_ent_emb)
    # qe_model.entity_emb.entity_emb.weight.data.copy_(pre_train_ent_emb)
    # qr_model.question_emb.ent_emb.weight.data.copy_(pre_train_ent_emb)
    # qr_model.relation_emb.relation_emb.weight.data.copy_(pre_train_rel_emb)

    #print (qe_model.question_emb.word_emb.weight.data)
    #print (qe_model.entity_emb.word_emb.weight.data)
    optimizer = torch.optim.Adagrad(qe_model.parameters(),
                                    lr=args.learning_rate)
    optimizer_r = torch.optim.Adagrad(qr_model.parameters(),
                                      lr=args.learning_rate)
    margin = 0.5
    mask = Variable(torch.rand(args.batch_size, 10))
    #train_data = prepair_data(train_data, args.triples)
    train_data = load_prepaired_data('../datas/prepaired_train_data_1w.csv')
    train_data['negative_type'] = train_data['negative_type'].fillna(value='none')
    train_data['positive_type'] = train_data['positive_type'].fillna(value='none')
    #print (qe_model.state_dict())
    def batch_train_entity(data_tensor):
        optimizer.zero_grad()
        optimizer_r.zero_grad()
        #print(optimizer.param_groups)
        if args.method == 'ent_idx':
            pos_entity_emb = data_tensor['positive_enti']
            neg_entity_emb = data_tensor['negative_enti']
            pos_relation_emb = data_tensor['positive_reli']
            neg_relation_emb = data_tensor['negative_reli']
            entity_emb = data_tensor['cand_ent']
        else:
            pos_entity_emb = data_tensor['positive_ente']
            neg_entity_emb = data_tensor['negative_ente']
            pos_relation_emb = data_tensor['positive_rele']
            neg_relation_emb = data_tensor['negative_rele']
            entity_emb = data_tensor['entity']

        score_ep = qe_model(data_tensor['qw'], data_tensor['qc'], entity_emb,
                            data_tensor['positive_entw'], data_tensor['positive_entc'],
                            pos_entity_emb, mask, args.method,
                            data_tensor['predict'], data_tensor['positive_type'])
        score_en = qe_model(data_tensor['qw'], data_tensor['qc'], entity_emb,
                            data_tensor['negative_entw'], data_tensor['negative_entc'],
                            neg_entity_emb, mask, args.method,
                            data_tensor['predict'], data_tensor['negative_type'])
        score_rp = qr_model(data_tensor['qw'], data_tensor['qc'], entity_emb,
                            data_tensor['positive_relw'],
                            pos_relation_emb, args.method,
                            mask, data_tensor['predict'])
        score_rn = qr_model(data_tensor['qw'], data_tensor['qc'], entity_emb,
                            data_tensor['negative_relw'],
                            neg_relation_emb, args.method,
                            mask, data_tensor['predict'])
        #loss_e = torch.mean(torch.clamp(margin + score_ep - score_en, min=0.0))
        #loss_r = torch.mean(torch.clamp(margin + score_rp - score_rn, min = 0.0))

        dist = torch.clamp(margin - score_ep + score_en, min=0.0)
        dist_r = torch.clamp(margin - score_rp + score_rn, min=0.0)
        #loss_e = torch.mean(dist)
        #loss_r = torch.mean(dist_r)
        loss = torch.mean(dist + dist_r)
        #print (loss_e)
        #loss_e.backward()
        #loss_r.backward()
        loss.backward()
        #score_ep.register_hook(print)
        #score_en.register_hook(print)

        # for p in qe_model.parameters():
        #     print (p.grad
        #raise)
        optimizer.step()
        optimizer_r.step()
        #print (loss_e)
        #print (optimizer.param_groups)
        return score_en, score_ep, loss
    # loss_avg = 0.0
    # loss_sum = 0.0
    for epoch in range(args.epoch):
        #train_data = train_data.sample(frac=1).reset_index(drop=True)
        # words, chars, entities, entities_p, entities_n = gen_batch_train_data(args, train_data,
        #                                                                       word_dict, char_dict)

        data_tensor = gen_train_data(train_data, word_dict, char_dict, args)
        # if data_tensor['positive_entw'].size(1) > 20 or data_tensor['negative_entw'] > 20:
        #     wf.writelines(data_tensor[''])
        score_n, score_p, loss = batch_train_entity(data_tensor)
        # loss_sum += loss
        # loss_avg = loss_sum / epoch
        print('epoch ===%s ====' % epoch)
        print (time_since(time_start))
        print (loss)

        viz.line(X=np.array([epoch]), Y=np.array([loss.data[0]]), win="loss_100",
                 name='train_loss_10', update='append')
        #print (score_n, score_p, loss)
    return qe_model, qr_model


def predict(qe_model, qr_model, word_dict, char_dict, test_data, entity_dict, relation_dict):
    qid_max = max(test_data['qid'].tolist())
    mask = Variable(torch.rand(args.batch_size, 10))
    cnt = 0
    rnt = 0
    no_in = 0
    for i in range(qid_max):
        data_tensor = gen_simgle_test_data(test_data, word_dict, char_dict,
                                           i, args.entity_dim, entity_dict)
        qw = data_tensor['qw_tensor']
        qc = data_tensor['qc_tensor']
        entity = data_tensor['entity_tensor']
        candw_tensor = data_tensor['candw_tensor']
        candc_tensor = data_tensor['candc_tensor']
        cande_tensor = data_tensor['cande_tensor']
        cand_ents = data_tensor['cand_mid']
        entity_idx_tensor = data_tensor['cand_indices']
        candi_tensor = data_tensor['candi_tensor']
        candt_tensor = data_tensor['cand_type']
        cand_pred_tensor = data_tensor['predict']

        #cand_vecs = data_tensor['cand_vecs']
        cand_score = dict()
        cand_gold = (data_tensor['golden'])
        max_score = -10000.0
        max_cand = 'None'
        max_pred = None
        for idx, cand in enumerate(cand_ents):
            if args.method == 'ent_idx':
                scores = qe_model(qw, qc, entity_idx_tensor.unsqueeze(0),
                                  candw_tensor[idx], candc_tensor[idx],
                                  candi_tensor[idx], mask, args.method,
                                  cand_pred_tensor, candt_tensor[idx])
            else:
                scores = qe_model(qw, qc, entity,
                                  candw_tensor[idx], candc_tensor[idx],
                                  cande_tensor[idx], mask, args.method,
                                  cand_pred_tensor, candt_tensor[idx])

            s = float(scores.data.cpu().numpy()[0])
            if s > max_score:
                max_score = s
                max_cand = cand
                #max_pred = cand_pred_tensor[idx]
            cand_score[cand] = scores

        rel_tensor = get_single_relation_tensor(test_data, max_cand, word_dict,
                                                i, args.triples, relation_dict)

        cand_rw = rel_tensor['candrw_tensor']
        cand_re = rel_tensor['candre_tensor']
        golden_rel = rel_tensor['golden_relation']
        cand_rels = rel_tensor['cand_rels']
        cand_ri = rel_tensor['candri_tensor']

        max_rel = 'None'
        max_rel_score = -10000
        for idx, cand_rel in enumerate(cand_rels):
            if args.method == 'ent_idx':
                score_rel = qr_model(qw, qc, entity_idx_tensor.unsqueeze(0),
                                     cand_rw[idx], cand_ri[idx],
                                     args.method, mask, cand_pred_tensor)
            else:
                score_rel = qr_model(qw, qc, entity, cand_rw[idx], cand_re[idx],
                                     args.method, mask, cand_pred_tensor)
            s_rel = float(score_rel.data.cpu().numpy()[0])
            if s_rel > max_rel_score:
                max_rel_score = s_rel
                max_rel = cand_rel
        print (max_rel, golden_rel)
        print (max_cand, cand_gold)
        if max_cand == cand_gold:
            cnt += 1
            if max_rel == golden_rel:
                rnt +=1
        if cand_gold  in cand_ents:
            no_in += 1
            #print (cand_score[cand_gold])
    print (cnt, rnt, no_in, qid_max)
    #print (cand_score)
    print ("predict done")


if __name__ == "__main__":
    file_name = '../entity_link/train_head_10.csv'
    train_data = load_data(file_name)
    train_data = train_data.fillna(value='none')
    #train_data = train_data['topic_words_names'].fillna(value='none')

    entity_dict = build_entity_dict(train_data['subject_id'].tolist() + train_data['object_id'].tolist()
                                    + train_data['topic_words'].tolist())

    train_data['question'] = train_data['question'].apply(lambda x: x.lower())
    train_data['topic_words_names'] = train_data['topic_words_names'].apply(lambda x: x.lower())
    relation_datas = train_data['relation'].apply(lambda x: x.lower().replace("/", " ").replace('_', ' ')).tolist()
    question_datas = train_data['question'].tolist()
    entitty_datas = train_data['topic_words_names'].tolist()
    #relation_datas = train_data['relation'].tolist()
    word_datas = question_datas + entitty_datas + relation_datas
    args = argparse.Namespace()
    args.batch_size = 32
    args.epoch = 10000
    args.learning_rate = 0.002
    args.embedding_file = '../datas/glove.6B.50d.txt'
    args.file_name = '../entity_link/train_head_10.csv'

    word_dict = buil_word_dict_simple(True, args.embedding_file,
                                word_datas)

    args.vocab_size = len(word_dict)
    args.embedding_dim = 50

    args.char_dim = 50
    args.char_vocab_size = 128
    args.char_hidden = 128

    args.entity_dim = 50
    args.entity_vocab_size = len(entity_dict)

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

    args.triples = pd.read_csv('../datas/FB2M-triples.txt', header=None, sep='\t')
    args.triples.columns = ['subject', 'relation', 'object']
    args.method = 'ent_idx'

    relation_dict = build_relation_dict(args.triples['relation'].tolist())
    args.rel_vocab_size = len(relation_dict)
    char_dict = build_char_dict()

    #data = prepair_data(train_data, args, entity_dict, relation_dict)
    qe_model, qr_model = train(args, word_dict, train_data, entity_dict, relation_dict)
    print ("train done =============================")
    # param = {
    #     'qe_model': qe_model,
    #     'qr_model': qr_model,
    #     'word_dict': word_dict,
    #     'char_dict': char_dict,
    #     'entity_dict': entity_dict,
    #     'relation_dict': relation_dict,
    #     'args': args
    # }
    # torch.save(param, 'qa-100-rnn.pt')
    qe_model.eval()
    qr_model.eval()
    test_data = pd.read_csv('../entity_link/test_head_10.csv')
    test_data = test_data.fillna(value='none')
    test_data['question'] = test_data['question'].apply(lambda x: x.lower())
    test_data['topic_words_names'] = test_data['topic_words_names'].apply(lambda x: x.lower())
    test_data['relation'] = test_data['relation'].apply(lambda x: x.lower())
    #test_data = test_data['topic_words_names'].fillna(value='none')
    predict(qe_model, qr_model, word_dict, char_dict, test_data, entity_dict, relation_dict)


class AverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
