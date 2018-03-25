import sys
sys.path.append('..')
sys.path.append('../..')
import torch
import torch.nn as nn
from torch.autograd import Variable
from data_processing.mysql import MySQL, get_mid_to_name_mysql, get_relation_vector, get_mid_type
from entity_link.features import get_entity_vector
from type_handle.data_handle import lines_to_word_tensor, lines_to_char_tensor, \
    line_to_word_tensor,line_to_char_tensor
from nltk.tokenize import word_tokenize
import numpy as np
import random
import pandas as pd
import ast
import re


def get_cand_rel(triples, mids):
    #triples = pd.read_csv('../datas/FB5M-triples.txt', header=None, sep='\t')
    #triples.columns = ['subject', 'relation', 'object']
    rels = set()
    for mid in mids:
        rel = triples.loc[triples['subject'] == mid, 'relation'].tolist()
        for r in rel:
            rels.add(r)
    #print (len(rels))
    return list(rels)


# ===================================== handle train data ========================
def prepair_data(data, args, entity_dict, relation_dict):
    data = data.sort_values(['qid', 'predict'], ascending=[True, False])
    qid_max = max(data['qid'].tolist())
    print (qid_max)
    train = {'question':[], 'cand_ent':[], 'pos':[], 'entities_vec':[], 'positive_vec':[],
             'positive':[], 'negative':[], 'predict':[], 'negative_vec':[], 'cand_rel':[],
             'pos_rel':[], 'pos_rel_vec':[], 'neg_rel':[], 'neg_rel_vec':[], 'cand_indices':[],
             'positive_idx':[], 'positive_type':[], 'negative_idx':[], 'negative_type':[],
             'pos_rel_idx': [], 'neg_rel_idx': []}
    db_conn = MySQL(ip='10.61.2.166', port=3306, user='zengyutao',
                                       pw='zengyutao', db_name='wikidata')
    #data['vector'] = data.apply(lambda x: get_entity_vector(db_conn, x['topic_words']), axis=1)
    #print(data['vector'])

    for i in range(qid_max+1):
        nums = len(data[data['qid'] == i])
        question = data.loc[data['qid'] == i, 'question'].iloc[0]
        train['question'].append(question)

        pos = data.loc[data['qid'] == i, 'pos'].iloc[0]
        train['pos'].append(pos)

        predict = data.loc[data['qid'] == i, 'predict']
        train['predict'].append(predict.tolist())

        cand_mid = data.loc[data['qid']==i, 'topic_words']
        #data['vectors'] = data['vector'].apply(lambda x: x.split(','))
        train['cand_ent'].append(cand_mid)
        #print (data.loc[data['qid'] == i, 'vectors'])
        vectors = []
        cand_vec_dict = {}
        cand_indices = []
        for cand in cand_mid:
            #print cand
            vec = get_entity_vector(db_conn, cand)
            #print vec
            vec = [float(x) for x in str(vec).split(',')]
            vectors.append(vec)
            cand_vec_dict[cand] = vec
            cand_indices.append(entity_dict[cand])

        entities_vec = np.matmul(np.array(predict), np.array(vectors))
        #entities_vec = mul_vec(predict.tolist(), vectors)
        train['entities_vec'].append(entities_vec)
        train['cand_indices'].append(cand_indices)

        positive = data.loc[data['qid'] == i, 'golden_word_name'].iloc[0]
        train['positive'].append(positive)

        # cands_name = data.loc[data['qid']==i, 'topic_words_names']
        # neg_name = [w for w in cands_name if w != positive]
        # train['negtive'].append(neg_name)

        positive_mid = data.loc[data['qid'] == i, 'golden_word'].iloc[0]
        train['positive_idx'].append(entity_dict[positive_mid])
        ptype = get_mid_type(db_conn, positive_mid)
        if ptype is None:
            train['positive_type'].append("none")
        else:
            ptype_name = get_mid_to_name_mysql(db_conn, ptype)
            if ptype_name is None:
                train['positive_type'].append("none")
            else:
                train['positive_type'].append(ptype_name)
        positive_vec = cand_vec_dict[positive_mid]
        train['positive_vec'].append(positive_vec)

        neg_vec = [cand_vec_dict[neg_mid] for neg_mid in cand_mid if neg_mid != positive_mid]
        neg = [get_mid_to_name_mysql(db_conn, neg_mid) for neg_mid in cand_mid if neg_mid != positive_mid]
        neg_idx = [entity_dict[neg_mid] for neg_mid in cand_mid if neg_mid != positive_mid]
        #neg_type = [get_mid_type(db_conn, neg_mid) for neg_mid in cand_mid if neg_mid != positive_mid]
        train['negative'].append(neg)
        train['negative_vec'].append(neg_vec)
        train['negative_idx'].append(neg_idx)

        neg_type = []
        for neg_mid in cand_mid:
            if neg_mid == positive_mid:
                continue
            ntype = get_mid_type(db_conn, neg_mid)
            if ntype is None:
                neg_type.append("none")
            else:
                ntype_name = get_mid_to_name_mysql(db_conn, ntype)
                if ntype_name is None:
                    neg_type.append("none")
                else:
                    neg_type.append(ntype_name)
        train['negative_type'].append(neg_type)
        #print ("================gen pos rel ==================")
        positive_rel = data.loc[data['qid'] == i, 'relation'].iloc[0]
        pos_rel_idx = relation_dict[positive_rel]
        pos_rel_vec = get_relation_vector(db_conn, positive_rel)
        pos_rel_vec = [float(x) for x in str(pos_rel_vec).split(',')]
        positive_rel_re = positive_rel.replace('/', ' ').replace('_', ' ')

        train['pos_rel'].append(positive_rel_re)
        train['pos_rel_vec'].append(pos_rel_vec)
        train['pos_rel_idx'].append(pos_rel_idx)

        #print ("----------------get cand rel----------")
        cand_rel = get_cand_rel(args.triples, cand_mid)
        train['cand_rel'].append(cand_rel)

        #print ("--------------get neg rel --------------")
        neg_rel = [rel for rel in cand_rel if rel != positive_rel]
        neg_rel_idx = [relation_dict[rel] for rel in cand_rel if rel != positive_rel]
        neg_rel_vec = []
        neg_rel_re = []
        for n in neg_rel:
            vec = get_relation_vector(db_conn, n)
            neg_v = [float(x) for x in str(vec).split(',')]
            neg_rel_vec.append(neg_v)
            neg_rel_re.append(n.replace('/', ' ').replace('_', ' '))

        train['neg_rel_idx'].append(neg_rel_idx)
        train['neg_rel'].append(neg_rel_re)
        train['neg_rel_vec'].append(neg_rel_vec)


    train_data = pd.DataFrame(train)

    # for key in train:
    #     train_data[key] = train.get(key)
    train_data.to_csv('../datas/prepaired_train_data_1w.csv', index=False)
    return train_data


def load_prepaired_data(file_name):
    data = pd.read_csv(file_name)
    print (data[0:10])
    return data


def gen_train_data(train_data, word_dict, char_dict, args):
    #train_data = prepair_data(data)
    #train_data = train_data.sample(frac=1).reset_index(drop=True)
    questions = []
    positive_ents = []
    negative_ents = []
    positive_vecs = []
    negative_vecs = []
    entities_vecs = []
    positive_rels = []
    pos_rel_vecs = []
    negative_rels = []
    neg_rel_vecs = []
    pos_ent_idxs = []
    neg_ent_idxs = []
    pos_rel_idxs = []
    neg_rel_idxs = []
    positive_types = []
    negative_types = []
    cand_idices = []
    predicts = []
    poses = []
    for i in range(0, args.batch_size):
        n = random.randint(0, len(train_data) - 1)
        row = train_data.loc[[n]]
        #print (row['question'])
        questions.append(row['question'].loc[n])

        positive_ents.append(row['positive'].loc[n])
        pos_ent_idxs.append(row['positive_idx'].loc[n])
        positive_types.append(row['positive_type'].loc[n])
        #positive_ents = ast.literal_eval(positive_ents)
        positive_vecs.append(ast.literal_eval(row['positive_vec'].loc[n]))
        #positive_vecs = ast.literal_eval(positive_vecs)

        neg_ents = row['negative'].loc[n]
        neg_ents = ast.literal_eval(neg_ents)

        neg_vecs = row['negative_vec'].loc[n]
        neg_vecs = ast.literal_eval(neg_vecs)

        nidx = row['negative_idx'].loc[n]
        nidx = ast.literal_eval(nidx)

        neg_type = row['negative_type'].loc[n]
        neg_type = ast.literal_eval(neg_type)
        assert (len(neg_ents) == len(neg_vecs) and
                len(neg_ents) == len(neg_vecs) and
                len(neg_ents) == len(neg_type)), "negative ents and vecs mush have same len"
        if len(neg_ents) == 0:
            negative_ents.append("none")
            negative_vecs.append(np.random.rand(args.entity_dim))
            neg_ent_idxs.append(0)
            negative_types.append("none")
        else:
            m = random.randint(0, len(neg_ents) - 1)
            negative_ents.append(neg_ents[m])
            negative_vecs.append(neg_vecs[m])
            neg_ent_idxs.append(nidx[m])
            negative_types.append(neg_type[m])
        #print ((row['entities_vec'].loc[n]))
        #ev = eval(row['entities_vec'].loc[n])
        #print (row['entities_vec'].loc[n])
        junkers = re.compile('[[\]]')
        result = junkers.sub('', row['entities_vec'].loc[n]).strip().split(' ')
        #print (result)
        ev = [float(x.strip()) for x in result if x != '']
        #print (ev)
        #ev = [float(x) for x in row['entities_vec'].loc[n] if x != '[' or x != ']']
        entities_vecs.append(ev)

        cand_idx = row['cand_indices'].loc[n]
        cand_idx = ast.literal_eval(cand_idx)
        cand_idices.append(cand_idx)
        predicts.append(ast.literal_eval(row['predict'].loc[n]))


        poses.append(ast.literal_eval(row['pos'].loc[n]))

        positive_rels.append(row['pos_rel'].loc[n])
        pos_rel_idxs.append(row['pos_rel_idx'].loc[n])
        pos_rel_vecs.append(ast.literal_eval(row['pos_rel_vec'].loc[n]))
        #print (pos_rel_vecs)
        neg_rels = ast.literal_eval(row['neg_rel'].loc[n])
        neg_rel_idx = ast.literal_eval(row['neg_rel_idx'].loc[n])
        neg_rel_vec = ast.literal_eval(row['neg_rel_vec'].loc[n])
        #print (neg_rels)
        #print (neg_rel_vec)
        #print (len(neg_rels), len(neg_rel_vec))
        assert (len(neg_rels) == len(neg_rel_vec) and
                len(neg_rels) == len(neg_rel_idx)), "negative rels and vecs mush have the same len"
        if len(neg_rels) == 0:
            negative_rels.append("none")
            neg_rel_vecs.append(np.random.rand(args.entity_dim))
            neg_rel_idxs.append(0)
        else:
            m = random.randint(0, len(neg_rels) - 1)
            negative_rels.append(neg_rels[m])
            neg_rel_vecs.append(neg_rel_vec[m])
            neg_rel_idxs.append(neg_rel_idx[m])


    max_len = 0
    for line in questions:
        if len(word_tokenize(line)) > max_len:
            max_len = len(word_tokenize(line))
    entity_tensor = torch.rand(args.batch_size, max_len, args.entity_dim)
    qw_tensor = lines_to_word_tensor(questions, word_dict, max_len)
    qc_tensor = lines_to_char_tensor(questions, char_dict, max_len)

    pos_vec = torch.LongTensor(args.batch_size, max_len).fill_(1)
    #cand_idx_tensor = torch.Tensor(cand_idices)
    max_cand_len = max([len(ci) for ci in cand_idices])
    cand_idx_tensor = torch.LongTensor(len(cand_idices), max_cand_len).zero_()
    for c, cand_i in enumerate(cand_idices):
        for cm, cand_cm in enumerate(cand_idices[c]):
            cand_idx_tensor[c][cm] = cand_cm

    max_pre_len = max([len(pred) for pred in predicts])
    predicts_tensor = torch.Tensor(len(predicts), max_pre_len).zero_()
    for pi, preds in enumerate(predicts):
        for pj, pred in enumerate(preds):
            predicts_tensor[pi][pj] = pred
    #predicts_tensor = torch.Tensor(predicts)

    for i in range(0, args.batch_size):
        entities_vec = entities_vecs[i]
        pos = poses[i]
        #pos = ast.literal_eval(pos)
        #print (pos)
        for j in range(len(pos)):
            #entity_tensor[i][pos[j]] = torch.from_numpy(entities_vec)
            entity_tensor[i][pos[j]] = torch.Tensor(entities_vec)
            pos_vec[i][pos[j]] = 2
            #entity_idx[i][pos[j]] = torch.Tensor(cand_idices)

    #positive_word_tensor = lines_to_word_tensor(positive_ents + positive_types, word_dict)
    #positive_entw_tensor = positive_word_tensor[: args.batch_size]
    ##positive_typew_tensor = positive_word_tensor[args.batch_size:]
    max_word_len = max([len(word_tokenize(w)) for w in (positive_ents + positive_types) if w is not None])
    if max_word_len > 20:
        print (positive_ents + positive_types)

    positive_entw_tensor = lines_to_word_tensor(positive_ents, word_dict, max_word_len)
    positive_entc_tensor = lines_to_char_tensor(positive_ents, char_dict, max_word_len)
    positive_typew_tensor = lines_to_word_tensor(positive_types, word_dict, max_word_len)
    # print (positive_typew_tensor)
    # positive_typew_tensor = positive_typew_tensor.resize_(positive_typew_tensor.size(0),
    #                                                       positive_entw_tensor.size(1))
    # print (positive_typew_tensor)

    positive_enti_tensor = torch.from_numpy(np.array(pos_ent_idxs))

    #positive_ente_tensor = torch.Tensor(positive_vecs)
    neg_word_len = max([len(word_tokenize(nw)) for nw in (negative_ents + negative_types) if nw is not None])
    if neg_word_len > 20:
        print(negative_ents + negative_types)
    negative_entw_tensor = lines_to_word_tensor(negative_ents, word_dict, neg_word_len)
    negative_entc_tensor = lines_to_char_tensor(negative_ents, char_dict, neg_word_len)
    negative_typew_tensor = lines_to_word_tensor(negative_types, word_dict, neg_word_len)
    # negative_typew_tensor = negative_typew_tensor.resize_(negative_typew_tensor.size(0),
    #                                                      negative_entw_tensor.size(1))
    negative_enti_tensor = torch.from_numpy(np.array(neg_ent_idxs))
    #negative_ente_tensor = torch.Tensor(negative_vecs)

    #print ("============rel tensor sizes ===========")
    pos_rel_word_len = max([len(word_tokenize(pr)) for pr in positive_rels if pr is not None])
    positive_relw_tensor = lines_to_word_tensor(positive_rels, word_dict, pos_rel_word_len)
    positive_reli_tensor = torch.from_numpy(np.array(pos_rel_idxs))
    #print (positive_relw_tensor.data.size())
    #positive_rele_tensor = torch.Tensor(pos_rel_vecs)

    neg_rel_word_len = max([len(word_tokenize(nr)) for nr in negative_rels if nr is not None])
    negative_relw_tensor = lines_to_word_tensor(negative_rels, word_dict, neg_rel_word_len)
    negative_reli_tensor = torch.from_numpy(np.array(neg_rel_idxs))
    #print (negative_relw_tensor.data.size())
    #negative_rele_tensor = torch.Tensor(neg_rel_vecs)

    positive_ente_tensor = torch.rand(args.batch_size, positive_entw_tensor.size(1), args.entity_dim)
    negative_ente_tensor = torch.rand(args.batch_size, negative_entw_tensor.size(1), args.entity_dim)
    positive_rele_tensor = torch.rand(args.batch_size, positive_relw_tensor.size(1), args.entity_dim)
    negative_rele_tensor = torch.rand(args.batch_size, negative_relw_tensor.size(1), args.entity_dim)
    for i in range(args.batch_size):
        for x in range(positive_entw_tensor.size(1)):
                positive_ente_tensor[i][x] = torch.Tensor(positive_vecs[i])
        for xn in range(negative_entw_tensor.size(1)):
                negative_ente_tensor[i][xn] = torch.Tensor(negative_vecs[i])
        for y in range(positive_relw_tensor.size(1)):
                positive_rele_tensor[i][y] = torch.Tensor(pos_rel_vecs[i])
        for yn in range(negative_relw_tensor.size(1)):
                negative_rele_tensor[i][yn] = torch.Tensor(neg_rel_vecs[i])

    data_tensor = dict()
    data_tensor['qw'] = Variable(qw_tensor)
    data_tensor['qc'] = Variable(qc_tensor)
    data_tensor['entity'] = Variable(entity_tensor.cuda(), requires_grad=False)
    data_tensor['position'] = Variable(pos_vec.cuda(), requires_grad=False)
    data_tensor['positive_entw'] = Variable(positive_entw_tensor)
    data_tensor['positive_entc'] = Variable(positive_entc_tensor)
    data_tensor['positive_ente'] = Variable(positive_ente_tensor.cuda(), requires_grad=False)
    data_tensor['negative_entw'] = Variable(negative_entw_tensor)
    data_tensor['negative_entc'] = Variable(negative_entc_tensor)
    data_tensor['negative_ente'] = Variable(negative_ente_tensor.cuda(), requires_grad=False)
    data_tensor['positive_relw'] = Variable(positive_relw_tensor)
    data_tensor['positive_rele'] = Variable(positive_rele_tensor.cuda(), requires_grad=False)
    data_tensor['negative_relw'] = Variable(negative_relw_tensor)
    data_tensor['negative_rele'] = Variable(negative_rele_tensor.cuda(), requires_grad=False)

    data_tensor['positive_enti'] = Variable(positive_enti_tensor.cuda())
    data_tensor['negative_enti'] = Variable(negative_enti_tensor.cuda())
    data_tensor['positive_reli'] = Variable(positive_reli_tensor.cuda())
    data_tensor['negative_reli'] = Variable(negative_reli_tensor.cuda())
    data_tensor['cand_ent'] = Variable(cand_idx_tensor.cuda())
    data_tensor['predict'] = Variable(predicts_tensor.cuda())
    data_tensor['positive_type'] = Variable(positive_typew_tensor)
    data_tensor['negative_type'] = Variable(negative_typew_tensor)


    return data_tensor

# ===============================handle test data tensor ===============================

def prepair_test_data(data):
    data = data.sort_values(['qid', 'predict'], ascending=[True, False])
    db_conn = MySQL(ip='10.61.2.166', port=3306, user='zengyutao',
                                       pw='zengyutao', db_name='wikidata')
    test = {'question':[], 'cand_ents':[], 'pos':[], 'entities_vec':[],
            'predict':[], 'cand_vecs': []}
    qid_max = max(data['qid'].tolist())
    for i in range(qid_max):
        question = data[data['qid'] == i, 'question'].iloc[0]
        test['question'].append(question)

        pos = data.loc[data['qid'] == i, 'pos'].iloc[0]
        test['pos'].append(pos)

        predict = data.loc[data['qid'] == i, 'predict']
        test['predict'].append(predict.tolist())

        cand_mid = data.loc[data['qid'] == i, 'topic_words']
        test['cant_ents'].append(cand_mid.tolist())

        vectors = []
        cand_vec_dict = {}
        for cand in cand_mid:
            #print cand
            vec = get_entity_vector(db_conn, cand)
            #print vec
            vec = [float(x) for x in str(vec).split(',')]
            vectors.append(vec)
            cand_vec_dict[cand] = vec

        entities_vec = np.matmul(np.array(predict), np.array(vectors))
        test['entities_vec'].append(entities_vec)

        cand_vec = [cand_vec_dict[c] for c in cand_mid]
        test['cand_vecs'].append(cand_vec)
    test_data = pd.DataFrame()
    for key in test:
        test_data[key] = test.get(key)
    test_data['qid'] = test_data.index
    return test_data


def gen_batch_test_data(data, word_dict, char_dict, args):
    questions = []
    positive_ents = []
    negative_ents = []
    positive_vecs = []
    negative_vecs = []
    entities_vecs = []
    poses = []
    for i in range(0, args.batch_size):
        n = random.randint(0, len(data) - 1)
        row = data.loc[[n]]
        #print (row['question'])
        questions.append(row['question'].loc[n])
        positive_ents.append(row['positive'].loc[n])
        positive_vecs.append(row['positive_vec'].loc[n])
        neg_ents = row['negative'].loc[n]
        #print neg_ents
        neg_vecs = row['negative_vec'].loc[n]
        assert len(neg_ents) == len(neg_vecs), "negative ents and vecs mush have same len"
        if len(neg_ents) == 0:
            negative_ents.append("none")
            negative_vecs.append(np.random.rand(args.entity_dim))
        else:
            m = random.randint(0, len(neg_ents) - 1)
            negative_ents.append(neg_ents[m])
            negative_vecs.append(neg_vecs[m])
        entities_vecs.append(row['entities_vec'].loc[n])
        poses.append(row['pos'].loc[n])


    max_len = 0
    for line in questions:
        if len(word_tokenize(line)) > max_len:
            max_len = len(word_tokenize(line))
    qw_tensor = lines_to_word_tensor(questions, word_dict, max_len)
    qc_tensor = lines_to_char_tensor(questions, char_dict, max_len)
    entity_tensor = torch.rand(args.batch_size, max_len, args.entity_dim)
    #print entity_tensor.size()
    #print type(entity_tensor)
    for i in range(0, args.batch_size):
        entities_vec = entities_vecs[i]
        pos = poses[i]
        pos = ast.literal_eval(pos)
        print (pos)
        for j in range(len(pos)):
            entity_tensor[i][pos[j]] =torch.from_numpy(entities_vec)

    #max_word_len = max(len(w) for w in (positive_ents + positive_))
    positive_entw_tensor = lines_to_word_tensor(positive_ents, word_dict)
    positive_entc_tensor = lines_to_char_tensor(positive_ents, char_dict)
    positive_ente_tensor = torch.Tensor(positive_vecs)

    negative_entw_tensor = lines_to_word_tensor(negative_ents, word_dict)
    negative_entc_tensor = lines_to_char_tensor(negative_ents, char_dict)
    negative_ente_tensor = torch.Tensor(negative_vecs)

    data_tensor = dict()
    data_tensor['qw'] = qw_tensor
    data_tensor['qc'] = qc_tensor
    data_tensor['entity'] = Variable(entity_tensor.cuda(), requires_grad=False)
    data_tensor['positive_entw'] = positive_entw_tensor
    data_tensor['positive_entc'] = positive_entc_tensor
    data_tensor['positive_ente'] = Variable(positive_ente_tensor.cuda(), requires_grad=False)
    data_tensor['negative_entw'] = negative_entw_tensor
    data_tensor['negative_entc'] = negative_entc_tensor
    data_tensor['negative_ente'] = Variable(negative_ente_tensor.cuda(), requires_grad=False)

    return data_tensor


def gen_simgle_test_data(data, word_dict, char_dict, i, entity_dim, entity_dict):
    data = data.sort_values(['qid', 'predict'], ascending=[True, False])
    db_conn = MySQL(ip='10.61.2.166', port=3306, user='zengyutao',
                                       pw='zengyutao', db_name='wikidata')
    test_tensor = {}

    question = data.loc[data['qid'] == i, 'question'].iloc[0]
    max_len = len(word_tokenize(question))
    qw_tensor = Variable(line_to_word_tensor(question, word_dict, max_len))
    qc_tensor = Variable(line_to_char_tensor(question, char_dict, max_len))
    test_tensor['qw_tensor'] = qw_tensor
    test_tensor['qc_tensor'] = qc_tensor

    pos = data.loc[data['qid'] == i, 'pos'].iloc[0]
    predict = data.loc[data['qid'] == i, 'predict'].tolist()

    cand_mid = data.loc[data['qid'] == i, 'topic_words'].tolist()
    test_tensor['cand_mid'] = cand_mid
    vectors = []
    cand_vec_dict = {}
    cand_indice = []
    cant_types = []
    for cand in cand_mid:
        #print cand
        vec = get_entity_vector(db_conn, cand)
        #print vec
        vec = [float(x) for x in str(vec).split(',')]
        vectors.append(vec)
        cand_vec_dict[cand] = vec
        cand_indice.append(entity_dict[cand])
        type_id = get_mid_type(db_conn, cand)
        type_name = get_mid_to_name_mysql(db_conn, type_id)
        cant_types.append(type_name)

    entities_vec = np.matmul(np.array(predict), np.array(vectors))
    test_tensor['cand_indices'] = Variable(torch.LongTensor(cand_indice).cuda())
    test_tensor['predict'] = Variable(torch.Tensor([predict]).cuda())
    # test_tensor['cand_type'] = lines_to_word_tensor(cant_types, word_dict)

    seq_len = len(word_tokenize(question))

    entity_tensor = torch.rand(1, seq_len, entity_dim)
    pos_vec = torch.rand(1, seq_len).fill_(1)
    pos = ast.literal_eval(pos)

    for p in pos:
        entity_tensor[0][p] = torch.from_numpy(entities_vec)
        pos_vec[0][p] = 2

    test_tensor['entity_tensor'] = Variable(entity_tensor.cuda(), requires_grad=False)
    test_tensor['position_tensor'] = Variable(pos_vec.cuda(), requires_grad=False)

    #cand_tensor = [Variable(torch.Tensor([cand_vec_dict[c]]).cuda(), requires_grad=False)for c in cand_mid]
    cand_tensor = []
    # for c in cand_mid:
    #     c_tensor = torch.rand(1, 1, entity_dim)
    #     for x in range(seq_len):
    #         c_tensor[0][0] = torch.Tensor(cand_vec_dict[c])
    #     cand_tensor.append(Variable(c_tensor.cuda()))
    # test_tensor['cande_tensor'] = cand_tensor
    cand_names = [get_mid_to_name_mysql(db_conn, mid) for mid in cand_mid]

    cand_tw = []
    cand_tc = []
    cand_ti = []
    cand_tt = []
    assert (len(cand_names) == len(cant_types) and
            len(cand_names) == len(cand_indice)), "len cands should be equal"
    for idx, line in enumerate(cand_names):
        print (line)
        if line is None and cant_types[idx] is None:
            max_line_len = 0
        elif line is None and cant_types[idx] is not None:
            max_line_len = len(word_tokenize(cant_types[idx]))
        elif line is not None and cant_types[idx] is None:
            max_line_len = len(word_tokenize(line))
        else:
            max_line_len = max(len(word_tokenize(line)), len(word_tokenize(cant_types[idx])))
        l_tw = line_to_word_tensor(line, word_dict, max_line_len)

        l_tc = line_to_char_tensor(line, char_dict, max_line_len)
        print (l_tc.size())
        cand_tw.append(Variable(l_tw))
        cand_tc.append(Variable(l_tc))

        l_tt = line_to_word_tensor(cant_types[idx], word_dict, max_line_len)
        cand_tt.append(Variable(l_tt))
        cand_ti.append(Variable(torch.LongTensor([cand_indice[idx]]).cuda()))

        c_tensor = torch.rand(1, max_line_len, entity_dim)
        for x in range(max_line_len):
            c_tensor[0][x] = torch.Tensor(cand_vec_dict[cand_mid[idx]])
        cand_tensor.append(Variable(c_tensor.cuda()))
    test_tensor['cande_tensor'] = cand_tensor

    test_tensor['candw_tensor'] = cand_tw
    test_tensor['candc_tensor'] = cand_tc
    test_tensor['candi_tensor'] = cand_ti
    test_tensor['cand_type'] = cand_tt
    test_tensor['golden'] = data.loc[data['qid'] == i, 'golden_word'].iloc[0]
    #test_tensor['candw_tensor'] = [lines_to_word_tensor(line, word_dict) for line in cand_names]
    #test_tensor['candc_tensor'] = [lines_to_char_tensor(line, char_dict) for line in cand_names]

    return test_tensor


def get_single_relation_tensor(data, max_cand, word_dict, i, triples, relation_dict):
    db_conn = MySQL(ip='10.61.2.166', port=3306, user='zengyutao',
                                       pw='zengyutao', db_name='wikidata')
    question = data.loc[data['qid'] == i, 'question'].iloc[0]
    seq_len = len(word_tokenize(question))
    rel_tensor = {}
    cand_rels = get_cand_rel(triples, [max_cand])
    cand_tw = []
    cand_te = []
    cand_ti = []
    for rel in cand_rels:
        cand_ti.append(Variable(torch.LongTensor([relation_dict[rel]]).cuda()))
        rel_re = rel.replace('/', ' ').replace('_', ' ')
        print (word_tokenize(rel_re))
        rel_len = len(word_tokenize(rel_re))
        r_tw = line_to_word_tensor(rel_re, word_dict, rel_len)
        cand_tw.append(Variable(r_tw))
        rel_vec = get_relation_vector(db_conn, rel)
        rel_vec = [float(x) for x in str(rel_vec).split(',')]
        #r_te = Variable(torch.Tensor([rel_vec])).cuda()
        r_tensor = torch.rand(1, r_tw.size(1), len(rel_vec))
        for x in range(r_tw.size(1)):
            r_tensor[0][x] = torch.Tensor(rel_vec)
        cand_te.append(Variable(r_tensor.cuda()))
        #cand_te.append(r_te)

    gold_rel = data.loc[data['qid'] == i, 'relation'].iloc[0]

    rel_tensor['candrw_tensor'] = cand_tw
    rel_tensor['candre_tensor'] = cand_te
    rel_tensor['candri_tensor'] = cand_ti
    rel_tensor['golden_relation'] = gold_rel
    rel_tensor['cand_rels'] = cand_rels
    return rel_tensor


# ===============================gen union data==================================

def gen_uion_data(train_data, word_dict, char_dict, args):
    questions = []
    positive_ents = []
    negative_ents = []
    positive_vecs = []
    negative_vecs = []
    entities_vecs = []
    positive_rels = []
    pos_rel_vecs = []
    negative_rels = []
    neg_rel_vecs = []
    poses = []
    for i in range(0, args.batch_size):
        n = random.randint(0, len(train_data) - 1)
        row = train_data.loc[[n]]
        questions.append(row['question'].loc[n])
        positive_ents.append(row['positive'].loc[n])
        positive_vecs.append(row['positive_vec'].loc[n])
        neg_ents = row['negative'].loc[n]
        neg_vecs = row['negative_vec'].loc[n]
        assert len(neg_ents) == len(neg_vecs), "negative ents and vecs mush have same len"
        if len(neg_ents) == 0:
            negative_ents.append("none")
            negative_vecs.append(np.random.rand(args.entity_dim))
        else:
            m = random.randint(0, len(neg_ents) - 1)
            negative_ents.append(neg_ents[m])
            negative_vecs.append(neg_vecs[m])
        entities_vecs.append(row['entities_vec'].loc[n])
        poses.append(row['pos'].loc[n])

        positive_rels.append(row['pos_rel'].loc[n])
        pos_rel_vecs.append(row['pos_rel_vec'].loc[n])

        neg_rels = row['neg_rel'].loc[n]
        neg_rel_vec = row['neg_rel_vec'].loc[n]

        assert len(neg_rels) == len(neg_rel_vec), "negative rels and vecs mush have the same len"
        if len(neg_rels) == 0:
            negative_rels.append("none")
            neg_rel_vecs.append(np.random.rand(args.entity_dim))
        else:
            m = random.randint(0, len(neg_rels) - 1)
            negative_rels.append(neg_rels[m])
            neg_rel_vecs.append(neg_rel_vec[m])


    qw_tensor = lines_to_word_tensor(questions, word_dict)
    qc_tensor = lines_to_char_tensor(questions, char_dict)
    max_len = 0
    for line in questions:
        if len(word_tokenize(line)) > max_len:
            max_len = len(word_tokenize(line))
    entity_tensor = torch.rand(args.batch_size, max_len, args.entity_dim)



    for i in range(0, args.batch_size):
        entities_vec = entities_vecs[i]
        pos = poses[i]
        pos = ast.literal_eval(pos)
        #print (pos)
        for j in range(len(pos)):
            entity_tensor[i][pos[j]] = torch.from_numpy(entities_vec)

    positive_entw_tensor = lines_to_word_tensor(positive_ents, word_dict)
    positive_entc_tensor = lines_to_char_tensor(positive_ents, char_dict)
    #positive_ente_tensor = torch.Tensor(positive_vecs)

    negative_entw_tensor = lines_to_word_tensor(negative_ents, word_dict)
    negative_entc_tensor = lines_to_char_tensor(negative_ents, char_dict)
    #negative_ente_tensor = torch.Tensor(negative_vecs)
    print ("============rel tensor sizes ===========")
    positive_relw_tensor = lines_to_word_tensor(positive_rels, word_dict)
    print (positive_relw_tensor.data.size())
    #positive_rele_tensor = torch.Tensor(pos_rel_vecs)

    negative_relw_tensor = lines_to_word_tensor(negative_rels, word_dict)
    print (negative_relw_tensor.data.size())
    #negative_rele_tensor = torch.Tensor(neg_rel_vecs)

    positive_ente_tensor = torch.rand(args.batch_size, positive_entw_tensor.size(1), args.entity_dim)
    negative_ente_tensor = torch.rand(args.batch_size, negative_entw_tensor.size(1), args.entity_dim)
    positive_rele_tensor = torch.rand(args.batch_size, positive_relw_tensor.size(1), args.entity_dim)
    negative_rele_tensor = torch.rand(args.batch_size, negative_relw_tensor.size(1), args.entity_dim)
    for i in range(args.batch_size):
        for x in range(positive_entw_tensor.size(1)):
                positive_ente_tensor[i][x] = torch.Tensor(positive_vecs[i])
        for xn in range(negative_entw_tensor.size(1)):
                negative_ente_tensor[i][xn] = torch.Tensor(negative_vecs[i])
        for y in range(positive_relw_tensor.size(1)):
                positive_rele_tensor[i][y] = torch.Tensor(pos_rel_vecs[i])
        for yn in range(negative_relw_tensor.size(1)):
                negative_rele_tensor[i][yn] = torch.Tensor(neg_rel_vecs[i])

    data_tensor = dict()
    data_tensor['qw'] = qw_tensor
    data_tensor['qc'] = qc_tensor
    data_tensor['entity'] = Variable(entity_tensor.cuda(), requires_grad=False)
    data_tensor['positive_entw'] = positive_entw_tensor
    data_tensor['positive_entc'] = positive_entc_tensor
    data_tensor['positive_ente'] = Variable(positive_ente_tensor.cuda(), requires_grad=False)
    data_tensor['negative_entw'] = negative_entw_tensor
    data_tensor['negative_entc'] = negative_entc_tensor
    data_tensor['negative_ente'] = Variable(negative_ente_tensor.cuda(), requires_grad=False)
    data_tensor['positive_relw'] = positive_relw_tensor
    data_tensor['positive_rele'] = Variable(positive_rele_tensor.cuda(), requires_grad=False)
    data_tensor['negative_relw'] = negative_relw_tensor
    data_tensor['negative_rele'] = Variable(negative_rele_tensor.cuda(), requires_grad=False)

    return data_tensor