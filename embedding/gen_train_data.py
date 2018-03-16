import sys
sys.path.append('..')
sys.path.append('../..')
# if torch.cuda.is_available():
#     import torch.cuda as torch
# else:
import torch
import torch.nn as nn
from torch.autograd import Variable
from data_processing.mysql import MySQL, get_mid_to_name_mysql, get_relation_vector
from entity_link.features import get_entity_vector
from type_handle.data_handle import lines_to_word_tensor, lines_to_char_tensor, \
    line_to_word_tensor,line_to_char_tensor
from nltk.tokenize import word_tokenize
import numpy as np
import random
import pandas as pd
import ast


def get_cand_rel(triples, mids):
    #triples = pd.read_csv('../datas/FB5M-triples.txt', header=None, sep='\t')
    #triples.columns = ['subject', 'relation', 'object']
    rels = set()
    for mid in mids:
        rel = triples.loc[triples['subject'] == mid, 'relation'].tolist()
        for r in rel:
            rels.add(r)
    print (len(rels))
    return list(rels)

def prepair_data(data, triples):
    data = data.sort_values(['qid', 'predict'], ascending=[True, False])
    qid_max = max(data['qid'].tolist())
    print (qid_max)
    train = {'question':[], 'cand_ent':[], 'pos':[], 'entities_vec':[], 'positive_vec':[],
             'positive':[], 'negative':[], 'predict':[], 'negative_vec':[], 'cand_rel':[],
             'pos_rel':[], 'pos_rel_vec':[], 'neg_rel':[], 'neg_rel_vec':[]}
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
        for cand in cand_mid:
            #print cand
            vec = get_entity_vector(db_conn, cand)
            #print vec
            vec = [float(x) for x in str(vec).split(',')]
            vectors.append(vec)
            cand_vec_dict[cand] = vec

        entities_vec = np.matmul(np.array(predict), np.array(vectors))
        #entities_vec = mul_vec(predict.tolist(), vectors)
        train['entities_vec'].append(entities_vec)

        positive = data.loc[data['qid'] == i, 'golden_word_name'].iloc[0]
        train['positive'].append(positive)

        # cands_name = data.loc[data['qid']==i, 'topic_words_names']
        # neg_name = [w for w in cands_name if w != positive]
        # train['negtive'].append(neg_name)

        positive_mid = data.loc[data['qid'] == i, 'golden_word'].iloc[0]
        positive_vec = cand_vec_dict[positive_mid]
        neg_vec = [cand_vec_dict[neg_mid] for neg_mid in cand_mid if neg_mid != positive_mid]
        neg = [get_mid_to_name_mysql(db_conn, neg_mid) for neg_mid in cand_mid if neg_mid != positive_mid]
        train['negative'].append(neg)
        train['positive_vec'].append(positive_vec)
        train['negative_vec'].append(neg_vec)

        #print ("================gen pos rel ==================")
        positive_rel = data.loc[data['qid'] == i, 'relation'].iloc[0]
        
        pos_rel_vec = get_relation_vector(db_conn, positive_rel)
        pos_rel_vec = [float(x) for x in str(pos_rel_vec).split(',')]
        positive_rel_re = positive_rel.replace('/', ' ')
        #print (positive_rel)
        #print (pos_rel_vec)
        train['pos_rel'].append(positive_rel_re)
        train['pos_rel_vec'].append(pos_rel_vec)

        #print ("----------------get cand rel----------")
        cand_rel = get_cand_rel(triples, [positive_mid])
        train['cand_rel'].append(cand_rel)

        #print ("--------------get neg rel --------------")
        neg_rel = [rel for rel in cand_rel if rel != positive_rel]
        neg_rel_vec = []
        neg_rel_re = []
        for n in neg_rel:
            vec = get_relation_vector(db_conn, n)
            neg_v = [float(x) for x in str(vec).split(',')]
            neg_rel_vec.append(neg_v)
            neg_rel_re.append(n.replace('/', ' '))

        #neg_rel = [rel.replace('/', ' ') for rel in cand_rel if rel.replace('/', ' ') != positive_rel]
        #print (neg_rel)
        #print (neg_rel_vec)
        train['neg_rel'].append(neg_rel_re)
        train['neg_rel_vec'].append(neg_rel_vec)


    train_data = pd.DataFrame()
    for key in train:
        train_data[key] = train.get(key)
    return train_data


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
    poses = []
    for i in range(0, args.batch_size):
        n = random.randint(0, len(train_data) - 1)
        row = train_data.loc[[n]]
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

        positive_rels.append(row['pos_rel'].loc[n])
        pos_rel_vecs.append(row['pos_rel_vec'].loc[n])
        #print (pos_rel_vecs)
        neg_rels = row['neg_rel'].loc[n]
        neg_rel_vec = row['neg_rel_vec'].loc[n]
        #print (neg_rels)
        #print (neg_rel_vec)
        #print (len(neg_rels), len(neg_rel_vec))
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
    #print entity_tensor.size()
    #print type(entity_tensor)
    for i in range(0, args.batch_size):
        entities_vec = entities_vecs[i]
        pos = poses[i]
        pos = ast.literal_eval(pos)
        #print (pos)
        for j in range(len(pos)):
            entity_tensor[i][pos[j]] =torch.from_numpy(entities_vec)
    positive_entw_tensor = lines_to_word_tensor(positive_ents, word_dict)
    positive_entc_tensor = lines_to_char_tensor(positive_ents, char_dict)
    positive_ente_tensor = torch.Tensor(positive_vecs)

    negative_entw_tensor = lines_to_word_tensor(negative_ents, word_dict)
    negative_entc_tensor = lines_to_char_tensor(negative_ents, char_dict)
    negative_ente_tensor = torch.Tensor(negative_vecs)
    print ("============rel tensor sizes ===========")
    positive_relw_tensor = lines_to_word_tensor(positive_rels, word_dict)
    print (positive_relw_tensor.data.size())
    positive_rele_tensor = torch.Tensor(pos_rel_vecs)
    print (positive_rele_tensor.size())

    negative_relw_tensor = lines_to_word_tensor(negative_rels, word_dict)
    print (negative_relw_tensor.data.size())
    negative_rele_tensor = torch.Tensor(neg_rel_vecs)
    print (negative_rele_tensor.size())

    data_tensor = dict()
    data_tensor['qw'] = qw_tensor
    data_tensor['qc'] = qc_tensor
    data_tensor['entity'] = Variable(entity_tensor, requires_grad=True)
    data_tensor['positive_entw'] = positive_entw_tensor
    data_tensor['positive_entc'] = positive_entc_tensor
    data_tensor['positive_ente'] = Variable(positive_ente_tensor, requires_grad=True)
    data_tensor['negative_entw'] = negative_entw_tensor
    data_tensor['negative_entc'] = negative_entc_tensor
    data_tensor['negative_ente'] = Variable(negative_ente_tensor, requires_grad=True)
    data_tensor['positive_relw'] = positive_relw_tensor
    data_tensor['positive_rele'] = Variable(positive_rele_tensor, requires_grad=True)
    data_tensor['negative_relw'] = negative_relw_tensor
    data_tensor['negative_rele'] = Variable(negative_rele_tensor, requires_grad=True)

    return data_tensor


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

    qw_tensor = lines_to_word_tensor(questions, word_dict)
    qc_tensor = lines_to_char_tensor(questions, char_dict)
    max_len = 0
    for line in questions:
        if len(word_tokenize(line)) > max_len:
            max_len = len(word_tokenize(line))
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
    positive_entw_tensor = lines_to_word_tensor(positive_ents, word_dict)
    positive_entc_tensor = lines_to_char_tensor(positive_ents, char_dict)
    positive_ente_tensor = torch.Tensor(positive_vecs)

    negative_entw_tensor = lines_to_word_tensor(negative_ents, word_dict)
    negative_entc_tensor = lines_to_char_tensor(negative_ents, char_dict)
    negative_ente_tensor = torch.Tensor(negative_vecs)

    data_tensor = dict()
    data_tensor['qw'] = qw_tensor
    data_tensor['qc'] = qc_tensor
    data_tensor['entity'] = Variable(entity_tensor, requires_grad=True)
    data_tensor['positive_entw'] = positive_entw_tensor
    data_tensor['positive_entc'] = positive_entc_tensor
    data_tensor['positive_ente'] = Variable(positive_ente_tensor, requires_grad=True)
    data_tensor['negative_entw'] = negative_entw_tensor
    data_tensor['negative_entc'] = negative_entc_tensor
    data_tensor['negative_ente'] = Variable(negative_ente_tensor, requires_grad=True)

    return data_tensor


def gen_simgle_test_data(data, word_dict, char_dict, i, entity_dim):
    data = data.sort_values(['qid', 'predict'], ascending=[True, False])
    db_conn = MySQL(ip='10.61.2.166', port=3306, user='zengyutao',
                                       pw='zengyutao', db_name='wikidata')
    test_tensor = {}

    question = data.loc[data['qid'] == i, 'question'].iloc[0]
    qw_tensor = line_to_word_tensor(question, word_dict)
    qc_tensor = line_to_char_tensor(question, char_dict)
    test_tensor['qw_tensor'] = qw_tensor
    test_tensor['qc_tensor'] = qc_tensor

    pos = data.loc[data['qid'] == i, 'pos'].iloc[0]
    predict = data.loc[data['qid'] == i, 'predict'].tolist()

    cand_mid = data.loc[data['qid'] == i, 'topic_words'].tolist()
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
    #print (entities_vec)
    seq_len = len(word_tokenize(question))

    entity_tensor = torch.rand(1, seq_len, entity_dim)
    pos = ast.literal_eval(pos)

    for p in pos:
        entity_tensor[0][p] = torch.from_numpy(entities_vec)
    test_tensor['entity_tensor'] = Variable(entity_tensor, requires_grad=True)

    cand_tensor = [Variable(torch.Tensor([cand_vec_dict[c]]), requires_grad=True)for c in cand_mid]

    test_tensor['cande_tensor'] = cand_tensor
    cand_names = [get_mid_to_name_mysql(db_conn, mid) for mid in cand_mid]
    cand_tw = []
    cand_tc = []
    for line in cand_names:
        print (line)
        l_tw = line_to_word_tensor(line, word_dict)
        print (l_tw.size())
        l_tc = line_to_char_tensor(line, char_dict)
        cand_tw.append(l_tw)
        cand_tc.append(l_tc)
    #print (len(cand_tw))
    test_tensor['candw_tensor'] = cand_tw
    test_tensor['candc_tensor'] = cand_tc
    test_tensor['golden'] = data.loc[data['qid'] == i, 'golden_word'].iloc[0]
    #test_tensor['candw_tensor'] = [lines_to_word_tensor(line, word_dict) for line in cand_names]
    #test_tensor['candc_tensor'] = [lines_to_char_tensor(line, char_dict) for line in cand_names]
    test_tensor['cand_mid'] = cand_mid


    return test_tensor

def get_single_relation_tensor(data, max_cand, word_dict, i, triples):
    db_conn = MySQL(ip='10.61.2.166', port=3306, user='zengyutao',
                                       pw='zengyutao', db_name='wikidata')
    rel_tensor = {}
    cand_rels = get_cand_rel(triples, [max_cand])
    cand_tw = []
    cand_te = []
    for rel in cand_rels:
        rel_vec = get_relation_vector(db_conn, rel)
        rel_vec = [float(x) for x in str(rel_vec).split(',')]
        r_te = Variable(torch.Tensor([rel_vec])).cuda()
        cand_te.append(r_te)
        rel_re = rel.replace('/', ' ')
        print (word_tokenize(rel_re))
        r_tw = line_to_word_tensor(rel_re, word_dict)
        cand_tw.append(r_tw)
    gold_rel = data.loc[data['qid'] == i, 'relation'].iloc[0]

    rel_tensor['candrw_tensor'] = cand_tw
    rel_tensor['candre_tensor'] = cand_te
    rel_tensor['golden_relation'] = gold_rel
    rel_tensor['cand_rels'] = cand_rels
    return rel_tensor

