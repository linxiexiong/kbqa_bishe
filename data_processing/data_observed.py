import pandas as pd
from utils.freebase_wiki import load_pkl_file
import os
import sys
import numpy as np
from mysql import MySQL
from entity_link.features import get_entity_vector

def observer(data_file):
    mid_name = pd.read_csv('../datas/fb2w.nt', sep='\t', header=None)
    mid_name.columns=['fb', 'rel', 'wiki']
    print (mid_name[0:10])
    mid_name['fb'] = mid_name['fb'].apply(
        lambda x: x.replace("<http://rdf.freebase.com/ns", "").replace(">","").replace(".", "/"))
    mid_name['wiki'] = mid_name['wiki'].apply(
        lambda x: x.replace("<", "").replace("> .", ""))
    mid_name_dict = dict(zip(mid_name.fb, mid_name.wiki))
    name_mid_dict = dict(zip(mid_name.wiki, mid_name.fb))
    print ({k: mid_name_dict[k] for k in list(mid_name_dict)[:20]})
    return mid_name_dict, name_mid_dict

#dn = observer('../datas/SimpleQuestions_v2/annotated_fb_data_train.txt')[0]

#if not os.path.exists('../datas/vocab/knwn_wid_vocab.pkl'):
#    print("Known Entities Vocab PKL missing")
#    sys.exit()
#(knwid2idx, idx2knwid) = load_pkl_file('../datas/vocab/knwn_wid_vocab.pkl')
#print (type(knwid2idx), type(idx2knwid))
#first20pairs = {k: knwid2idx[k] for k in list(knwid2idx)[:20]}
#wid2labels = load_pkl_file('../datas/vocab/wid2labels_vocab.pkl')
#print type(wid2labels)

def explode(df, lst_cols, fill_value=''):
    # make sure `lst_cols` is a list
    if lst_cols and not isinstance(lst_cols, list):
        lst_cols = [lst_cols]
    # all columns except `lst_cols`
    idx_cols = df.columns.difference(lst_cols)

    # calculate lengths of lists
    lens = df[lst_cols[0]].str.len()

    if (lens > 0).all():
        # ALL lists in cells aren't empty
        return pd.DataFrame({
            col:np.repeat(df[col].values, df[lst_cols[0]].str.len())
            for col in idx_cols
        }).assign(**{col:np.concatenate(df[col].values) for col in lst_cols}).loc[:, df.columns]
    else:
        # at least one list in cells is empty
        return pd.DataFrame({
            col:np.repeat(df[col].values, df[lst_cols[0]].str.len())
            for col in idx_cols
        }).assign(**{col:np.concatenate(df[col].values) for col in lst_cols}).append(
            df.loc[lens==0, idx_cols]).fillna(fill_value).loc[:, df.columns]

#df = pd.read_csv('/Users/linxiexiong/Downloads/test.csv')
#lst_cols = ['topic_words','topic_words_names']
#print df
#df['topic_words'] = df['topic_words'].apply(lambda x: x.split(','))
#df['topic_words_names'] = df['topic_words_names'].apply(lambda x: x.split(','))
#print len(df.topic_words[0].split(',')) , len((df.topic_words_names[0]))
#w_sq_datas = explode(df, lst_cols, fill_value='')
# w_sq_datas = w_sq_datas[0:50]
#w_sq_datas['label'] = w_sq_datas.apply(lambda x: 1 if x['golden_word'] == x['topic_words'] else 0, axis=1)
#w_sq_datas['label'] = w_sq_datas.apply()

#print w_sq_datas
# db_conn = MySQL(ip='10.61.2.166', port=3306, user='zengyutao',
#                                   pw='zengyutao', db_name='wikidata')
# for i in range(100):
#     mid = 'm.04whkz5'
#     table_name = 'mid2name'
#     query = "select name from %s where mid = '%s' " % (table_name, mid)
#     print query
#     vec = db_conn.search(query)
#
#     #print vec[0].split(',')
# db_conn.connect.close()

def get_topic_word_pos(str, str1):
    s = str.strip().split(' ')
    s1 = str1.strip().split(' ')
    leng = max(0, len(s) - len(s1))
    #print (leng)
    idxs = []
    for idx, w in enumerate(s1):
        if len(w) == 0 or len(w) == 1:
            continue
        if w[0] == '#' and w[len(w)-1] == '#':
            for i in range(0, leng+1):
                idxs.append(idx + i)
            #print (idxs)
            return idxs
    return idxs

def has_four(str):
    s = str.strip().split(' ')
    cnt = 0
    for w in s:
        if w[0] == '#' and w[len(w)-1] == '#':
            cnt += 1
    if cnt > 1 :
        return str
    return ' '

# dataa = pd.read_csv('../datas/SimpleQuestions_v2/small_train_100.txt', header=None, sep='\t')
# dataa.columns = ['subid', 'rel', 'objid', 'question']
# data = pd.read_csv('../datas/topic_words/topic_words_pos_train_100.txt', header=None, sep='\t')
# data.columns = ['sid', 'cids', 'q']
# #data['is'] = data['q'].apply(lambda x: has_four(x))
# #print (data[data['is'] != ' '])
# data = pd.concat([dataa,data], axis=1)
# print type(data)
#
# data['pos'] = data.apply(lambda x: get_topic_word_pos(x['question'], x['q']), axis=1)
# print(data.loc[:,['q', 'question', 'pos']])
# print (data.columns)


def get_mid_to_name_mysql(db_conn, mid):
        table_name = 'mid2name'
        query = "select name from %s where mid = '%s' " % (table_name, mid)
        #print query
        name = db_conn.search(query)
        if name is not None and len(name) >= 1:
            return name[0]
        return None

def mul_vec(weight, vectors):
    assert len(weight) == len(vectors) , "must the same length"
    print (len(weight))
    new_vecs = []
    for i in range(len(weight)):
        #print vectors[i]
        print type(i)
        print type(weight)
        print weight[i]
        we = weight[i]
        vector = vectors[i]
        #print (vectors[i][0])
        new_vec = []
        for j in range(len(vector)):
            v = vector[j]
            new_vec.append(we * v)
        new_vecs.append(new_vecs)

    vec = np.sum(new_vecs, axis=1)
    return vec

data = pd.read_csv('../entity_link/head_10.csv')
data = data.sort_values(['qid', 'predict'], ascending=[True, False]).reset_index(drop=True)
qid_max = max(data['qid'].tolist())
print (qid_max)
train = {'question':[], 'cand_ent':[], 'pos':[], 'entities_vec':[], 'positive_vec':[],
             'positive':[], 'negtive':[], 'predict':[], 'negative_vec':[]}
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
    #print cand_mid
    #data['vectors'] = data['vector'].apply(lambda x: x.split(','))
    train['cand_ent'].append(cand_mid.tolist())
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
    #entities_vec = mul_vec(predict.tolist(), vectors)
    #print entities_vec

    positive = data.loc[data['qid'] == i, 'golden_word_name'].iloc[0]
    train['positive'].append(positive)

    # cands_name = data.loc[data['qid']==i, 'topic_words_names']
    # #print cands_name
    # neg = [w for w in cands_name if w != positive]
    # train['negtive'].append(neg)

    positive_mid = data.loc[data['qid'] == i, 'golden_word'].iloc[0]
    positive_vec = cand_vec_dict[positive_mid]
    neg_vec = [cand_vec_dict[neg_mid] for neg_mid in cand_mid if neg_mid != positive_mid]
    neg = [get_mid_to_name_mysql(db_conn, neg_mid) for neg_mid in cand_mid if neg_mid != positive_mid]
    train['negtive'].append(neg)
    #print len(cand_id),len(neg_vec)
    train['positive_vec'].append(positive_vec)
    train['negative_vec'].append(neg_vec)
    #print (len(neg), len(neg_vec))

train_data = pd.DataFrame()
for key in train:
    #print (key)
    train_data[key] = train.get(key)

#print train_data[:10]
#lst_cols = ['negtive', 'negative_vec']
for i in range(len(train_data)):
    row = train_data.loc[[i]]
    print row['question']
    print(row['question'].tolist())
#w_sq_datas = explode(df, lst_cols, fill_value='')
#print w_sq_datas
    #entity =

    #print (positive, cands_name, neg)


#print (data.loc[:, ['qid', 'predict']])
