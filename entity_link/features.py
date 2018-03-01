from __future__ import unicode_literals, print_function, division
import sys
sys.path.append('..')
sys.path.append('../..')
#reload(sys)
#sys.setdefaultencoding('utf-8')
from data_processing.load_datas import DataReader
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
import re
from data_processing.mysql import MySQL


def data_load(stage="train"):
    mid_name_file = "../datas/mid2name.tsv"
    mid_qid_file = "../datas/fb2w.nt"

    if stage == "train":
        topic_words_file = '../datas/topic_words/train.fuzzy_p2_linker.simple_linker.original.union'
        sq_data_file = '../datas/SimpleQuestions_v2/annotated_fb_data_train.txt'
        #topic_words_file = '../datas/topic_words/small_topic_words_1w.txt'
        #sq_data_file = '../datas/SimpleQuestions_v2/small_train_1w.txt'
    elif stage == "valid":
        topic_words_file = '../datas/topic_words/valid.fuzzy_p2_linker.simple_linker.original.union'
        sq_data_file = '../datas/SimpleQuestions_v2/annotated_fb_data_valid.txt'
        #topic_words_file = '../datas/topic_words/small_topic_words_1w_valid.txt'
        #sq_data_file = '../datas/SimpleQuestions_v2/small_valid_1w.txt'
    elif stage == "test":
        topic_words_file = '../datas/topic_words/test.fuzzy_p2_linker.simple_linker.original.union'
        sq_data_file = '../datas/SimpleQuestions_v2/annotated_fb_data_test.txt'
        #topic_words_file = '../datas/topic_words/small_topic_words_test.txt'
        #sq_data_file = '../datas/SimpleQuestions_v2/small_test_1w.txt'
    else:
        raise ValueError('invalid stage, which should be one of {train, valid, test}')

    datas = DataReader(mid_name_file, mid_qid_file,
                       topic_words_file, sq_data_file)
    datas.read_sq_data_pd()
    sq_datas = datas.load_topic_words(stage, datas.sq_dataset)
    #sq_datas.to_csv('../datas/sq_data_' + stage + '.csv', index=False)
    #print (sq_datas[0: 10])
    return sq_datas


def negative_sampling(sq_datas):
    sq_datas['label'] = 1
    ne_datas = sq_datas.copy()
    for i in range(5):
        ne_datas['label'] = 0
        ne_datas['golden_word'] = ne_datas['topic_words'].apply(lambda x: x[np.random.randint(len(x))])
        sq_datas = pd.concat([sq_datas, ne_datas], ignore_index=True)
    print (len(sq_datas[sq_datas['label'] == 0]))
    print (len(sq_datas[sq_datas['label'] == 1]))
    return sq_datas


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


def whole_sample(sq_datas):
    # sq_datas['label'] = 1
    # ne_datas = sq_datas.copy()
    lst_cols = ['topic_words','topic_words_names', 'word_score']
    w_sq_datas = explode(sq_datas, lst_cols, fill_value='')
    # w_sq_datas = w_sq_datas[0:50]
    w_sq_datas['label'] = w_sq_datas.apply(lambda x: 1 if x['golden_word'] == x['topic_words'] else 0, axis=1)
    #w_sq_datas['label'] = w_sq_datas.apply()
    return w_sq_datas


def longest_common_substring(left, right):
    if left is None or right is None:
        return 0, None
    left = re.sub('\s', '', left)
    right = re.sub('\s', '', right)
    m = np.zeros((len(left)+1, len(right)+1))
    max_len = 0
    last_pos = 0
    for i in range(len(left)):
        for j in range(len(right)):
            if left[i] == right[j]:
                m[i+1][j+1] = m[i][j] + 1
                if m[i+1][j+1] > max_len:
                    max_len = m[i+1][j+1]
                    last_pos = i + 1
    return max_len, left[int(last_pos-max_len): int(last_pos)]


def get_lcs_pq(df):
    left = df['question']
    right = df['topic_words_names']
    if left is None or right is None:
        return 0
    common_len, _ = longest_common_substring(left.lower(), right.lower())
    question_len = len(left)
    if question_len == 0:
        return 0
    portion = common_len / question_len
    return portion


def get_lcs_pe(df):
    left = df['question']
    right = df['topic_words_names']
    if right is None or left is None:
        return 0
    common_len, _ = longest_common_substring(left.lower(), right.lower())
    entity_len = len(right)
    if entity_len == 0:
        return 0
    portion = common_len / entity_len
    return portion


def longest_common_words(left, right):
    if left is None or right is None:
        return 0
    m = np.zeros((len(left) + 1, len(right) + 1))
    max_len = 0
    for i in range(0, len(left)):
        for j in range(0, len(right)):
            if left[i] == right[j]:
                m[i+1][j+1] = m[i][j] + 1
                if m[i+1][j+1] > max_len:
                    max_len = m[i+1][j+1]
            else:
                m[i+1][j+1] = 0
    return max_len


def get_lcw_pq(df):
    left = df['question']
    right = df['topic_words_names']
    if left is None or right is None:
        return 0
    left = word_tokenize(left.lower())
    right = word_tokenize(right.lower())

    lcw = longest_common_words(left, right)
    question_len = len(left)
    if question_len == 0:
        return 0
    portion = lcw / question_len
    return portion


def get_lcw_pe(df):
    left = df['question']
    right = df['topic_words_names']
    if right is None or left is None:
        return 0
    left = word_tokenize(left.strip().lower())
    right = word_tokenize(right.strip().lower())
    lcw = longest_common_words(left, right)
    entity_len = len(right)
    if entity_len == 0:
        return 0
    portion = lcw / entity_len
    return portion


def get_tf_idf(golden_word, index, tf_idf):
    if golden_word is None:
        return 0
    golden_word_token = word_tokenize(golden_word)
    score = 0
    for gw in golden_word_token:
        gw = gw.lower()
        if gw not in tf_idf:
            score += 1
        else:
            score += tf_idf[gw][index]
    if len(golden_word_token) == 0:
        return 0
    avg_score = score / len(golden_word_token)
    return avg_score


def get_entity_vector(conn, mid):
    table_name = 'entity2vec_2'
    query = "select vector from %s where mid = '%s' " % (table_name, mid)
    vector = conn.search(query)
    # print (vector)
    if vector is not None and len(vector) >= 1:
        return vector[0]
    return None


def mid_type(conn, mid):
    table_name = 'mid2type'
    query = "select notable_type from %s where mid = '%s' " % (table_name, mid)
    vector = conn.search(query)
    # print (vector)
    if vector is not None and len(vector) >= 1:
        return vector[0]
    return None


def feature_select(sq_datas):
    feature_columns = ['lcs_pq', 'lcs_pe', 'lcw_pq', 'lcw_pe', 'word_score']

    sq_datas['lcs_pq'] = sq_datas.apply(get_lcs_pq, axis=1)
    sq_datas['lcs_pe'] = sq_datas.apply(get_lcs_pe, axis=1)
    sq_datas['lcw_pq'] = sq_datas.apply(get_lcw_pq, axis=1)
    sq_datas['lcw_pe'] = sq_datas.apply(get_lcw_pe, axis=1)
    # sq_datas['word_score'] = sq_datas['word_score'].replace(to_replace="[a-zA-Z]", value='', regex=True, inplace=True)
    # sq_datas['word_score'] = sq_datas['word_score'].apply(pd.to_numeric, errors='coerce')


    # sq_datas['qid'] = sq_datas.index
    sq_datas['token_question'] = sq_datas['question'].apply(lambda x: word_tokenize(x))
    tf = sq_datas.token_question.apply(pd.value_counts).fillna(0)
    idf = np.log((len(sq_datas) + 1) / (tf.gt(0).sum() + 1))
    tf_idf = tf * idf
    sq_datas['tf_idf'] = sq_datas.apply(lambda x: get_tf_idf(x.topic_words_names,
                                                             x.qid,
                                                             tf_idf), axis=1)

    db_conn = MySQL(ip='10.61.2.166', port=3306, user='zengyutao',
                                  pw='zengyutao', db_name='wikidata')

    sq_datas['vector'] = sq_datas.apply(lambda x: get_entity_vector(db_conn, x['topic_words']), axis=1)
    vec_cols = []
    for i in range(50):
        feature_columns.append("vec_" + str(i))
        vec_cols.append("vec_" + str(i))
    # print (sq_datas)
    sq_datas[vec_cols] = pd.DataFrame(sq_datas['vector'].str.split(',').values.tolist())
    sq_datas[vec_cols] = sq_datas[vec_cols].astype(float)
    sq_datas = sq_datas.fillna(value=0)

    sq_datas['type'] = sq_datas.apply(lambda x: mid_type(db_conn, x['topic_words']), axis=1)
    dummy = pd.get_dummies(sq_datas['type'], prefix='type')
    sq_datas = pd.concat([sq_datas, dummy], axis=1)
    for col in dummy.columns:
        feature_columns.append(col)
    print (sq_datas[0: 50])
    #sq_datas = sq_datas.sample(frac=0.6)
    return sq_datas[feature_columns], sq_datas['label']

