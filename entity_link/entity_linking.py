from models import XGBoostModel
from features import data_load, feature_select, negative_sampling, whole_sample
import pandas as pd
from sklearn.metrics import log_loss
import numpy as np


def entity_linking():
    #sq_data_train = data_load("train")
    # print len(sq_data_train.topic_words[0]), (sq_data_train.topic_words[0][0]), len(sq_data_train.topic_words_names[0])
    # print len(sq_data_train.topic_words[1]), len(sq_data_train.topic_words_names[1])
    #sq_data_train_with_ns = whole_sample(sq_data_train)

    # print (sq_data_train_with_ns[0:50])
    #sq_data_train_with_ns.to_csv('../datas/sq_data_whole_train.csv')
    # print len(sq_data_train_with_ns[sq_data_train_with_ns.label == 1])
    #train_features, train_label = feature_select(sq_data_train_with_ns)

    #sq_data_valid = data_load("valid")
    #sq_data_valid_with_ns = whole_sample(sq_data_valid)
    #print sq_data_valid_with_ns['word_score'].tolist()
    #print (type(sq_data_valid_with_ns.loc[50,'word_score']))
    #sq_data_valid_with_ns.to_csv('../datas/sq_data_whole_valid.csv')
    #valid_features, valid_label = feature_select(sq_data_valid_with_ns)

    #sq_data_test = data_load("test")
    #sq_data_test_with_ns = whole_sample(sq_data_test)
    #sq_data_test_with_ns.to_csv('../datas/sq_data_whole_test.csv')
    #test_features, test_label = feature_select(sq_data_test_with_ns)
    train_features_file = '../datas/features/small_train_features.csv'
    train_label_file = '../datas/features/small_train_labels.csv'
    test_features_file = '../datas/features/small_test_features.csv'
    test_label_file = '../datas/features/small_test_labels.csv'
    train_features = pd.read_csv(train_features_file)
    train_label = pd.read_csv(train_label_file)
    test_features = pd.read_csv(test_features_file)
    test_label = pd.read_csv(test_label_file)
    sq_data_test_with_ns = pd.read_csv('../datas/SimpleQuestions_v2/small_test_whole.csv')
    train_features, test_features = fix_columns(train_features, test_features)
    #sq_data_test = data_load("test")
    xgb = XGBoostModel(max_depth=4,
                       learning_rate=0.05,
                       n_estimators=200).xgb_ranker()
    clf = xgb.fit(train_features, train_label)
    predict = clf.predict_proba(test_features)
    #print predict
    sq_data_test_with_ns['predict'] = predict[:, 1]
    sq_data_test_with_ns['word_score_sum'] = sq_data_test_with_ns.groupby('qid').agg({'word_score': sum})
    sq_data_test_with_ns['word_score_norm'] = sq_data_test_with_ns.apply(
            lambda x: x['word_score'] / x['word_score_sum'] if x['word_score_sum'] != 0 else 0, axis=1)

    print sq_data_test_with_ns[0: 100]
    sq_data_test_with_ns['score'] = sq_data_test_with_ns.apply(
        lambda x: float(x['predict']) + float(x['word_score']), axis=1)

    hit1 = evaluate(sq_data_test_with_ns, 1)
    hit5 = evaluate(sq_data_test_with_ns, 5)
    hit10 = evaluate(sq_data_test_with_ns, 50)
    print (hit1, hit5, hit10)
    #print log_loss(sq_data_valid_with_ns['label'], predict)


def evaluate(df, n):
    df = df.sort_values(['predict'], ascending=False).groupby('qid').head(n)
    df.to_csv('head_' + str(n) + ".csv")
    df['is_equal'] = df.apply(lambda x: 1 if x['golden_word'] == x['topic_words'] else 0, axis=1)
    dff = df.groupby(['qid'], as_index=False).agg({'is_equal': sum})
    dff[dff.is_equal == 0].to_csv('error_sample.csv')
    #df[df.qid == dff[dff.is_equal == 0].qid].to_csv('error_example')
    print np.sum(dff['is_equal'])
    if len(dff) == 0:
        return 0
    return sum(dff['is_equal']) / len(dff)

def fix_columns(df1, df2):
    missing_cols = set(df1.columns) - set(df2.columns)
    extra_cols = set(df2.columns) - set(df1.columns)
    for c in missing_cols:
        df2[c] = 0
    return df1, df2[df1.columns]
entity_linking()
#df = pd.read_csv('head_10.csv')
#dfg = df.groupby(['qid'])[u'qid', u'topic_words',u'topic_words_names', u'golden_word', u'golden_word_name', u'label']

#print df.loc[:, [u'qid', u'topic_words',u'topic_words_names', u'golden_word', u'golden_word_name', u'label']]