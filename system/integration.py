from __future__ import unicode_literals
import pandas as pd
import pickle
from entity_link.features import feature_select


def gen_item_features(input):
    data = {'question': [], 'topic_word': [], 'score': [], 'topic_words_names': [], }
    assert len(input['topic_words']) == len(input['topic_words_names'])
    cand_name = input['topic_words_names']
    for i, cand in enumerate(input['topic_words']):
        data['question'].append(input['question'])
        data['topic_words'].append(cand)
        data['topic_words_names'].append(cand_name[i])
        data['score'].append(1)
        data['label'].append(1)
    df = pd.DataFrame(data)
    features, _ = feature_select(df)
    return features


def integrate(input):

    features = gen_item_features(input)
    xgb = pickle.load(open("../datas/models/xgb_100.pickle.dat", "rb"))
    predict = xgb.predict_prob(features)
    input['predict'] = predict[:, 1]
    head_10 = input.sort_values(['predict'], ascending=False).head(10)





integrate("ddd", ['d', 'd'])