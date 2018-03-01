from __future__ import unicode_literals, division
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import pandas as pd
from utils.freebase_wiki import load_pkl_file
from embedding.basic import Dictionary, FbDictionary
import numpy as np
import codecs
from mysql import MySQL
import random


class DataReader(object):
    def __init__(self, mid_name_file=None,
                 mid_qid_file=None,
                 topic_words_file=None,
                 sq_data_file=None):
        self.sq_dataset = pd.DataFrame()
        self.subject_ids = {}
        self.relations = {}
        self.object_ids = {}
        self.questions = {}
        self.subject_names = {}
        self.object_names = {}
        self.fb_dict = FbDictionary()
        self.fb_entities = {}
        self.fb_relations = set()
        self.word_dict = Dictionary()
        self.mid_name_file = "../datas/mid2name.tsv" if mid_name_file is None else mid_name_file
        self.mid_qid_file = "../datas/fb2w.nt" if mid_qid_file is None else mid_qid_file
        self.topic_words_file = topic_words_file
        self.sq_data_file = sq_data_file
        self.mid_name_dict, self.name_mid_dict = self.mid_name_convert(mid_name_file)
        self.db_conn = MySQL(ip='10.61.2.166', port=3306, user='zengyutao',
                                  pw='zengyutao', db_name='wikidata')

    def read_sq_data(self):
        cur_idx = 0
        datas = open(self.sq_data_file, 'r')

        for line in datas.readlines():
            line = line.split('\t')
            assert len(line) == 4, "invalid input format" + str(cur_idx)
            sub_id, rel, obj_id, question = line[0:4]
            self.subject_ids[cur_idx] = sub_id.replace("www.freebase.com","")
            self.relations[cur_idx] = rel
            self.object_ids[cur_idx] = obj_id.replace("www.freebase.com", "")
            self.questions[cur_idx] = question
            # mid_name_dict = self.mid_name_convert('../datas/mid2name.tsv')[0]
            self.subject_names[cur_idx] = self.get_mid_to_name_mysql(sub_id.replace("www.freebase.com", ""))
            self.object_names[cur_idx] = self.get_mid_to_name_mysql(obj_id.replace("www.freebase.com", ""))
            cur_idx += 1

        self.sq_dataset['subject_ids'] = self.subject_ids.values()
        self.sq_dataset['relations'] = self.relations.values()
        self.sq_dataset['object_ids'] = self.object_ids.values()
        self.sq_dataset['questions'] = self.questions.values()
        self.sq_dataset['subject_names'] = self.subject_names.values()
        self.sq_dataset['object_names'] = self.object_names.values()
        self.sq_dataset['qid'] = self.sq_dataset.index
        datas.close()
        print ("read sq_data done!")

    def read_sq_data_pd(self):
        print ("read data start")
        datas = pd.read_csv(self.sq_data_file, header=None, sep='\t', skip_blank_lines=False)
        datas.columns = ['subject_id', 'relation', 'object_id', 'question']
        for c in ['subject_id', 'relation', 'object_id']:
            datas[c] = datas[c].apply(lambda x: x.replace("www.freebase.com", ""))
        datas['subject_name'] = datas['subject_id'].apply(lambda x: self.get_mid_to_name_mysql(x))
        datas['object_name'] = datas['object_id'].apply(lambda x: self.get_mid_to_name_mysql(x))
        self.sq_dataset = datas
        self.sq_dataset['qid'] = self.sq_dataset.index
        #print self.sq_dataset[len(self.sq_dataset) - 10:]
        print ("load sq data with df done!")

    @staticmethod
    def mid_name_convert(mid_name_file):
        mid_name = pd.read_csv(mid_name_file, sep='\t', header=None)
        mid_name.columns = ['mid', 'name']
        # print (mid_name[0:10])
        mid_name_dict = dict(zip(mid_name.mid, mid_name.name))
        # print ({k: mid_name_dict[k] for k in list(mid_name_dict)[:20]})
        name_mid_dict = dict(zip(mid_name.name, mid_name.mid))
        return mid_name_dict, name_mid_dict

    @staticmethod
    def get_mid_to_name(mid, mid_name_dict):
        # (mid_name_dict, _) = self.mid_name_convert(self.mid_name_file)
        if mid in mid_name_dict:
            return mid_name_dict[mid]
        return None

    def get_mid_to_name_mysql(self, mid):
        table_name = 'mid2name'
        query = "select name from %s where mid = '%s' " % (table_name, mid)
        #print query
        name = self.db_conn.search(query)
        if name is not None and len(name) >= 1:
            return name[0]
        return None

    def get_mid_to_type_mysql(self, mid):
        table_name = 'mid2type'
        query = "select notable_type from %s where mid = '%s' " % (table_name, mid)
        #print query
        name = self.db_conn.search(query)
        if name is not None and len(name) >= 1:
            return name[0]
        return None

    @staticmethod
    def get_name_to_mid(name, name_mid_dict):
        # (_, name_mid_dict) = self.mid_name_convert(self.mid_name_file)
        if name in name_mid_dict:
            return name_mid_dict[name]
        return None

    @staticmethod
    def mid_qid_convert(mid_qid_file):
        mid_qid = pd.read_csv(mid_qid_file, sep='\t', header=None)
        mid_qid.columns=['fb', 'rel', 'wiki']
        print (mid_qid[0:10])
        mid_qid['fb'] = mid_qid['fb'].apply(
            lambda x: x.replace("<http://rdf.freebase.com/ns", "").replace(">","").replace(".", "/"))
        mid_qid['wiki'] = mid_qid['wiki'].apply(
            lambda x: x.replace("<", "").replace("> .", ""))
        mid_qid_dict = dict(zip(mid_qid.fb, mid_qid.wiki))
        qid_mid_dict = dict(zip(mid_qid.wiki, mid_qid.fb))
        # print ({k: qid_mid_dict[k] for k in list(qid_mid_dict)[:20]})
        return mid_qid_dict, qid_mid_dict

    def load_fb(self, fb_file, rp_dict):
        fb = pd.read_csv(fb_file, sep='\t', header=None)
        fb.columns = ['sub', 'rel', 'obj']
        for c in fb.columns:
            fb[c] = fb[c].apply(lambda x: x.replace("www.freebase.com", ""))

        # load entities
        for sub in fb['sub']:
            sub_name = self.get_mid_to_name(sub, self.mid_name_dict)
            if sub not in self.fb_entities:
                self.fb_entities[sub] = sub_name
            if sub in rp_dict:
                vocab = rp_dict[sub]
            else:
                vocab = np.random.rand(1, 50)
            if sub in self.fb_dict:
                continue
            self.fb_dict[('ent', sub)] = vocab

        for obj in fb['obj']:
            obj_name = self.get_mid_to_name(obj, self.mid_name_dict)
            if obj not in self.fb_entities:
                self.fb_entities[obj] = obj_name
            if obj_name in rp_dict:
                vocab = rp_dict[obj_name]
            else:
                vocab = np.random.rand(1, 50)
            if obj in self.fb_dict:
                continue
            self.fb_dict[('ent', obj)] = vocab

        # load relation
        for rel in fb['rel']:
            self.fb_relations.add(rel)
            if rel in rp_dict:
                vocab = rp_dict[rel]
            else:
                vocab = np.random.rand(1, 50)
            if rel in self.fb_dict:
                continue
            self.fb_dict[('rel', rel)] = vocab

    # load topic words
    def load_topic_words(self, stage, sq_datas):
        with codecs.open(self.topic_words_file, 'r', encoding='utf-8', errors='ignore') as tw_data:
            data = {"label": [], "word_list": [],
                    "label_name": [], "word_name_list": [],
                    "word_score": []}
            index = 0
            if stage == 'train':
                for line in tw_data.readlines():
                    line = line.strip().split('\t')
                    assert len(line) >= 1, "should be no less than one item " + str(index)
                    label = line[0].replace('m.', '/m/')
                    label_name = self.get_mid_to_name_mysql(label)
                    word_list = list()
                    word_name_list = list()
                    score_list = list()
                    word_score_dict = dict()
                    for i in range(1, len(line)):
                        word_score = line[i].strip().split(' ')
                        assert len(word_score) == 2, "item should be contain 2 parts " + str(index)
                        word_list.append(word_score[0].replace('m.', '/m/'))
                        word_score_dict[word_score[0].replace('m.', '/m/')] = float(word_score[1])

                    if len(word_list) <= 10:
                        if label not in word_list:
                            word_list.append(label)
                            word_score_dict[label] = 2.0
                        data["word_list"].append(word_list)
                        for i in range(len(word_list)):
                            word_name_list.append(self.get_mid_to_name_mysql(word_list[i]))
                            score_list.append(word_score_dict[word_list[i]])
                        data["word_name_list"].append(word_name_list)
                        data['word_score'].append(score_list)
                    else:
                        word_list_sample = [word_list[i] for i in sorted(random.sample(xrange(len(word_list)), 10))]
                        if label not in word_list_sample:
                            word_list_sample.append(label)
                            word_score_dict[label] = 2.0
                        data["word_list"].append(word_list_sample)
                        for i in range(len(word_list_sample)):
                            word_name_list.append(self.get_mid_to_name_mysql(word_list_sample[i]))
                            score_list.append(word_score_dict[word_list_sample[i]])
                        data["word_name_list"].append(word_name_list)
                        data['word_score'].append(score_list)
                    data["label"].append(label)
                    data["label_name"].append(label_name)
                    index += 1
            elif stage == "valid" or stage == "test":
                for line in tw_data.readlines():
                    line = line.strip().split('\t')
                    assert len(line) >= 1, "should be no less than one item " + str(index)
                    label = line[0].replace('m.', '/m/')
                    label_name = self.get_mid_to_name_mysql(label)
                    word_list = list()
                    word_name_list = list()
                    score_list = list()
                    word_score_dict = dict()

                    for i in range(1, len(line)):
                        word_score = line[i].strip().split(' ')
                        assert len(word_score) == 2, "item should be contain 2 parts " + str(index)
                        word_list.append(word_score[0].replace('m.', '/m/'))
                        score_list.append(float(word_score[1]))
                        word_name_list.append(self.get_mid_to_name_mysql(word_score[0].replace('m.', '/m/')))
                    if len(score_list) == 0:
                        word_list.append(label)
                        score_list.append(2.0)
                        word_name_list.append(label_name)
                    data["word_list"].append(word_list)
                    data["word_name_list"].append(word_name_list)
                    data['word_score'].append(score_list)
                    data["label"].append(label)
                    data["label_name"].append(label_name)
                    index += 1
        print index, len(data['label']), len(sq_datas)
        assert len(data['label']) == len(self.sq_dataset), "two dataset must have same lines"
        sq_datas['topic_words'] = data.get('word_list')
        sq_datas['topic_words_names'] = data.get('word_name_list')
        sq_datas['golden_word'] = data.get('label')
        sq_datas['golden_word_name'] = data.get('label_name')
        sq_datas['word_score'] = data.get('word_score')
        sq_datas['word_score'] = sq_datas['word_score'].fillna(value=0.0)
        self.db_conn.connect.close()
        print (type(sq_datas.loc[50, 'word_score']))
        print (sq_datas.loc[50, 'word_score'])
        print ("read topic words done!")
        return sq_datas




