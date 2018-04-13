from __future__ import unicode_literals
import os
import re
import sys
import json
import time
sys.path.append("..")
import pickle
import util
import pandas as pd
from entity_link.features import feature_select
from stanfordcorenlp import StanfordCoreNLP
from nltk.tokenize import word_tokenize
import numpy as np

class EntityDetect(object):
    def __init__(self):
        self.nlp = StanfordCoreNLP("http://10.60.1.82", port=9999, lang="en")
        print("Stanford CoreNLP Server connnected ...")
        self.word_frequence = json.load(open("../datas/zyt/keyword_frequence.json","r",encoding="UTF-8"))
        self.tag_list = ["FW", "NN", "NNP", "NNPS", "NNS"]
        self.tag_NN = ["NN", "NNP", "NNPS", "NNS"]
        self.id2name = {}
        self.name2id = {}
        self.strict = True
        self.keyword_strict = True
        self.proportion_strict = True
        if(self.strict):
            self.keyword_strict = True
            self.proportion_strict = True
        self.keywords = []
    
    def getKeyWords(self, question, strict=True):
        ## 一些专有名词便是直接大写化的
        ## question = question.lower()
        keyword_list = set()
        
        word_tag = self.nlp.pos_tag(question)
        ner_tag = self.nlp.ner(question)
        tag_length = len(word_tag)
        ner_length = len(ner_tag)
        
        ## 从词性标注中添加单个关键词
        for item in word_tag:
            if(item[1] in self.tag_list):
                keyword_list.add(item[0])
                
        ## 从词性标注中添加多个关键词
        for i in range(tag_length):
            if(word_tag[i][1] == "FW"):
                string = ""
                while(i < tag_length and word_tag[i][1] == "FW"):
                    string =  string + word_tag[i][0] + " "
                    i = i + 1
                keyword_list.add(string.rstrip(" "))

            if(i < tag_length and word_tag[i][1] in self.tag_NN):
                string = ""
                while(i < tag_length and word_tag[i][1] in self.tag_NN):
                    string =  string + word_tag[i][0] + " "
                    i = i + 1
                keyword_list.add(string.rstrip(" "))

        ## 从命名实体识别中添加单个关键词
        for item in ner_tag:
            if(item[1] != "O"):
                keyword_list.add(item[0])

        
        ## 从命名实体识别中添加多个关键词
        for i in range(ner_length):
            if(ner_tag[i][1] != "O"):
                tag = ner_tag[i][1] 
                string = ""
                while(i < ner_length and ner_tag[i][1] == tag):
                    string =  string + ner_tag[i][0] + " "
                    i = i + 1
                keyword_list.add(string.rstrip(" "))
        
        print("\n" + "ori_keyword_list: "+ str(keyword_list))
        if(strict and self.keyword_strict):
            tmp_list = keyword_list.copy()
            for item in tmp_list:
                inflag = False
                for word in keyword_list:
                    if(item in word and item != word):
                        inflag = True
                        break
                if(inflag):
                    keyword_list.remove(item)
        print("cur_keyword_list: "+ str(keyword_list) + "\n")
        keyword_list = [x.lower() for x in keyword_list]
        self.keywords = keyword_list
        return keyword_list
    
    def loadData(self, filepath):
        count = 0
        with open(filepath,"r",encoding="UTF-8") as file:
            for line in file:
                count = count + 1
                if(count % 1000000 == 0):
                    print("loaded %d entities ... " % count)
                index = line.find(",")
                ID = line[:index]
                name = line[index+1:-1]
                self.id2name[ID] = name
                if(name in self.name2id):
                    self.name2id[name].append(ID)
                else:
                    self.name2id[name] = [ID]
        file.close()
        print("entity names loaded !")
        return self.id2name, self.name2id
    
    def IsInString(self, name, string):
        name = re.escape(name)
        regex = "(^" + name + "$)|(^" + name + "\W.*)|(.*\W" + name + "\W.*)|(.*\W" + name + "$)"
        pattern = re.compile(regex)
        if(pattern.match(string)):
            return True
        else:
            return False
    
    def ProportionStrict(self, substring, string):
        if(self.proportion_strict == False):
            return True
        word_threshold = 0
        char_threshold = 0
        word_list =  word_tokenize(string)
        sub_word_list = word_tokenize(substring)
        string_chars = sum([len(x) for x in word_list])
        sub_string_chars = sum([len(x) for x in sub_word_list])
        string_words = len(word_list)
        sub_string_words = len(sub_word_list)
        
        return sub_string_words/string_words >= word_threshold and sub_string_chars/string_chars >= char_threshold
        
    
    def DetectEntities(self, keywords, threshold=1000):
        result = {"topic_words":[], "topic_words_names":[]}
        
        ## 根据词频进行排序
        keywords = sorted(keywords, key=lambda x: self.word_frequence[x])
        ## 这里是测试的部分：如果某个词的词频超过了其它词一个数量级的优势，则只对词频最少的词进行搜索
        ## TODO:这里需要处理 keywords为空的情况
        min_frequence = self.word_frequence[keywords[0]]
        keywords = [x for x in keywords if self.word_frequence[x] < min_frequence*10 ]
        
        ## 为了通过词频进行实体筛选，外层循环为关键词
        for word in keywords:
            for key in self.name2id.keys():
                tmp = key.lower()
                if(word in tmp and self.IsInString(word, tmp) and self.ProportionStrict(word, tmp)):
                    for ID in self.name2id[key]:
                        result['topic_words'].append(ID)
                        result['topic_words_names'].append(key)
                        if(len(result['topic_words']) >= threshold):
                            return result
        return result
    
class QustionAnswering(object):
    def __init__(self):
        self.xgb = pickle.load(open("../datas/models/xgb_all.pickle.dat", "rb"))
        print("Model loaded ...")
        print (self.xgb.booster().get_fscore())
        self.detect = EntityDetect()
        self.keywords = set()
        self.word_frequence = self.detect.word_frequence
        
#         self.total_words = sum(self.word_frequence.values())
    
    def load_data(self, filepath):
        return self.detect.loadData(filepath)
    
    def set_data(self, id2name, name2id):
        self.detect.id2name = id2name
        self.detect.name2id = name2id
    
    def longest_common_words(self, left, right):
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
    
    def gen_item_features(self, input):
        data = {'question': [], 'topic_words': [], 'word_score': [], 'topic_words_names': [], "label":[] }
        assert len(input['topic_words']) == len(input['topic_words_names'])
        cand_name = input['topic_words_names']
        question = input['question'][0]
        data['question'] = question
        for i, cand in enumerate(input['topic_words']):
            data['topic_words'].append(cand)
            data['topic_words_names'].append(cand_name[i])
            data['label'].append(1)
            tmpscore = 0
            #print (cand_name[i])
            for word in self.keywords:
                if(word in cand_name[i].lower()):
                    #print (cand_name[i])
                    #print (word_tokenize(question.lower()))
                    pq = self.longest_common_words(word_tokenize(question.lower()), word_tokenize(cand_name[i].lower()))/ len(word_tokenize(cand_name[i].lower()))
                    tmpscore = tmpscore + (pq/(self.word_frequence[word]))**2
            #print (tmpscore)
            #tmpscore = tmpscore if(tmpscore <= 1) else 1                    
            data['word_score'].append(tmpscore)
            
        df = pd.DataFrame(data)
        features, _ = feature_select(df)
        return features
    
    def get_cand_entities(self,question):
        time1 = time.time()
        self.keywords = self.detect.getKeyWords(question)
        time2 = time.time()
        print("get keywords time used: " + str(time2-time1) + "s\n")
        
        time1 = time.time()
        input = self.detect.DetectEntities(self.keywords)
        time2 = time.time()
        print("candidates number: " + str(len(input['topic_words'])))
        print("get all candidates time used: " + str(time2-time1) + "s\n")
        
        input['question'] = question
        input = pd.DataFrame(input)
        print("------ top 5 candidates below ------")
        print(input['topic_words_names'].tolist())
        print("------ top 5 candidates above -------\n")
        return input
    
    def get_top_entities(self, input):
        time1 = time.time()
        features = self.gen_item_features(input)
        predict = self.xgb.predict_proba(features)
        input['predict'] = predict[:, 1]
        head_20 = input.sort_values(['predict'], ascending=False).head(20)
        time2 = time.time()
        print("get top entities time used: " + str(time2-time1) + "s\n")
        
        return head_20
    
    def get_results(self, question):
        input = self.get_cand_entities(question)
        input = input.reset_index(drop=True)
        return self.get_top_entities(input)  
    
    def ret_keywords(self):
        return self.keywords
    
