from __future__ import unicode_literals
import os
import re
import sys
sys.path.append("..")
import pickle
import util
import pandas as pd
from entity_link.features import feature_select
from stanfordcorenlp import StanfordCoreNLP
from nltk.tokenize import word_tokenize

class EntityDetect(object):
    def __init__(self):
        self.nlp = StanfordCoreNLP("http://10.60.1.82", port=9999, lang="en")
        print("Stanford CoreNLP Server connnected ...")
        self.tag_list = ["FW", "NN", "NNP", "NNPS", "NNS"]
        self.tag_NN = ["NN", "NNP", "NNPS", "NNS"]
        self.keyword_strict = True      
    
    def getKeyWords(self, question):
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
                
        if(self.keyword_strict):
            tmp_list = keyword_list.copy()
            for item in tmp_list:
                inflag = False
                for word in keyword_list:
                    if(item in word and item != word):
                        inflag = True
                        break
                if(inflag):
                    keyword_list.remove(item)
    
        return keyword_list

nlp = EntityDetect()
count = 0 
wordsBag = []
newfile = open("./WordOfBag.txt","w",encoding="UTF-8")
with open("./questions_all.txt","r", encoding="UTF-8") as file:
    for line in file:
        count = count + 1
        if(count % 100 == 0):
            print("line-%d ..." % count)
        question = line[1:-2]
        keywords = nlp.getKeyWords(question)
        for word in keywords:
            wordsBag.append(word)
print("Finished!")
for item in wordsBag:
    newfile.write(item+"\n")
print("File created!")