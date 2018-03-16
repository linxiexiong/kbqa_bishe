# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
import sys
sys.path.append("..")
sys.path.append("../..")
import unicodedata
import string
import pandas as pd
import torch
# if torch.cuda.is_available():
#     import torch.cuda as torch
# else:
import torch as torch
from torch.autograd import Variable
import random
import numpy as np
from nltk.tokenize import word_tokenize
from embedding.vocab import build_word_dict, build_char_dict


def unicodeToAscii(s):
    all_letters = string.ascii_letters + ".,;' "
    n_letters = len(all_letters)
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn'
                   and c in all_letters)


def letter_to_index(letter, all_letters):
    return all_letters.find(letter)


def letter_to_tensor(letter, all_letters):
    tensor = torch.zeros(1, len(all_letters))
    tensor[0][letter_to_index(letter, all_letters)] = 1
    return tensor


def line_to_tensor(line, all_letters):
    tensor = torch.zeros(len(line), 1, len(all_letters))
    for li, letter in enumerate(line):
        # letter = unicodeToAscii(letter)
        tensor[li][0][letter_to_index(letter, all_letters)] = 1
    return tensor


def line_to_word_tensor(line, word_dict):
    #print ([(line.split(' ')) for line in lines])
    #max_len = max([len(line.split(' ')) for line in lines ])
    if line is None:
        return Variable(torch.LongTensor(1, 1).zero_())
    line = line.lower()
    max_len = len(word_tokenize(line))
    #print (max_len)
    #tensor = torch.zeros(len(lines), max_len)
    tensor = torch.LongTensor(1, max_len).zero_()
    line = word_tokenize(line)
        #print(line)
    for i, word in enumerate(line):
            #print (word)
        if word not in word_dict:
            tensor[0][i] = 0
            continue
        tensor[0][i] = word_dict[word]
    return Variable(tensor)

def line_to_char_tensor(line, char_dict):
    #max_line_len = max([len(line.split(' ')) for line in lines ])
    if line is None:
        return Variable(torch.LongTensor(1, 1, 1).zero_())
    line = line.lower()
    max_line_len = len(word_tokenize(line))
    max_word_len = 0

    for word in word_tokenize(line):
        if len(word) > max_word_len:
            max_word_len = len(word)
    tensor = torch.LongTensor(1, max_line_len, max_word_len).zero_()

    line = word_tokenize(line)
    for wi, word in enumerate(line):
        for ci, c in enumerate(word):
            if c not in char_dict:
                tensor[0][wi][ci] = 0
                continue
            tensor[0][wi][ci] = char_dict[c]
    return Variable(tensor)


def lines_to_word_tensor(lines, word_dict):
    #print ([(line.split(' ')) for line in lines])
    #max_len = max([len(line.split(' ')) for line in lines ])
    max_len = 0
    for line in lines:
        #line = line.decode('utf8')
        #print (line)
        if line is None:
            continue
        if len(word_tokenize(line)) > max_len:
            max_len = len(word_tokenize(line))
    #print (max_len)
    #tensor = torch.zeros(len(lines), max_len)
    tensor = torch.LongTensor(len(lines), max_len).zero_()
    for li, line in enumerate(lines):
        #print (line)
        #line = line.decode('utf8')
        if line is None:
            continue
        line = line.lower()
        line = word_tokenize(line)
        #print(line)
        for i, word in enumerate(line):
            #print (word)
            if word not in word_dict:
                tensor[li][i] = 0
                continue
            tensor[li][i] = word_dict[word]
    return Variable(tensor)


def lines_to_char_tensor(lines, char_dict):
    #max_line_len = max([len(line.split(' ')) for line in lines ])
    max_line_len = 0
    for line in lines:
        #line = line.decode('utf8')
        if line is None:
            continue
        if len(word_tokenize(line)) > max_line_len:
            max_line_len = len(word_tokenize(line))
    max_word_len = 0
    for line in lines:
        #line = line.decode('utf8')
        if line is None:
            continue
        line = line.lower()
        for word in word_tokenize(line):
            if len(word) > max_word_len:
                max_word_len = len(word)
    tensor = torch.LongTensor(len(lines), max_line_len, max_word_len).zero_()
    for li, line in enumerate(lines):
        #line = line.decode('utf8')
        if line is None:
            continue
        line = line.lower()
        line = word_tokenize(line)
        for wi, word in enumerate(line):
            for ci, c in enumerate(word):
                if c not in char_dict:
                    tensor[li][wi][ci] = 0
                    continue
                tensor[li][wi][ci] = char_dict[c]
    return Variable(tensor)


def type_to_index(types):
    type_idx = dict()
    for idx, type in enumerate(types):
        type_idx[type] = idx
    return type_idx


def get_index(type_idx, type):
    return type_idx[type]


def random_choice(l):
    return l[random.randint(0, len(l)-1)]


def data_handle(file_name):
    all_letters = string.ascii_letters + ".,;' "
    train_data = pd.read_csv(file_name)
    train_data = train_data[['question', 'type', 'type_name']].fillna(value='a')
    categories = set(train_data.type)
    type_idx = type_to_index(categories)
    #print (type_idx.keys()[177])
    train_data['cate'] = train_data['type'].apply(lambda x: get_index(type_idx, x))

    #train_data['cate_tensor'] = train_data['cate'].apply(lambda x: Variable(torch.LongTensor([x])))
    #train_data['qw_tensor'] = train_data['question'].apply(lambda x: Variable())
    #train_data['q_tensor'] = train_data['question'].apply(lambda x: Variable(line_to_tensor(x, all_letters)))
    #print (train_data.loc[0, 'question'])
    #print (line_to_tensor(train_data.loc[0, 'question'], all_letters))
    #print (len(train_data[train_data.cate == 178]))
    #train_data['question'] = train_data['question'].apply(lambda x: unicodeToAscii(x))
    #print (type(train_data.groupby('cate', as_index=False).agg(np.size)))
    return train_data


def get_len(file_name):
    all_letters = string.ascii_letters + ".,;' "
    #train_data = pd.read_csv(file_name)
    ##train_data = train_data[['question', 'type', 'type_name']].fillna(value='')
    train_data = data_handle(file_name)
    categories = list(set(train_data.cate))

    return len(all_letters), categories, len(categories)


def random_train_pair(train_data):
    #train_data = data_handle(df)
    rand = random.randint(0, len(train_data)-1)
    return (train_data.loc[rand, 'question'],
            train_data.loc[rand, 'cate'],
            train_data.loc[rand, 'q_tensor'],
            train_data.loc[rand, 'cate_tensor'])


def get_batch_datas(train_data, word_dict, char_dict):
    #print (train_data)
    #print (len(train_data))
    train_data.fillna(value='none')
    questions = train_data['question'].tolist()
    labels = train_data['cate'].tolist()
    label_tensor = torch.LongTensor(len(labels)).zero_()
    for i, label in enumerate(labels):
        #print (label)
        label_tensor[i] = label
    #label_tensor = train_data['cate_tensor']
    label_tensor = Variable(label_tensor)
    qw_tensor = lines_to_word_tensor(questions, word_dict)
    qc_tensor = lines_to_char_tensor(questions, char_dict)

    return questions, labels, label_tensor, qw_tensor, qc_tensor


#data_handle('type_test.csv')
