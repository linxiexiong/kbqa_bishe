from __future__ import unicode_literals, print_function, division
import unicodedata
import string
import pandas as pd
import torch
from torch.autograd import Variable
import random


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
        letter = unicodeToAscii(letter)
        tensor[li][0][letter_to_index(letter, all_letters)] = 1
        return tensor


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
    train_data['cate'] = train_data['type'].apply(lambda x: get_index(type_idx, x))
    train_data['cate_tensor'] = train_data['cate'].apply(lambda x: Variable(torch.LongTensor([x])))
    train_data['q_tensor'] = train_data['question'].apply(lambda x: Variable(line_to_tensor(x, all_letters)))
    #print (train_data.loc[0, 'question'])
    #print (line_to_tensor(train_data.loc[0, 'question'], all_letters))

    #print (train_data.loc[0, 'question'])
    return train_data


def get_len(file_name):
    all_letters = string.ascii_letters + ".,;' "
    #train_data = pd.read_csv(file_name)
    ##train_data = train_data[['question', 'type', 'type_name']].fillna(value='')
    train_data = data_handle(file_name)
    categories = list(set(train_data.cate))

    return len(all_letters), categories, len(categories)


def random_train_pair(file_name):
    train_data = data_handle(file_name)
    return (train_data.loc[random.randint(0, len(train_data)-1),'question'],
            train_data.loc[random.randint(0, len(train_data)-1),'cate'],
            train_data.loc[random.randint(0, len(train_data)-1),'q_tensor'],
            train_data.loc[random.randint(0, len(train_data)-1),'cate_tensor'])



#data_handle('type_train.csv')