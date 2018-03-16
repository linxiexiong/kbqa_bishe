# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
import sys
sys.path.append("..")
sys.path.append("../..")
import torch
import torch.nn as nn
from type_rnn import TypeRNN
from data_handle import *
import time
import math
import argparse
from embedding.vocab import *


n_letters, categories, n_categories = get_len('type_train.csv')
print (n_letters, categories, n_categories)

train_data = data_handle('type_train.csv')
word_dict = buil_word_dict_simple(restrict_vocab=True,
                                embedding_file='../datas/glove.6B.50d.txt',
                                examples=train_data['question'])
embedding_file = '../datas/glove.6B.50d.txt'
words = index_embedding_words(embedding_file)
print (len(word_dict))
print (len(words))
#for i in range(len(word_dict)):
    #print (word_dict[i])


args = argparse.Namespace()
args.vocab_size = len(word_dict)
args.emb_dim = 50
args.padding_idx = 0
args.char_dim = 128
args.char_hidden = 128
args.char_vocab_size = 128
args.batch_size = 32
args.word_embedding = torch.randn(args.vocab_size, args.emb_dim)
args.char_embedding = torch.randn(args.char_vocab_size, args.char_dim)
args.sent_hidden = 256
args.num_label = n_categories
args.dropout = 0.8
args.num_layer = 3

rnn = TypeRNN(args)
load_pretrain_embedding(words, word_dict, embedding_file, args.emb_dim)
print (rnn.word_emb.weight.data)

#n_hidden = 128
n_epochs = 5
print_every = 1
plot_every = 100
learning_rate = 0.005


def category_from_output(output):
    #print (output)
    top_n, top_i = output.data.topk(1)
    #print(top_i)
    category_i = [top_i[i][0] for i in range(args.batch_size)]
    categorys = [categories[c_i] for c_i in category_i]
    # print (categories[category_i])
    return categorys, category_i

#rnn = TypeRNN(n_letters, n_hidden, n_categories)
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()


def train(category_tensor, qw_tensor, qc_tensor):
    #hidden = rnn.init_hidden()
    optimizer.zero_grad()
    # print (type(q_tensor[0]))

    output = rnn(qw_tensor, qc_tensor)
    #print (output, hidden)
    loss = criterion(output, category_tensor)
    loss.backward()
    optimizer.step()

    return output, loss.data[0]

current_loss = 0
all_losses = []


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def get_accuracy(guess_label, labels):
    assert (len(guess_label) == len(labels))
    correct = 0.0
    for i in range(len(labels)):
        if guess_label[i] == labels[i]:
            #print (guess_label[i], labels[i])
            correct += 1
    rate = correct / len(labels)
    return rate

start = time.time()
word_dict = buil_word_dict_simple(restrict_vocab=True,
                            embedding_file='../datas/glove.6B.50d.txt',
                            examples=train_data['question'])
char_dict = build_char_dict()
for epoch in range(1, n_epochs + 1):
    train_data = train_data.sample(frac=1).reset_index(drop=True)
    # questions, cate, q_tensor, cate_tensor = random_train_pair(train_data)
    length = len(train_data)
    accuracy = 0
    #print (train_data[0: 32])
    for i in range(0, int(length / args.batch_size)):
        start = i * args.batch_size
        end = (i + 1) * args.batch_size
        #print (start, end)
        if end > length - 1:
            end = length - 1
        batch_train_data = train_data[start: end]
        questions, labels, label_tensor, qw_tensor, qc_tensor = get_batch_datas(batch_train_data,
                                                                                word_dict,
                                                                                char_dict)

        # print (label_tensor)
        output, loss = train(label_tensor, qw_tensor, qc_tensor)
        #print (output)
        current_loss += loss
        guess_label, _ = category_from_output(output)
        accuracy += get_accuracy(guess_label, labels)
        #print ('%d %d %.4f' % (epoch, i,  accuracy))
    print ('%d %.4f %.4f' % (epoch, current_loss, accuracy))

        # if epoch % print_every == 0:
        #     guess, guess_i = category_from_output(output)
        #     correct = 'correct' if guess == cate else 'error (%s)' % cate
        #     print('%d %d%% (%s) %.4f  / %s %s' % (epoch, epoch / n_epochs * 100, time_since(start), loss, guess, correct))
        #
        # if epoch % plot_every == 0:
        #     all_losses.append(current_loss / plot_every)
        #     current_loss = 0
rnn.eval()
test_data = data_handle('type_test.csv')
length = len(test_data)
for i in range(0, int(length / args.batch_size)):
    start = i * args.batch_size
    end = (i + 1) * args.batch_size
    #print (start, end)
    if end > length - 1:
        end = length - 1
    test_data_batch = test_data[start: end]
    tq, tl, test_label_tensor, test_qw_tensor, test_qc_tensor = get_batch_datas(test_data_batch,
                                                                                word_dict,
                                                                                char_dict)
    #print (test_qw_tensor, test_qc_tensor)
    #print (test_qw_tensor)
    predict = rnn(test_qw_tensor, test_qc_tensor)
    print (predict)
param = {
    'network': rnn,
    'word_dict': word_dict,
    'char_dict': char_dict
}
torch.save(param, 'char-typernn-classification.pt')