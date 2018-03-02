from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
from type_rnn import TypeRNN
from data_handle import *
import time
import math


n_hidden = 128
n_epochs = 5000
print_every = 50
plot_every = 100
learning_rate = 0.005
n_letters, categories, n_categories = get_len('type_train.csv')
print (n_letters, categories, n_categories)

def category_from_output(output):
    print (output)
    top_n, top_i = output.data.topk(1)
    print(top_i)
    category_i = top_i[0][0]
    # print (categories[category_i])
    return categories[category_i], category_i

rnn = TypeRNN(n_letters, n_hidden, n_categories)
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()


def train(category_tensor, q_tensor):
    hidden = rnn.init_hidden()
    optimizer.zero_grad()
    # print (type(q_tensor[0]))
    for i in range(q_tensor.size()[0]):
        output, hidden = rnn(q_tensor[i], hidden)
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

start = time.time()

for epoch in range(1, n_epochs + 1):
    questions, cate, q_tensor, cate_tensor = random_train_pair('type_train.csv')
    output, loss = train(cate_tensor, q_tensor)
    current_loss += loss

    if epoch % print_every == 0:
        guess, guess_i = category_from_output(output)
        correct = 'correct' if guess == cate else 'error (%s)' % cate
        print('%d %d%% (%s) %.4f  / %s %s' % (epoch, epoch / n_epochs * 100, time_since(start), loss, guess, correct))

    if epoch % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

torch.save(rnn, 'char-typernn-classification.pt')