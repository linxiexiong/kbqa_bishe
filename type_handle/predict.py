from type_rnn import *
from data_handle import *
import sys


rnn = torch.load('char-typernn-classification.pt')


def evaluate(line_tensor):
    hidden = rnn.init_hidden()
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output


def predict(line, type, n_predictions=5,):
    output = evaluate(Variable(line_to_tensor((line))))
    topv, topi = output.data.topk(n_predictions, 1, True)
    is_type_in = 0
    for i in range(n_predictions):
        category_index = topi[0][i]
        if category_index == type:
            is_type_in = 1
    return is_type_in
