from type_rnn import *
from data_handle import *
import sys


rnn = torch.load('char-typernn-classification.pt')

print (rnn.word_emb.weight.data)
print (rnn.char_emb.weight.data)
rnn.training = False
print (rnn)


def predict(category_tensor, qw_tensor, qc_tensor):
    #hidden = rnn.init_hidden()
    #optimizer.zero_grad()
    # print (type(q_tensor[0]))

    output = rnn(qw_tensor, qc_tensor)
    #print (output, hidden)
    #loss = criterion(output, category_tensor)
    #loss.backward()
    #optimizer.step()

    return output