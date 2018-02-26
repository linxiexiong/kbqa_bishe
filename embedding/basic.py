from __future__ import unicode_literals, print_function, division
import torch
from torch.autograd import Variable
import numpy as np
import unicodedata


class Embedder(object):
    def __init__(self, in_dim=None, out_dim=None, normalize=False, trainfrac=1., **kw):
        super(Embedder, self).__init__(**kw)
        assert(in_dim is not None and out_dim is not None)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.normalize = normalize
        self.trainfrac = trainfrac

    def apply(self, idxs):
        raise NotImplementedError("use subclass")


class IdxToOneHot(Embedder):
    def __init__(self, vocsize, **kw):
        super(IdxToOneHot,self).__init__(vocsize, vocsize, **kw)
        self.W = Variable(np.eye(vocsize, vocsize))

    def apply(self, inp):
        return self.W[inp, :]


class Dictionary(object):
    NULL = '<NULL>'
    UNK = '<UNK>'
    START = 2

    @staticmethod
    def normalize(token):
        return unicodedata.normalize('NFD', token)

    def __init__(self):
        self.tok2ind = {self.NULL:0, self.UNK:1}
        self.ind2tok = {0:self.NULL, 1:self.UNK}

    def __len__(self):
        return len(self.tok2ind)

    def __iter__(self):
        return iter(self.tok2ind)

    def __contains__(self, key):
        if type(key) == int:
            return key in self.ind2tok
        elif type(key) == str:
            return self.normalize(key)

    def __getitem__(self, key):
        if type(key) == int:
            return self.ind2tok.get(key)
        if type(key) == str:
            return self.tok2ind.get(self.normalize(key),
                                    self.tok2ind.get(self.UNK))

    def __setitem__(self, key, item):
        if type(key) == int and type(item) == str:
            self.ind2tok[key] = item
        elif type(key) == str and type(item) == int:
            self.tok2ind[key] = item
        else:
            raise RuntimeError('Invalid (key, item) types.')

    def add(self, token):
        token = self.normalize(token)
        if token not in self.tok2ind:
            index = len(self.tok2ind)
            self.tok2ind[token] = index
            self.ind2tok[index] = token

    def tokens(self):
        """Get dictionary tokens.
        Return all the words indexed by this dictionary, except for special
        tokens.
        """
        tokens = [k for k in self.tok2ind.keys()
                  if k not in {'<NULL>', '<UNK>'}]
        return tokens


class FbDictionary(object):

    def __init__(self):
        self.entity_dict = {}
        self.relation_dict = {}

    def __contains__(self, key):
        return (key in self.entity_dict) or (key in self.relation_dict)

    def __getitem__(self, key):
        if key in self.entity_dict:
            return self.entity_dict.get(key)
        elif key in self.relation_dict:
            return self.relation_dict.get(key)
        return None

    def __setitem__(self, key, value):
        if len(key) == 2:
            if key[0] == 'ent':
                self.entity_dict[key[1]] = value
            elif key[0] == 'rel':
                self.relation_dict[key[1]] = value
            else:
                raise RuntimeError('Invalid key types.')
        else:
            raise RuntimeError('Invalid (key, item) types.')
