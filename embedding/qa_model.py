import torch
import torch.nn as nn
import sys
sys.path.append('..')
sys.path.append('../..')
from embedding.question_emb import *
from data_processing.mysql import MySQL
from embedding.vocab import *
import pandas as pd


class EntityModel(nn.Module):
    def __init__(self, args):
        super(EntityModel, self).__init__()
        self.question_emb = QuestionEmb(args)
        self.entity_emb = EntityEmb(args)
        self.fc = nn.Linear((args.hidden_size*2*args.num_layers +
                             args.entity_hidden*2*args.num_layers+args.entity_dim), 1)

    def forward(self, words, chars, entities, entity, mask):
        ent_emb = self.entity_emb(words, chars, entity)
        q_emb = self.question_emb(words, chars, entities, mask)
        print (ent_emb.size())
        print (q_emb.size())
        inputs = torch.cat([q_emb, ent_emb], dim=1)
        score = self.fc(inputs)
        return score


class RelationModel(nn.Module):
    def __init__(self, args):
        super(RelationModel, self).__init__()
        self.question_emb = QuestionEmb(args)
        self.relation_emb = RelationEmb(args)
        self.fc = nn.Linear(args.char_dim+2*(args.embedding_dim+args.entity_dim), 1)

    def forward(self, words, chars, entities, rel_emb, mask):
        q_emb = self.question_emb(words, chars, entities, mask)
        r_emb = self.relation_emb(words, rel_emb)
        inputs = torch.cat([q_emb, r_emb], dim=1)
        score = self.fc(inputs)
        return score


