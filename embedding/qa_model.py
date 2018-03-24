import torch
import torch.nn as nn
import torch.nn.functional as F
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
        # self.fc = nn.Linear((args.hidden_size*2*args.num_layers +
        #                      args.entity_hidden*2*args.num_layers), 500)
        # self.efc = nn.Linear((args.entity_hidden*2*args.num_layers + args.entity_dim),
        #                      args.entity_hidden*2*args.num_layers)
        # self.fc2 = nn.Linear(500, 256)
        # self.fc3 = nn.Linear(256, 1)

    def forward(self, words, chars, entities, e_words, e_chars, entity, mask, method, predict, types):
        ent_emb = self.entity_emb(e_words, e_chars, entity, method, types)
        q_emb = self.question_emb(words, chars, entities, mask, method, predict)
        #print (ent_emb.size())
        #print (q_emb.size())
        #inputs = torch.cat([q_emb, ent_emb], dim=1)
        #ent_emb = self.efc(ent_emb)
        # fc = self.fc(inputs)
        # fc2 = self.fc2(F.relu(fc))
        # fc3 = self.fc3(F.relu(fc2))
        score = F.cosine_similarity(q_emb, ent_emb)
        #print (score.size())
        return score


class RelationModel(nn.Module):
    def __init__(self, args):
        super(RelationModel, self).__init__()
        self.question_emb = QuestionEmb(args)
        self.relation_emb = RelationEmb(args)
        # self.fc = nn.Linear((args.hidden_size*2*args.num_layers +
        #                      args.relation_hidden*2*args.num_layers), 1)
        # self.rfc = nn.Linear(args.relation_hidden*2*args.num_layers + args.entity_dim,
        #                      args.relation_hidden*2*args.num_layers)

    def forward(self, words, chars, entities, r_words, rel_emb, method, mask, predict):
        q_emb = self.question_emb(words, chars, entities, mask, method, predict)
        r_emb = self.relation_emb(r_words, rel_emb, method)
        #r_emb = self.rfc(r_emb)
        #inputs = torch.cat([q_emb, r_emb], dim=1)
        #fc = self.fc(inputs)
        score = F.cosine_similarity(q_emb, r_emb)
        #score = F.softmax(fc)
        return score


