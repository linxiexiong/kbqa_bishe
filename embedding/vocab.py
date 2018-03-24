from .basic import Dictionary
import codecs
from nltk.tokenize import word_tokenize
# if torch.cuda.is_available():
#     import torch.cuda as torch
# else:
import torch
from data_processing.mysql import MySQL, get_entity_vector, get_relation_vector

def build_char_dict():
    char_dict = dict()
    for i in range(128):
        char_dict[chr(i)] = i
    return char_dict


def index_embedding_words(embedding_file):
    words = set()
    with codecs.open(embedding_file) as f:
        for line in f:
            #print (line.rstrip().split(' ')[0])
            w = line.rstrip().split(' ')[0]
            #w = Dictionary.normalize(line.rstrip().split(' ')[0])
            words.add(w)
    return words


def load_words(restrict_vocab, embedding_file, example):
    def _insert(iterable):
        #print (iterable)
        for w in iterable:
            #print(w)
            w = w.lower()
            #w = Dictionary.normalize(w)
            if valid_words and w not in valid_words:
                continue
            words.add(w)

    if restrict_vocab and embedding_file:
        valid_words = index_embedding_words(embedding_file)
    else:
        valid_words = None
    words = set()
    for ex in example:
        ex = word_tokenize(ex)
        _insert(ex)
    return words


def build_word_dict(restrict_vocab, embedding_file, examples):
    word_dict = Dictionary()
    for w in load_words(restrict_vocab, embedding_file, examples):
        word_dict.add(w)
    return word_dict


def buil_word_dict_simple(restrict_vocab, embedding_file, examples):
    word_dict = dict()
    index = 0
    for w in load_words(restrict_vocab, embedding_file, examples):
        word_dict[w] = index
        index += 1
    return word_dict


def load_embeddings(words, word_dict, embedding_file, network):
    words = {w for w in words if w in word_dict}
    #print (len(words))
    embedding = network.word_emb.weight.data
    vec_counts = {}
    with open(embedding_file) as f:
        for line in f:
            parsed = line.rstrip().split(' ')
            assert (len(parsed) == embedding.size(1) + 1)
            #w = word_dict.normalize(parsed[0])
            w = parsed[0]
            if w in words:
                vec = torch.Tensor([float(i) for i in parsed[1:]]).cuda()
                if w not in vec_counts:
                    vec_counts[w] = 1
                    embedding[word_dict[w]].copy_(vec)
                else:
                    vec_counts[w] = vec_counts[w] + 1
                    embedding[word_dict[w]].add_(vec)
            for w, c in vec_counts.items():
                embedding[word_dict[w]].div_(c)


def load_pretrain_embedding(words, word_dict, embedding_file, dim):
    #print (words)
    words = {w for w in words if w in word_dict}

    #print (len(words))
    embedding = torch.randn(len(word_dict), dim)
    #print (words)
    with open(embedding_file) as f:
        for line in f:
            parsed = line.rstrip().split(' ')
            w = parsed[0]
            if w in words:
                vec = torch.Tensor([float(i) for i in parsed[1:]])
                embedding[word_dict[w]] = vec
    return torch.Tensor(embedding).cuda()


def build_entity_dict(entities):
    ent_dict = {}
    index = 0
    for ent in entities:
        if ent not in ent_dict:
            ent_dict[ent] = index
            index += 1
    return ent_dict


def load_entity_embedding(entity_dict, dim):
    db_conn = MySQL(ip='10.61.2.166', port=3306, user='zengyutao',
                                       pw='zengyutao', db_name='wikidata')
    embedding = torch.randn(len(entity_dict), dim)

    for e in entity_dict:
        vec = get_entity_vector(db_conn, e)
        vec_s = [float(x) for x in str(vec).split(',')]
        embedding[entity_dict[e]] = torch.Tensor(vec_s)
    return torch.Tensor(embedding).cuda()


def build_relation_dict(relations):
    ent_dict = {}
    index = 0
    for ent in relations:
        if ent not in ent_dict:
            ent_dict[ent] = index
            index += 1
    return ent_dict


def load_relation_embedding(relation_dict, dim):
    db_conn = MySQL(ip='10.61.2.166', port=3306, user='zengyutao',
                                       pw='zengyutao', db_name='wikidata')
    embedding = torch.randn(len(relation_dict), dim)

    for e in relation_dict:
        vec = get_relation_vector(db_conn, e)
        vec_s = [float(x) for x in str(vec).split(',')]
        embedding[relation_dict[e]] = torch.Tensor(vec_s)
    return torch.Tensor(embedding).cuda()
