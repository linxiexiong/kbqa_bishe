from .basic import Dictionary
import codecs

def index_embedding_words(embedding_file):
    words = set()
    with codecs.open(embedding_file, encoding='utf8') as f:
        for line in f:
            #print (line.rstrip().split(' ')[0])
            w = Dictionary.normalize(line.rstrip().split(' ')[0])
            words.add(w)
    return words


def load_words(restrict_vocab, embedding_file, example):
    def _insert(iterable):
        print (iterable)
        for w in iterable:
            print(w)
            w = Dictionary.normalize(w.decode('utf-8'))
            if valid_words and w not in valid_words:
                continue
            words.add(w)

    if restrict_vocab and embedding_file:
        valid_words = index_embedding_words(embedding_file)
    else:
        valid_words = None
    words = set()
    for ex in example:
        _insert(ex)
    return words


def build_word_dict(restrict_vocab, embedding_file, examples):
    word_dict = Dictionary()
    for w in load_words(restrict_vocab, embedding_file, examples):
        word_dict.add(w)
    return word_dict


