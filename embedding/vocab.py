from .basic import Dictionary


def index_embedding_words(embedding_file):
    words = set()
    with open(embedding_file) as f:
        for line in f:
            w = Dictionary.normalize(line.rstrip().split(' ')[0])
            words.add(w)
    return words


def load_words(args, example):
    def _insert(iterable):
        for w in iterable:
            w = Dictionary.normalize(w)
            if valid_words and w not in valid_words:
                continue
            words.add(w)

    if args.restrict_vocab and args.embedding_file:
        valid_words = index_embedding_words(args.embedding_file)
    else:
        valid_words = None
    words = set()
    for ex in example:
        _insert(ex['question'])
        _insert(ex['description'])
    return words


def build_word_dict(args, examples):
    word_dict = Dictionary()
    for w in load_words(args, examples):
        word_dict.add(w)
    return word_dict

