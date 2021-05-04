import pickle
import copy
from collections.abc import Iterable


class LabelEncoder:

    def __init__(self, vocab2id):
        self._vocab2id = vocab2id

    @property
    def vocab_size(self):
        return len(self._vocab2id)

    def __call__(self, corpus):
        if isinstance(corpus, Iterable):
            return [self._vocab2id[token] for token in corpus]
        elif isinstance(corpus, int):
            return self._vocab2id[corpus]
        else:
            raise ValueError('Incorrect value provided for tokenization.')

    @classmethod
    def from_corpus(cls, corpus):
        unique_tokens = list(set(corpus))
        vocab2id = dict(zip(unique_tokens, range(len(unique_tokens))))
        return cls(vocab2id)

    @classmethod
    def from_file(cls, path):
        with open(path, 'rb') as f:
            vocab2id = pickle.load(f)
        return cls(vocab2id)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self._vocab2id, f)

    def get_vocab2id(self):
        return copy.deepcopy(self._vocab2id)