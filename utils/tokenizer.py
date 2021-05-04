import os
import shutil
import pickle

from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizerFast


class BaseTokenizer:

    def encode(self, text):
        raise NotImplementedError

    @property
    def cls_token_id(self):
        raise NotImplementedError

    @property
    def pad_token_id(self):
        raise NotImplementedError

    @property
    def mask_token_id(self):
        raise NotImplementedError

    @property
    def sep_token_id(self):
        raise NotImplementedError

    @property
    def unk_token_id(self):
        raise NotImplementedError

    @property
    def vocab_size(self):
        raise NotImplementedError


class WordpieceTokenizer(BaseTokenizer):

    def __init__(self, vocab_path, strip_accents, clean_text, lowercase, from_pretrained=False):
        common_params = {'strip_accents': strip_accents, 'clean_text': clean_text, 'lowercase': lowercase}
        if from_pretrained:
            self._tokenizer = BertTokenizerFast.from_pretrained(
                pretrained_model_name_or_path=vocab_path, **common_params
            )
        else:
            self._tokenizer = BertTokenizerFast(
                vocab_file=vocab_path, **common_params
            )

    @classmethod
    def from_corpus(
            cls,
            corpus,
            corpus_save_path,
            tokenizer_save_path,
            tokenizer_name,
            vocab_size,
            min_frequency,
            strip_accents,
            clean_text,
            lowercase
    ):
        with open(corpus_save_path, 'wb') as f:
            f.write('\n'.join(corpus).encode())

        tokenizer = BertWordPieceTokenizer(
            strip_accents=strip_accents,
            clean_text=clean_text,
            lowercase=lowercase,
        )
        tokenizer.train(
            [corpus_save_path],
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            show_progress=True,
            special_tokens=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'],
            wordpieces_prefix="##",
        )

        if os.path.exists(tokenizer_save_path):
            shutil.rmtree(tokenizer_save_path)
        os.mkdir(tokenizer_save_path)

        tokenizer.save_model(tokenizer_save_path, tokenizer_name)
        vocab_path = os.path.join(tokenizer_save_path, f'{tokenizer_name}-vocab.txt')
        return cls(
            vocab_path,
            strip_accents,
            clean_text,
            lowercase
        )

    def encode(self, text, add_cls_token=True):
        ids = []
        if add_cls_token:
            ids.append(self.cls_token_id)
        ids.extend(self._tokenizer.encode(text, add_special_tokens=False))
        return ids

    @property
    def cls_token_id(self):
        return self._tokenizer.cls_token_id

    @property
    def pad_token_id(self):
        return self._tokenizer.pad_token_id

    @property
    def mask_token_id(self):
        return self._tokenizer.mask_token_id

    @property
    def sep_token_id(self):
        return self._tokenizer.sep_token_id

    @property
    def unk_token_id(self):
        return self._tokenizer.unk_token_id

    @property
    def vocab_size(self):
        return self._tokenizer.vocab_size


class CharTokenizer(BaseTokenizer):

    def __init__(self, vocab_path, lowercase):
        with open(vocab_path, 'rb') as f:
            self._vocab2id = pickle.load(f)
        self._lowercase = lowercase

    @classmethod
    def from_corpus(cls, corpus, tokenizer_save_path, lowercase):
        chars = set()
        for text in corpus:
            if lowercase:
                text = text.lower()
            chars.update(text)

        vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'] + list(chars)
        vocab2id = dict(zip(vocab, range(len(vocab))))

        with open(tokenizer_save_path, 'wb') as f:
            pickle.dump(vocab2id, f)

        return cls(tokenizer_save_path, lowercase)

    def encode(self, text, add_cls_token=True):
        if self._lowercase:
            text = text.lower()
        ids = []
        if add_cls_token:
            ids.append(self.cls_token_id)
        ids.extend([self._vocab2id[char] for char in text])
        return ids

    @property
    def cls_token_id(self):
        return 2

    @property
    def pad_token_id(self):
        return 0

    @property
    def mask_token_id(self):
        return 4

    @property
    def sep_token_id(self):
        return 3

    @property
    def unk_token_id(self):
        return 1

    @property
    def vocab_size(self):
        return len(self._vocab2id)

