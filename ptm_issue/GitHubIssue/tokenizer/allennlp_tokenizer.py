import spacy
from typing import Sequence

from allennlp.data.tokenizers import Token


class AllennlpTokenizer(object):
    def __init__(self, vocab, tokenizer, token_indexer):
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.token_indexer = token_indexer

    def __call__(self, text, truncation=False, max_length=None, padding=None):
        # tokenize
        tokens = self.tokenizer.tokenize(text)
        if padding is not None and padding == 'max_length' and len(tokens) < max_length:
            tokens += [Token(self.vocab._padding_token)] * (max_length - len(tokens))

        ids = self.token_indexer.tokens_to_indices(tokens, self.vocab)['tokens']

        if truncation:
            ids = ids[:max_length]

        return {'input_ids': ids}
