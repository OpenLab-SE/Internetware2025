import json

from typing import Iterator, List, Dict
from overrides import overrides

from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField, LabelField, MultiLabelField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, SpacyTokenizer
from allennlp.data.vocabulary import Vocabulary


class AllennlpIssueDatasetReader(DatasetReader):
    def __init__(self, tokenizer: Tokenizer = None, token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self.tokenizer = tokenizer or SpacyTokenizer()
        self.token_indexers = token_indexers or {
            "tokens": SingleIdTokenIndexer(token_min_padding_length=8, lowercase_tokens=True)
        }

    def text_to_instance(self, text: str, labels: str = None) -> Instance:
        tokenized_text = self.tokenizer.tokenize(text)
        text_field = TextField(tokenized_text, self.token_indexers)
        fields = {'text': text_field}
        if labels is not None:
            labels_field = MultiLabelField(labels=labels)
            fields['labels'] = labels_field
        return Instance(fields)

    def read(self, file_path: str) -> Iterator[Instance]:
        dataset = []
        with open(file_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        for each_data in dataset:
            yield self.text_to_instance(each_data['title'] + ' ' + each_data['description'], each_data['labels'])
