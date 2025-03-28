import argparse
import json
import random
import glob
import tqdm
import os
import torch
import itertools

import numpy as np
import pandas as pd

from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from GitHubIssue.dataset.issue_dataset import IssueDataset
from GitHubIssue.dataset.allennlp_issue_dataset import AllennlpIssueDatasetReader
from GitHubIssue.models.textcnn import TextCNN
from GitHubIssue.models.bilstm import BiLSTM
from GitHubIssue.models.rcnn import RCNN
from GitHubIssue.models.bert import Bert
from GitHubIssue.util.mem import occupy_mem

from GitHubIssue.tokenizer.allennlp_tokenizer import AllennlpTokenizer
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer
from allennlp.data.token_indexers.single_id_token_indexer import SingleIdTokenIndexer
from allennlp.modules.token_embedders.embedding import Embedding


from transformers import BertTokenizer, XLNetTokenizer, AlbertTokenizer, RobertaTokenizer, AutoTokenizer

MODEL_CONFIG = [
    "bert-base-uncased",
    "xlnet-base-cased",
    "albert-base-v2",
    "roberta-base",
    "microsoft/codebert-base",
    "jeniya/BERTOverflow",
    "huggingface/CodeBERTa-language-id",
    "seBERT",
]

TOKENIZER_CONFIG = {
    "bert-base-uncased": BertTokenizer,
    "xlnet-base-cased": XLNetTokenizer,
    "albert-base-v2":  AlbertTokenizer,
    "roberta-base": RobertaTokenizer,
    "microsoft/codebert-base": RobertaTokenizer,
    "jeniya/BERTOverflow": AutoTokenizer,
    "huggingface/CodeBERTa-language-id": RobertaTokenizer,
    "seBERT": BertTokenizer
}


def build_vocab(data, tokenizer):
    """
    对输入数据进行tokenize
    """
    words = set()
    for d in data:
        # 将title和description用空格拼接起来
        tokens = tokenizer.tokenize(d['title'] + ' ' + d['description'])
        for t in tokens:
            words.add(t)
    return list(words)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def train_single(train_data_path: str, test_data_path: str, model_name: str, embedding_type=None, device=0, use_sequence=False, disablefinetune=False, local_model=False, do_predict=False):
    
    if train_data_path == test_data_path:
        with open(train_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        random.seed(42)
        random.shuffle(data)
        random.seed()
        split_1 = int(0.8 * len(data))
        split_2 = int(0.9 * len(data))
        # 训练集:验证集:测试集 = 80%:10%:10%
        train_data = data[:split_1]
        valid_data = data[:split_2]
        test_data = data[split_2:]
    else:
        input_train_data, input_test_data = [], []
        with open(train_data_path, 'r', encoding='utf-8') as f:
            input_train_data = json.load(f)
        with open(test_data_path, 'r', encoding='utf-8') as f:
            input_test_data = json.load(f)

        random.seed(42)
        random.shuffle(input_train_data)
        random.shuffle(input_test_data)
        random.seed()

        split_1 = int(0.8 * len(input_train_data))

        train_data = input_train_data[:split_1]
        valid_data = input_train_data
        test_data = input_test_data

    # 本地模型需要从路径中提取出模型名称
    # model_name like '../model/seBERT/pytorch_model.bin'
    if local_model:
        model_path = model_name # 保存本地模型路径
        model_name = model_name.split('/')[-2]
        print(f"model_name: {model_path}")

    # init tokenizer
    if model_name in ["textcnn", "bilstm", "rcnn"]:
        # build vocab
        allennlp_tokenizer = SpacyTokenizer()
        allennlp_token_indexer = SingleIdTokenIndexer(token_min_padding_length=8, lowercase_tokens=True)
        allennlp_datareader = AllennlpIssueDatasetReader(allennlp_tokenizer, {'tokens': allennlp_token_indexer})
        vocab = Vocabulary.from_instances(allennlp_datareader.read(train_data_path))

        tokenizer = AllennlpTokenizer(vocab, allennlp_tokenizer, allennlp_token_indexer)
    elif model_name in MODEL_CONFIG:
        if not local_model:
            tokenizer = TOKENIZER_CONFIG[model_name].from_pretrained(model_name)
        else:
            # tokenizer_path like '../model/seBERT'
            tokenizer_path = "/".join(model_path.split(r'/')[:-1])
            print(f"tokenizer_path: {tokenizer_path}")
            tokenizer = TOKENIZER_CONFIG[model_name].from_pretrained(tokenizer_path, do_lower_case=True)
    else:
        raise Exception("unknown model")

    # init batch size
    if model_name == "textcnn":
        batch_size = 256
    elif model_name == "bilstm":
        batch_size = 256
    elif model_name == "rcnn":
        batch_size = 256
    elif model_name in MODEL_CONFIG:
        # batch_size = 32
        #TODO change batch size refer to gpu
        batch_size = 64

    # init embedding
    token_embedding = None
    if embedding_type is not None:
        if embedding_type == 'glove':
            token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                        embedding_dim=300,
                                        pretrained_file='embed/glove.6B/glove.6B.300d.txt',
                                        vocab=vocab).weight.data
        elif embedding_type == 'word2vec':
            token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                        embedding_dim=300,
                                        pretrained_file='embed/word2vec/word2vec-google-news-300.txt',
                                        vocab=vocab).weight.data
        elif embedding_type == 'fasttext':
            token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                        embedding_dim=300,
                                        pretrained_file='embed/fasttext/wiki.en.vec',
                                        vocab=vocab).weight.data
        elif embedding_type.lower() == 'none':
            print('no pretrained embeddings')
        else:
            print('unknown embeddings')

    # label num
    # TODO: 替换为project中的label
    all_labels = set()
    for obj in train_data:
        all_labels.add(obj['labels'])
    for obj in valid_data:
        all_labels.add(obj['labels'])
    for obj in test_data:
        all_labels.add(obj['labels'])

    all_labels = sorted(list(all_labels))

    # init dataset
    train_dataset = IssueDataset(train_data, all_labels, tokenizer)
    valid_dataset = IssueDataset(valid_data, all_labels, tokenizer)
    test_dataset = IssueDataset(test_data, all_labels, tokenizer)

    num_workers = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=4, num_workers=num_workers)

    # init model
    class_num = len(all_labels)
    if model_name == "textcnn":
        model = TextCNN(num_classes=class_num, vocab_size=vocab.get_vocab_size(), embedding_size=300,
                        word_embeddings=token_embedding)
    elif model_name == "bilstm":
        model = BiLSTM(num_classes=class_num, vocab_size=vocab.get_vocab_size(), embedding_size=300,
                       word_embeddings=token_embedding)
    elif model_name == "rcnn":
        model = RCNN(num_classes=class_num, vocab_size=vocab.get_vocab_size(), embedding_size=300,
                     word_embeddings=token_embedding)
    elif model_name in MODEL_CONFIG:
        if not local_model:
            model = Bert(num_classes=class_num, model_name=model_name, use_sequence=use_sequence, disablefinetune=disablefinetune, local_model=local_model)
        else:
            model = Bert(num_classes=class_num, model_name=model_path, use_sequence=use_sequence, disablefinetune=disablefinetune, local_model=local_model)    
    else:
        raise Exception("unknown model")

    # train
    trainer = pl.Trainer(
        # accelerator='ddp',
        amp_backend='native',
        amp_level='O2',
        gpus=[device],
        callbacks=[EarlyStopping(monitor='val_loss')],
        checkpoint_callback=False
    )
    trainer.fit(model,
                train_dataloader=train_loader,
                val_dataloaders=[valid_loader],
                )

    ret = trainer.test(model, test_dataloaders=test_loader)

    # predict
    model.eval()
    pred_dict = {
        'title': [],
        'description': [],
        'true_label': [],
        'pred_label': [],
    }
    
    # 使用模型进行预测，并获取在测试集上的结果
    if do_predict:
        text_list = []
        for i in tqdm.tqdm(range(len(test_data)), desc="generate predictions for test data"):
            obj = test_data[i]
            text = obj['title'] + ' ' + obj['description']
            text_ids = tokenizer(text, truncation=True, max_length=128, padding='max_length')['input_ids']
            text_list.append(text_ids)
            pred_dict['title'].append(obj['title'])
            pred_dict['description'].append(obj['description'])
            pred_dict['true_label'].append(obj['labels'])
            
            if (i != 0 and i % 32 == 0) or (i == len(test_data) - 1):
                text_list = torch.tensor(text_list, dtype=torch.long)
                logits = model(text_list)
                for i in range(len(logits)):
                    pred_dict['pred_label'].append(test_dataset.id_to_label[int(logits[i].argmax())])
                text_list = []

        df = pd.DataFrame(pred_dict)
        save_path = os.path.join('./output', 'eval')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        name = 'train_' + train_data_path.split('/')[-1].split('.')[0] + '_test_' + test_data_path.split('/')[-1].split('.')[0]
        name = os.path.join(save_path, name)
        print(name)
        df.to_csv(f'{name}.csv', index=False)

    return ret[0]


def main():
    parser = argparse.ArgumentParser(description='Training parameters.')
    parser.add_argument('--device', default=0, type=int, required=False, help='使用的实验设备, -1:CPU, >=0:GPU')
    parser.add_argument('--model', default='textcnn', type=str, required=False, help='模型名称')
    parser.add_argument('--embed', default='glove', type=str, required=False, help='词嵌入')
    parser.add_argument('--train_file', type=str, required=True, help='训练文件路径')
    parser.add_argument('--test_file', type=str, required=True, help='测试文件路径')
    parser.add_argument('--sequence', required=False, action="store_true", help='序列模型')
    parser.add_argument('--disablefinetune', required=False, action="store_true", help='禁止微调')
    parser.add_argument('--train_time', default=1, type=int, required=False, help='训练次数')
    parser.add_argument('--local_model', required=False, action="store_true", help='使用本地模型')
    parser.add_argument('--do_predict', required=False, action="store_true", help='获取测试机预测结果')

    args = parser.parse_args()
    print('args:\n' + args.__repr__())

    if not args.local_model:
        out_name = f"output/rq1/{args.model.replace('-', '_').replace('/', '_')}_{args.embed}_1_out.csv"
    else:
        # args.model like '../models/seBERT/pytorch.bin', model_path = 'seBERT'
        model_path = args.model.split('/')[-2]
        out_name = f"output/rq1/{model_path.replace('-', '_').replace('/', '_')}_{args.embed}_1_out.csv"

    if os.path.exists(out_name):
        metric_dict_df = pd.read_csv(out_name)
        metric_dict = metric_dict_df.to_dict(orient="list")
    else:
        if not os.path.exists('./output/rq1'):
            os.makedirs('output/rq1')

        metric_dict = {
            'repo': [],
            'test_acc_1_epoch': [],
            'test_precision_1_epoch': [],
            'test_recall_1_epoch': [],
            'test_f1_marco_1_epoch': [],
            'test_f1_marco_weight_1_epoch': [],
            'test_f1_mirco_1_epoch': [],

            'test_acc_2_epoch': [],
            'test_precision_2_epoch': [],
            'test_recall_2_epoch': [],
            'test_f1_marco_2_epoch': [],
            'test_f1_marco_weight_2_epoch': [],
            'test_f1_mirco_2_epoch': [],

        }
        metric_dict_df = pd.DataFrame(metric_dict)


    print(f'train_file:{args.train_file}, test-file:{args.test_file}')

    each_metrics = train_single(args.train_file, args.test_file, args.model, args.embed, args.device, args.sequence, args.disablefinetune, args.local_model, args.do_predict)
    name = 'train_' + args.train_file.split('/')[-1].split('.')[0] + '_test_' + args.test_file.split('/')[-1].split('.')[0] + '.csv'
    metric_dict['repo'].append(name)
    for k, v in each_metrics.items():
        if k in metric_dict:
            metric_dict[k].append(v)

    df = pd.DataFrame(metric_dict)
    df.to_csv(out_name, index=False)
    

if __name__ == "__main__":
    main()
