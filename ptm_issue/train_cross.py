import argparse
import glob
import json
import os
import random

import numpy as np
import pandas as pd
import torch
import tqdm
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedShuffleSplit

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pytorch_lightning as pl
from allennlp.data.token_indexers.single_id_token_indexer import \
    SingleIdTokenIndexer
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.token_embedders.embedding import Embedding
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from transformers import (AlbertTokenizer, AutoTokenizer, BertTokenizer,
                          GPT2Tokenizer, RobertaTokenizer, T5Tokenizer,
                          XLNetTokenizer)

from GitHubIssue.dataset.allennlp_issue_dataset import \
    AllennlpIssueDatasetReader
from GitHubIssue.dataset.issue_dataset import IssueDataset, concat_str
from GitHubIssue.metrics.log_metrics import log_metrics
from GitHubIssue.models.bert import Bert
from GitHubIssue.models.bilstm import BiLSTM
from GitHubIssue.models.gpt import Gpt
from GitHubIssue.models.rcnn import RCNN
from GitHubIssue.models.textcnn import TextCNN
from GitHubIssue.models.transformer import Transformer
from GitHubIssue.tokenizer.allennlp_tokenizer import AllennlpTokenizer
# from GitHubIssue.models.model import TextLabelRecModel
from GitHubIssue.util.mem import occupy_mem
from GitHubIssue.util.my_callback import MySubClassPredictCallback
from mylogger import CustomTensorBoardLogger

MODEL_CONFIG = [
    "bert-base-uncased",
    "xlnet-base-cased",
    "albert-base-v2",
    "roberta-base",
    "microsoft/codebert-base",
    "jeniya/BERTOverflow",
    "BERTOverflow",
    "huggingface/CodeBERTa-language-id",
    "seBERT",
    "t5-base",
    "t5-large",
    "gpt2"
]

BERT_MODEL_CONFIG = [
    "bert-base-uncased",
    "xlnet-base-cased",
    "albert-base-v2",
    "roberta-base",
    "microsoft/codebert-base",
    "codebert-base",
    "jeniya/BERTOverflow",
    "BERTOverflow",
    "huggingface/CodeBERTa-language-id",
    "seBERT",
]

GPT_MODEL_CONFIG = [
    "gpt2",
    "microsoft/CodeGPT-small-py",
    "CodeGPT-small-py",
]

TRANSFORMER_MODEL_CONFIG = [
    "t5-base",
    "t5-large",
    "Salesforce/codet5-base",
    "codet5-base",
]


TOKENIZER_CONFIG = {
    "bert-base-uncased": BertTokenizer,
    "xlnet-base-cased": XLNetTokenizer,
    "albert-base-v2":  AlbertTokenizer,
    "roberta-base": RobertaTokenizer,
    "microsoft/codebert-base": RobertaTokenizer,
    "codebert-base": RobertaTokenizer,
    "jeniya/BERTOverflow": AutoTokenizer,
    "BERTOverflow": AutoTokenizer,
    "huggingface/CodeBERTa-language-id": RobertaTokenizer,
    "seBERT": BertTokenizer,
    "t5-base": T5Tokenizer,
    "t5-large": T5Tokenizer,
    "Salesforce/codet5-base": RobertaTokenizer,
    "codet5-base": RobertaTokenizer,
    "gpt2": GPT2Tokenizer,
    "microsoft/CodeGPT-small-py": GPT2Tokenizer,
    "CodeGPT-small-py": GPT2Tokenizer
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

def count_labels(data, dataset):
    count = dict()
    for obj in data:
        if count.get(obj['labels']) is None:
            count[obj['labels']] = 1
        else:
            count[obj['labels']] += 1
    from pprint import pprint
    print(f'label count for {dataset} dataset')
    pprint(count)


def train_single(
    train_file: str, 
    valid_file: str, 
    test_file: str, 
    model_name: str, 
    embedding_type=None, 
    device=0, 
    use_sequence=False, 
    disablefinetune=False, 
    local_model=False, 
    do_predict=False,
    batch_size=8,
    base_lr=5e-5,
    trial="trial"):
    
    data = []
    if train_file is not None:
        with open(train_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

    if train_file == test_file and train_file == valid_file:
        X = []
        y = []
        for obj in data:
            X.append(obj)
            y.append(obj['labels'])

        split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
        for train_index, test_index in split.split(X, y):
            train_data, test_data = np.array(X)[train_index], np.array(X)[test_index] # 训练集对应的值
        
        X = []
        y = []
        for obj in train_data:
            X.append(obj)
            y.append(obj['labels'])
        split1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, test_index in split1.split(X, y):
            train_data, valid_data = np.array(X)[train_index], np.array(X)[test_index] #训练集对应的值
    elif train_file == valid_file and train_file != test_file:
        X = []
        y = []
        for obj in data:
            X.append(obj)
            y.append(obj['labels'])
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, valid_index in split.split(X, y):
            train_data, valid_data = np.array(X)[train_index], np.array(X)[valid_index] # 训练集对应的值
        
        test_data = []
        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
    else:
        train_data = data
        valid_data = []
        with open(valid_file, 'r', encoding='utf-8') as f:
            valid_data = json.load(f)
        test_data = []
        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
    
    count_labels(train_data, 'train')
    count_labels(valid_data, 'val')
    count_labels(test_data, 'test')
    
    # 用于评估训练集的结果
    # test_data = data[:split_1]

    # 本地模型需要从路径中提取出模型名称
    if local_model:
        model_path = model_name # 保存本地模型路径
        model_name = model_name.split('/')[-1]
        print(f"model_name: {model_path}")

    # init tokenizer
    if model_name in ["textcnn", "bilstm", "rcnn"]:
        # build vocab
        allennlp_tokenizer = SpacyTokenizer()
        allennlp_token_indexer = SingleIdTokenIndexer(token_min_padding_length=8, lowercase_tokens=True)
        allennlp_datareader = AllennlpIssueDatasetReader(allennlp_tokenizer, {'tokens': allennlp_token_indexer})
        vocab = Vocabulary.from_instances(allennlp_datareader.read(train_file))
        
        from allennlp.data.tokenizers import Token
        ids = allennlp_token_indexer.tokens_to_indices([Token(vocab._padding_token)], vocab)['tokens']
        print(f"padding tokens is {ids}")

        tokenizer = AllennlpTokenizer(vocab, allennlp_tokenizer, allennlp_token_indexer)
    elif model_name in BERT_MODEL_CONFIG or model_name in TRANSFORMER_MODEL_CONFIG:
        if not local_model:
            tokenizer = TOKENIZER_CONFIG[model_name].from_pretrained(model_name)
        else:
            # tokenizer_path = model_path
            tokenizer_path = model_path
            print(f"tokenizer_path: {tokenizer_path}")
            tokenizer = TOKENIZER_CONFIG[model_name].from_pretrained(tokenizer_path, do_lower_case=True)
    elif model_name in GPT_MODEL_CONFIG:
        if not local_model:
            tokenizer = TOKENIZER_CONFIG[model_name].from_pretrained(model_name)
        else:
            # tokenizer_path = model_path
            tokenizer_path = model_path
            print(f"tokenizer_path: {tokenizer_path}")
            tokenizer = TOKENIZER_CONFIG[model_name].from_pretrained(tokenizer_path, do_lower_case=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'
    else:
        raise Exception("unknown model")

    # init batch size
    if model_name == "textcnn":
        batch_size = 64
    elif model_name == "bilstm":
        batch_size = 64
    elif model_name == "rcnn":
        batch_size = 64
    elif model_name in MODEL_CONFIG:
        # batch_size = 32
        #TODO change batch size refer to gpu
        # batch_size = 32
        # batch_size = 32
        # batch_size = 48
        # batch_size = 8
        batch_size = batch_size

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
    all_labels = set()
    for obj in train_data:
        all_labels.add(obj['labels'])
    for obj in valid_data:
        all_labels.add(obj['labels'])
    for obj in test_data:
        all_labels.add(obj['labels'])
    
    # ['Error', 'Low efficiency and Effectiveness', 'deployment', 'other', 'tensor&inputs']
    all_labels = sorted(list(all_labels))
    print(f"all_labels:{all_labels}")

    # init dataset
    train_dataset = IssueDataset(train_data, all_labels, tokenizer)
    # if model_name in GPT_MODEL_CONFIG:
    #     tokenizer.padding_side = 'left'
    valid_dataset = IssueDataset(valid_data, all_labels, tokenizer)
    test_dataset = IssueDataset(test_data, all_labels, tokenizer)

    num_workers = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=8, num_workers=num_workers)

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
    elif model_name in BERT_MODEL_CONFIG:
        if not local_model:
            model = Bert(num_classes=class_num, base_lr=base_lr, model_name=model_name, use_sequence=use_sequence, disablefinetune=disablefinetune, local_model=local_model)
        else:
            model = Bert(num_classes=class_num, base_lr=base_lr, model_name=model_path, use_sequence=use_sequence, disablefinetune=disablefinetune, local_model=local_model)    
    elif model_name in GPT_MODEL_CONFIG:
        if not local_model:
            model = Gpt(num_classes=class_num, base_lr=base_lr, model_name=model_name, use_sequence=use_sequence, disablefinetune=disablefinetune, local_model=local_model)
        else:
            model = Gpt(num_classes=class_num,  base_lr=base_lr, model_name=model_path, use_sequence=use_sequence, disablefinetune=disablefinetune, local_model=local_model)
    elif model_name in TRANSFORMER_MODEL_CONFIG:
        if not local_model:
            model = Transformer(num_classes=class_num, base_lr=base_lr, model_name=model_name, use_sequence=use_sequence, disablefinetune=disablefinetune, local_model=local_model)
        else:
            model = Transformer(num_classes=class_num, base_lr=base_lr, model_name=model_path, use_sequence=use_sequence, disablefinetune=disablefinetune, local_model=local_model)
    else:
        raise Exception("unknown model")

    # add model checkpoint
    # checkpoint_callback  = ModelCheckpoint(
    #     dirpath='ckpts/',
    #     filename=f'{model_name.replace("/", "_")}' + '-{epoch:02d}-{step:04d}',
    #     save_last=True, # whether to save last checkpoint
    #     # save_top_k=-1,
    #     # every_n_train_steps=50
    #     )
    
    
    # max_epochs=35
    max_epochs=30
    log_name = trial
    # log_name= 'newtaglabel_lr_5e-5_bert'
    # log_name= 'newtaglabel_lr_5e-5_t5_enc6_dec0_times10'
    # log_name= 'newtaglabel_lr_5e-5_gpt2_l3_times10'

    log_experiment = 'ep_' + str(max_epochs) + '_maxf1'
    # log_experiment = 'ep_' + str(max_epochs) + '_minloss'
    ckpt_name = f'{model_name.replace("/", "_")}' + f'-best_model_{log_name}_{log_experiment}'
    ckpt_path = f'./ckpts/' + ckpt_name + '.ckpt'
    checkpoint_callback = ModelCheckpoint(
        monitor = 'valid_f1_marco_1_epoch',  # 监视验证集上的marco f1
        # monitor = 'val_custom_marco_f1',  # 监视验证集上的marco f1
        # monitor = 'val_loss',  # 监视验证集上的损失
        # mode='min',          # 模式可以是'min'或'max'，取决于你要最小化还是最大化指标
        mode='max',          # 模式可以是'min'或'max'，取决于你要最小化还是最大化指标
        save_top_k=1,        # 仅保存性能最佳的模型
        # save_weights_only=True,  # 仅保存模型权重而不是整个模型
        filename=ckpt_name,    # 保存文件的名称
        dirpath='ckpts/',
    )


    subclass_predict_callback_val = MySubClassPredictCallback(
        stage="valid",
        model_name=model_name,
        trial=trial,
        model=model,
        tokenizer=tokenizer,
        test_dataset=valid_dataset,
        train_file=train_file,
        test_file=test_file,
        all_labels=all_labels,
        device=device,
    )

    subclass_predict_callback_test = MySubClassPredictCallback(
        stage="test",
        model_name=model_name,
        trial=trial,
        model=model,
        tokenizer=tokenizer,
        test_dataset=test_dataset,
        train_file=train_file,
        test_file=test_file,
        all_labels=all_labels,
        device=device,
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # logger = CustomTensorBoardLogger(save_dir='lightning_logs', name=log_name, log_dir_name=log_experiment)
    # logger = CustomTensorBoardLogger(save_dir='lightning_logs/realtime', name=log_name, log_dir_name=log_experiment)
    logger = CustomTensorBoardLogger(save_dir='lightning_logs/tensorflow', name=log_name, log_dir_name=log_experiment)
    # train
    trainer = pl.Trainer(
        logger=logger,
        # accelerator='ddp',
        max_epochs=max_epochs,
        amp_backend='native',
        # amp_level='O2',
        # amp_level='O0',
        gpus=[device],
        # accumulate_grad_batches=2,
        callbacks=[
            # EarlyStopping(monitor='val_loss'),
            subclass_predict_callback_val,
            subclass_predict_callback_test,
            checkpoint_callback,
            lr_monitor
            ],
        # checkpoint_callback=False
    )
    
    trainer.fit(model,
                train_dataloader=train_loader,
                val_dataloaders=[valid_loader],
                )

    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint["state_dict"])
    # model.load_from_checkpoint(
    #     f'./ckpts/{model_name.replace("/", "_")}' + '-best_model.ckpt',
    #     strict=False
    #     #  num_labels=class_num
    #     )

    ret = trainer.test(model, test_dataloaders=test_loader)

    model.eval()
    pred_dict = {
        'number': [],
        'html_url': [],
        'title': [],
        'description': [],
        'true_label': [],
        'pred_label': [],
    }
    
    if do_predict:
        print("start predict")
        if model_name in GPT_MODEL_CONFIG:
            tokenizer.padding_side = 'right'

        # get model predict labels
        text_list = []
        for i in tqdm.tqdm(range(len(test_data)), desc="generate predictions for test data"):
            obj = test_data[i]
            # text = obj['title'] + ' ' + obj['description']
            # text = concat_str(tokenizer, [obj['title'], obj['description']])
            title = "Title: "+ obj['title']
            description = "Details: " + obj['description']
            if obj.get("commment_concat_str") is not None:
                # text += " " + obj["commment_concat_str"]
                comments_list = obj['commment_concat_str'].split("concatcommentsign")
                if len(comments_list) != 0:
                    comments_list[0] = "Comments: " + comments_list[0]
                text = concat_str(tokenizer, [title, description] + comments_list)
            else:
                text = concat_str(tokenizer, [title, description])
            # text_ids = tokenizer(text, truncation=True, max_length=512, padding='max_length')['input_ids']
            if isinstance(tokenizer, AllennlpTokenizer):
                text_ids = tokenizer(text, truncation=True, max_length=512, padding='max_length')
                # allennlp tokenizer不会自动附加维度，因此最后增加一维，便于后续concat
                text_ids['input_ids'] = torch.tensor(text_ids['input_ids'], dtype=torch.long).unsqueeze(0)
                text_list.append(text_ids)
            else:
                text_ids = tokenizer(
                    text, 
                    truncation=True, 
                    max_length=512, 
                    padding='max_length', 
                    return_tensors="pt",
                    return_token_type_ids=True if "token_type_ids" in tokenizer.model_input_names else False)
                text_list.append(text_ids)
            pred_dict['number'].append(obj['number'])
            pred_dict['html_url'].append(obj['html_url'])
            pred_dict['title'].append(obj['title'])
            pred_dict['description'].append(obj['description'])
            pred_dict['true_label'].append(obj['labels'])
            
            model.to(f'cuda:{device}')
            if (i != 0 and i % 8 == 0) or (i == len(test_data) - 1):
                keys = text_list[0].keys()
                inputs = {}
                for k in keys:
                    inputs[k] = torch.cat([item[k] for item in text_list], dim=0).to(f'cuda:{device}')

                if isinstance(tokenizer, AllennlpTokenizer):
                    logits = model(**inputs)
                else:
                    logits = model(inputs)
                
                for i in range(len(logits)):
                    pred_dict['pred_label'].append(test_dataset.id_to_label[int(logits[i].argmax())])
                text_list = []

        save_path = os.path.join('./output', 'subclass')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if train_file == test_file:
            name = train_file.split('/')[-1].split('.')[0]
        else:
            name = train_file.split('/')[-1].split('.')[0] + '_' + test_file.split('/')[-1].split('.')[0]

        name = os.path.join(save_path, name)
        true_label_id = [test_dataset.label_to_id[x] for x in pred_dict['true_label']]
        pred_label_id = [test_dataset.label_to_id[x] for x in pred_dict['pred_label']]
        report = classification_report(true_label_id, pred_label_id, labels=list(range(len(all_labels))),
                                       target_names=list(all_labels), output_dict=True)
        print(report)
        # 确保你使用的是 TensorBoard Logger
        if isinstance(trainer.logger, TensorBoardLogger):
            log_metrics(trainer.logger, report, '', global_step=trainer.global_step)

        # save subclass report
        df = pd.DataFrame(report)
        df = df.T
        df.to_csv(f"{name}_{model_name.replace('-', '_').replace('/', '_')}_{trial}.csv", mode='a')

        # save eval result
        df = pd.DataFrame(pred_dict)
        save_path = os.path.join('./output', 'eval')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if train_file == test_file:
            name = train_file.split('/')[-1].split('.')[0]
        else:
            name = train_file.split('/')[-1].split('.')[0] + '_' + test_file.split('/')[-1].split('.')[0]
        name = os.path.join(save_path, name)
        df.to_csv(f"{name}_{model_name.replace('-', '_').replace('/', '_')}_{trial}.csv", index=False)

        # ============================  predict valid file ===========================
        # pred_dict = {
        #     'number': [],
        #     'html_url': [],
        #     'title': [],
        #     'description': [],
        #     'true_label': [],
        #     'pred_label': [],
        # }

        # print("start predict")
        # if model_name in GPT_MODEL_CONFIG:
        #     tokenizer.padding_side = 'right'

        # # get model predict labels
        # text_list = []
        # for i in tqdm.tqdm(range(len(valid_data)), desc="generate predictions for val data"):
        #     obj = valid_data[i]
        #     # text = obj['title'] + ' ' + obj['description']
        #     # text = concat_str(tokenizer, [obj['title'], obj['description']])
        #     title = "Title: "+ obj['title']
        #     description = "Details: " + obj['description']
        #     if obj.get("commment_concat_str") is not None:
        #         # text += " " + obj["commment_concat_str"]
        #         comments_list = obj['commment_concat_str'].split("concatcommentsign")
        #         if len(comments_list) != 0:
        #             comments_list[0] = "Comments: " + comments_list[0]
        #         text = concat_str(tokenizer, [title, description] + comments_list)
        #     else:
        #         text = concat_str(tokenizer, [title, description])
        #     # text_ids = tokenizer(text, truncation=True, max_length=512, padding='max_length')['input_ids']
        #     if isinstance(tokenizer, AllennlpTokenizer):
        #         text_ids = tokenizer(text, truncation=True, max_length=512, padding='max_length')
        #         # allennlp tokenizer不会自动附加维度，因此最后增加一维，便于后续concat
        #         text_ids['input_ids'] = torch.tensor(text_ids['input_ids'], dtype=torch.long).unsqueeze(0)
        #         text_list.append(text_ids)
        #     else:
        #         text_ids = tokenizer(
        #             text, 
        #             truncation=True, 
        #             max_length=512, 
        #             padding='max_length', 
        #             return_tensors="pt",
        #             return_token_type_ids=True if "token_type_ids" in tokenizer.model_input_names else False)
        #         text_list.append(text_ids)
        #     pred_dict['number'].append(obj['number'])
        #     pred_dict['html_url'].append(obj['html_url'])
        #     pred_dict['title'].append(obj['title'])
        #     pred_dict['description'].append(obj['description'])
        #     pred_dict['true_label'].append(obj['labels'])
            
        #     model.to(f'cuda:{device}')
        #     if (i != 0 and i % 8 == 0) or (i == len(valid_data) - 1):
        #         keys = text_list[0].keys()
        #         inputs = {}
        #         for k in keys:
        #             inputs[k] = torch.cat([item[k] for item in text_list], dim=0).to(f'cuda:{device}')

        #         if isinstance(tokenizer, AllennlpTokenizer):
        #             logits = model(**inputs)
        #         else:
        #             logits = model(inputs)
                
        #         for i in range(len(logits)):
        #             pred_dict['pred_label'].append(valid_dataset.id_to_label[int(logits[i].argmax())])
        #         text_list = []

        # save_path = os.path.join('./output', 'subclass')
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)

        # if train_file == valid_file:
        #     name = train_file.split('/')[-1].split('.')[0]
        # else:
        #     name = train_file.split('/')[-1].split('.')[0] + '_' + valid_file.split('/')[-1].split('.')[0]

        # name = os.path.join(save_path, name)
        # true_label_id = [valid_dataset.label_to_id[x] for x in pred_dict['true_label']]
        # pred_label_id = [valid_dataset.label_to_id[x] for x in pred_dict['pred_label']]
        # report = classification_report(true_label_id, pred_label_id, labels=list(range(len(all_labels))),
        #                                target_names=list(all_labels), output_dict=True)
        # print(report)
        # # 确保你使用的是 TensorBoard Logger
        # if isinstance(trainer.logger, TensorBoardLogger):
        #     log_metrics(trainer.logger, report, '', global_step=trainer.global_step)

        # # save subclass report
        # df = pd.DataFrame(report)
        # df = df.T
        # df.to_csv(f"{name}_{model_name.replace('-', '_').replace('/', '_')}_{trial}.csv", mode='a')

        # # save eval result
        # df = pd.DataFrame(pred_dict)
        # save_path = os.path.join('./output', 'eval')
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)

        # if train_file == valid_file:
        #     name = train_file.split('/')[-1].split('.')[0]
        # else:
        #     name = train_file.split('/')[-1].split('.')[0] + '_' + valid_file.split('/')[-1].split('.')[0]
        # name = os.path.join(save_path, name)
        # df.to_csv(f"{name}_{model_name.replace('-', '_').replace('/', '_')}_{trial}.csv", index=False)

    # 训练结束后删除 checkpoint 文件
    if os.path.isfile(ckpt_path):
        os.remove(ckpt_path)  # 删除文件
        print(f"File {ckpt_path} has been removed successfully")
    else:
        print(f"File {ckpt_path} does not exist")

    return ret[0]


def main():
    parser = argparse.ArgumentParser(description='Training parameters.')
    parser.add_argument('--device', default=0, type=int, required=False, help='使用的实验设备, -1:CPU, >=0:GPU')
    parser.add_argument('--model', default='textcnn', type=str, required=False, help='模型名称')
    parser.add_argument('--embed', default='glove', type=str, required=False, help='词嵌入')
    parser.add_argument('--sequence', required=False, action="store_true", help='序列模型')
    parser.add_argument('--disablefinetune', required=False, action="store_true", help='禁止微调')
    parser.add_argument('--train_time', default=1, type=int, required=False, help='训练次数')
    parser.add_argument('--local_model', required=False, action="store_true", help='使用本地模型')
    parser.add_argument('--do_predict', required=False, action="store_true", help='获取测试集预测结果')
    
    parser.add_argument('--file', type=str, help='训练数据')
    parser.add_argument('--train_file', type=str, help='训练数据')
    parser.add_argument('--valid_file', type=str, help='验证数据')
    parser.add_argument('--test_file', type=str, help='测试数据')
    parser.add_argument('--batch_size', default=8, type=int, required=False, help='模型输入batch size')
    parser.add_argument('--base_lr', default=5e-5, type=float, required=False, help='训练学习率')
    parser.add_argument('--trial', type=str, help='训练名称')
    

    args = parser.parse_args()
    print('args:\n' + args.__repr__())
    
    # 占用全部显存
    if not args.local_model:
        out_name = f"output/rq1/{args.model.replace('-', '_').replace('/', '_')}_{args.embed}_{args.trial}_out.csv"
    else:
        # model_path = args.model.split('/')[-2]
        model_path = args.model.split('/')[-1]
        out_name = f"output/rq1/{model_path.replace('-', '_').replace('/', '_')}_{args.embed}_{args.trial}_out.csv"

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
    
    # train on concat file
    # training_times = 10
    for t in  range(args.train_time):
        concat_file = args.train_file
        # concat_file = './my_data/train/concat_concat/concat_concat.txt'
        # concat_file = './my_data/train/pytorch-CycleGAN-and-pix2pix_TRAIN_Aug/pytorch-CycleGAN-and-pix2pix_TRAIN_Aug.txt'
        # concat_file = './my_data/train/Real-Time-Voice-Cloning_TRAIN_Aug/Real-Time-Voice-Cloning_TRAIN_Aug.txt'
        # concat_file = './my_data/train/EasyOCR_TRAIN_Aug/EasyOCR_TRAIN_Aug.txt'
        # concat_file = './my_data/train/recommenders1_TRAIN_Aug/recommenders1_TRAIN_Aug.txt'
        # concat_file = './my_data/train/streamlit1_TRAIN_Aug/streamlit1_TRAIN_Aug.txt'
        # random.seed(hash(concat_file))
        train_file, test_file = args.train_file, args.test_file
        valid_file = args.valid_file
        print(f'train_file:{train_file}, test_file:{test_file}')
        each_metrics = train_single(
            train_file, 
            valid_file, 
            test_file, 
            args.model, 
            args.embed, 
            args.device, 
            args.sequence, 
            args.disablefinetune, 
            args.local_model, 
            args.do_predict, 
            args.batch_size,
            args.base_lr,
            args.trial)
        name = concat_file.split('/')[-1].split('.')[0]
        metric_dict['repo'].append(name + '_times_' + str(t))
        for k, v in each_metrics.items():
            if k in metric_dict:
                metric_dict[k].append(v)

        df = pd.DataFrame(metric_dict)
        df.to_csv(out_name, index=False)
if __name__ == "__main__":
    # os.environ['http_proxy'] = 'http://nbproxy.mlp.oppo.local:8888'
    # os.environ['https_proxy'] = 'http://nbproxy.mlp.oppo.local:8888'
    main()
    # del os.environ['http_proxy']   #用完需要del代理，否则训练的所有流量都走代理访问，有安全风险
    # del os.environ['https_proxy']
