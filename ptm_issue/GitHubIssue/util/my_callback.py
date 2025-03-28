import os

import pandas as pd
import torch
import tqdm
from GitHubIssue.tokenizer.allennlp_tokenizer import AllennlpTokenizer
from pytorch_lightning import Callback
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import classification_report
from GitHubIssue.dataset.issue_dataset import concat_str

from GitHubIssue.metrics.log_metrics import log_metrics

BERT_MODEL_CONFIG = [
    "bert-base-uncased",
    "xlnet-base-cased",
    "albert-base-v2",
    "roberta-base",
    "microsoft/codebert-base",
    "jeniya/BERTOverflow",
    "BERTOverflow",
    "huggingface/CodeBERTa-language-id",
    "seBERT",
]

GPT_MODEL_CONFIG = [
    "gpt2",
    "microsoft/CodeGPT-small-py"
]

TRANSFORMER_MODEL_CONFIG = [
    "t5-base",
    "t5-large",
    "Salesforce/codet5-base"
]


class MySubClassPredictCallback(Callback):
    def __init__(self, stage, model_name, trial, model, tokenizer, test_dataset, train_file, test_file, all_labels, device):
        super().__init__()
        self.stage = stage
        self.model_name = model_name
        self.trial = trial
        self.model = model
        self.tokenizer = tokenizer
        self.test_dataset = test_dataset
        self.train_file = train_file
        self.test_file = test_file
        self.all_labels = all_labels
        self.device = device

    def log_custom_avg(
        self,
        trainer,
        metrics_dict, 
        metric_list=["Error", "Performance", "deployment"], 
        ):
        marco_avg = 0
        count = 0
        for k, v in metrics_dict.items():
            if k in metric_list:
                marco_avg += float(v["f1-score"])
                count += 1
        if marco_avg != 0:
            trainer.logger.experiment.add_scalar(f'{self.stage}_custom_marco_f1', marco_avg / count, global_step=trainer.global_step)
            # self.logger.log('val_custom_macro_f1', marco_avg / count)
            # logger.log_metrics({'val_custom_macro_f1': marco_avg / count})


    # def on_epoch_end(self, trainer, pl_module):
    def on_train_epoch_end(self, trainer, pl_module):
        pred_dict = {
            'title': [],
            'description': [],
            'true_label': [],
            'pred_label': [],
        }
        print("start predict")
        test_data = self.test_dataset.data

        if self.model_name in GPT_MODEL_CONFIG:
            self.tokenizer.padding_side = 'right'

        # get model predict labels
        text_list = []
        for i in tqdm.tqdm(range(len(test_data)), desc="generate predictions for test data"):
            obj = test_data[i]
            # text = obj['title'] + ' ' + obj['description']
            # text = concat_str(self.tokenizer, [obj['title'], obj['description']])
            title = "Title: "+ obj['title']
            description = "Details: " + obj['description']
            if obj.get("commment_concat_str") is not None:
                # text += " " + obj["commment_concat_str"]
                comments_list = obj['commment_concat_str'].split("concatcommentsign")
                if len(comments_list) != 0:
                    comments_list[0] = "Comments: " + comments_list[0]
                text = concat_str(self.tokenizer, [title, description] + comments_list)
            else:
                text = concat_str(self.tokenizer, [title, description])
            # text_ids = tokenizer(text, truncation=True, max_length=512, padding='max_length')['input_ids']
            if isinstance(self.tokenizer, AllennlpTokenizer):
                text_ids = self.tokenizer(text, truncation=True, max_length=512, padding='max_length')
                # allennlp tokenizer不会自动附加维度，因此最后增加一维，便于后续concat
                text_ids['input_ids'] = torch.tensor(text_ids['input_ids'], dtype=torch.long).unsqueeze(0)
                text_list.append(text_ids)
            else:
                text_ids = self.tokenizer(
                    text, 
                    truncation=True, 
                    max_length=512, 
                    padding='max_length', 
                    return_tensors="pt",
                    return_token_type_ids=True if "token_type_ids" in self.tokenizer.model_input_names else False)
                text_list.append(text_ids)

            pred_dict['title'].append(obj['title'])
            pred_dict['description'].append(obj['description'])
            pred_dict['true_label'].append(obj['labels'])
            
            self.model.eval()
            self.model.to(f'cuda:{self.device}')
            if (i != 0 and i % 8 == 0) or (i == len(test_data) - 1):
                keys = text_list[0].keys()
                inputs = {}
                for k in keys:
                    inputs[k] = torch.cat([item[k] for item in text_list], dim=0).to(f'cuda:{self.device}')

                if isinstance(self.tokenizer, AllennlpTokenizer):
                    logits = self.model(**inputs)
                else:
                    logits = self.model(inputs)

                for i in range(len(logits)):
                    pred_dict['pred_label'].append(self.test_dataset.id_to_label[int(logits[i].argmax())])
                text_list = []

        save_path = os.path.join('./output', 'subclass')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if self.train_file == self.test_file:
            name = self.train_file.split('/')[-1].split('.')[0]
        else:
            name = self.train_file.split('/')[-1].split('.')[0] + '_' + self.test_file.split('/')[-1].split('.')[0]

        name = os.path.join(save_path, name)
        true_label_id = [self.test_dataset.label_to_id[x] for x in pred_dict['true_label']]
        pred_label_id = [self.test_dataset.label_to_id[x] for x in pred_dict['pred_label']]
        report = classification_report(true_label_id, pred_label_id, labels=list(range(len(self.all_labels))),
                                        target_names=list(self.all_labels), output_dict=True)
        print(f"======== stage: {self.stage} sub class metric ============")
        print(report)
        print(f"==========================================================")
        # 确保你使用的是 TensorBoard Logger
        if isinstance(trainer.logger, TensorBoardLogger):
            log_metrics(trainer.logger, report, self.stage + " ", global_step=trainer.global_step)
            if self.stage == "test":
                self.log_custom_avg(trainer, report)

        # save subclass report
        # df = pd.DataFrame(report)
        # df = df.T
        # df.to_csv(f"{name}_{self.model_name.replace('-', '_').replace('/', '_')}_{self.trial}.csv", mode='a')
