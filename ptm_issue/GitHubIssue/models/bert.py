import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader, random_split
from transformers import (AlbertModel, AlbertTokenizer,
                          AutoModelForSequenceClassification,
                          AutoModelForTokenClassification, AutoTokenizer,
                          BertForSequenceClassification, BertModel,
                          BertTokenizer, RobertaForSequenceClassification,
                          RobertaModel, RobertaTokenizer,
                          T5ForSequenceClassification, T5Tokenizer,
                          XLNetForSequenceClassification, XLNetTokenizer)

from ..loss.focal_loss import FocalLoss
from ..metrics.accuracy import MultiLabelAccuracy
from ..metrics.precision import MultiLabelPrecision
from ..metrics.recall import MultiLabelRecall
from ..metrics.select_f1 import SelectedClassesF1Score

MODEL_CONFIG = {
    "bert-base-uncased": BertModel,
    "albert-base-v2": AlbertModel,
    "roberta-base": RobertaModel,
    "microsoft/codebert-base": RobertaModel,
}

SEQUENCE_MODEL_CONFIG = {
    "xlnet-base-cased": XLNetForSequenceClassification,
    "bert-base-uncased": BertForSequenceClassification,
    "roberta-base": RobertaForSequenceClassification,
    "microsoft/codebert-base": BertForSequenceClassification,
    "codebert-base": BertForSequenceClassification,
    "seBERT": BertForSequenceClassification,
    "jeniya/BERTOverflow": AutoModelForSequenceClassification,
    "BERTOverflow": AutoModelForSequenceClassification,
    "t5-base": T5ForSequenceClassification,
    "t5-large": T5ForSequenceClassification,
}

class Bert(pl.LightningModule):
    def __init__(self, num_classes: int, base_lr: float=5e-5, model_name: str='bert-base-uncased', use_sequence: bool=False, disablefinetune: bool=False, local_model: bool=False):
        super().__init__()
        self.class_num = num_classes
        self.base_lr = base_lr
        self.model_name = model_name
        self.use_sequence = use_sequence
        self.disablefinetune = disablefinetune
        self.local_model = local_model

        print(f"current model is :{model_name}")
        print(f"current num_classes is :{num_classes}")

        # 本地模型需要从路径中提取出模型名称
        if self.local_model:
            self.model_path = self.model_name # 本地模型上级文件夹
            model_name = self.model_name.split('/')[-1]
            self.model_name = model_name

        if not self.use_sequence:
            if not self.local_model:
                self.model = MODEL_CONFIG[model_name].from_pretrained(model_name)
            else:
                self.model = MODEL_CONFIG[model_name].from_pretrained(self.model_path)
        else:
            if self.local_model:
                self.config = self.model_path + "/config.json"
                # model_file = self.model_path + "/pytorch_model.bin"
                # model_state_dict = torch.load(model_file)
                # self.model = SEQUENCE_MODEL_CONFIG[model_name].from_pretrained(self.model_path, state_dict=model_state_dict, num_labels=num_classes, ignore_mismatched_sizes=True)
                self.model = SEQUENCE_MODEL_CONFIG[model_name].from_pretrained(self.model_path, config=self.config, num_labels=num_classes, ignore_mismatched_sizes=True)
                # self.model = SEQUENCE_MODEL_CONFIG[model_name].from_pretrained(self.model_path, num_labels=num_classes, ignore_mismatched_sizes=True)
            else:
                # ignore_mismatched_sizes will randomly generate the initial parameters for classifier
                # after transformers version == 4.9.0
                print(f"model_name:{model_name}")
                self.model = SEQUENCE_MODEL_CONFIG[model_name].from_pretrained(model_name, num_labels=num_classes, ignore_mismatched_sizes=True)
                # code for transformers version == 4.5.1
                # self.model = SEQUENCE_MODEL_CONFIG[model_name].from_pretrained(model_name, num_labels=num_classes)
            
            if disablefinetune:  # disable finetune for sequence model
                for name, param in self.model.named_parameters():
                    # if 'logits' not in name: # classifier layer for xlnet
                    #     param.requires_grad = False

                    if 'classifier' in name: # classifier layer for bert base model
                        param.requires_grad = True
                    elif 'classification_head' not in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False

        # for name, param in self.model.named_parameters():
        #     print(name, param.shape)

        self.hid_dim = self.model.config.hidden_size
        
        # self.fc_0 = nn.Linear(self.hid_dim, self.hid_dim, bias=True)

        self.fc = nn.Linear(self.hid_dim, self.class_num, bias=True)
        self.dropout = nn.Dropout(p=0.5)
        self.loss = nn.CrossEntropyLoss()
        # self.class_weights = torch.tensor([1.2, 1.2, 1.2, 1.0, 0.8]) 
        # self.loss = FocalLoss(gamma=2.0, alpha=self.class_weights)
        # self.loss = FocalLoss(gamma=5.0)
        
        # self.class_weights = torch.tensor([2.0, 2.0, 2.0, 1.0])
        # self.class_weights = torch.tensor([1.4040, 2.7983, 5.7148, 0.3629])
        # self.class_weights = torch.tensor([0.8733, 1.7467, 3.5568, 0.4997])
        # ['Error', 'Performance', 'deployment', 'other', 'question']
        # self.class_weights = torch.tensor([3.7677, 4.1714, 9.7333, 0.6871, 0.3405])
        # self.class_weights = torch.tensor([1.88385, 2.0857, 4.86665, 0.34355, 0.3405])
        # self.class_weights = torch.tensor([1.88385, 2.0857, 4.86665, 0.6871, 0.3405])
        # self.class_weights = torch.tensor([0.941925, 1.04285, 2.433325, 0.34355, 0.3405])
        # self.class_weights = torch.tensor([1.5, 1.5, 1.5, 1.0, 1.0])
        # self.class_weights = torch.tensor([1.2, 1.2, 1.2, 1.0, 0.8])
        # self.class_weights = torch.tensor([1.2, 1.2, 1.2, 0.8, 0.8]) # crossentrophy2 有效
        # self.loss = nn.CrossEntropyLoss(weight=self.class_weights)
        # print(f'class_weights: {self.class_weights}')


        self.metrics = nn.ModuleDict()
        stage_name = ['train', 'valid', 'test']
        for stage in stage_name:
            for k in range(1, 3):
                self.metrics[f"{stage}_acc_{k}"] = MultiLabelAccuracy(top_k=k)
                self.metrics[f"{stage}_precision_{k}"] = torchmetrics.Precision(top_k=k)
                self.metrics[f"{stage}_recall_{k}"] = torchmetrics.Recall(top_k=k)
                # self.metrics[f"{stage}_acc_{k}"] = MultiLabelAccuracy(top_k=k)
                # self.metrics[f"{stage}_precision_{k}"] = MultiLabelPrecision(top_k=k)
                # self.metrics[f"{stage}_recall_{k}"] = MultiLabelRecall(top_k=k)
                self.metrics[f"{stage}_f1_marco_{k}"] = torchmetrics.F1(average='macro', num_classes=num_classes, top_k=k)
                self.metrics[f"{stage}_f1_marco_weight_{k}"] = torchmetrics.F1(average='weighted', num_classes=num_classes, top_k=k)
                self.metrics[f"{stage}_f1_mirco_{k}"] = torchmetrics.F1(average='micro', num_classes=num_classes, top_k=k)
        
        self.val_selected_f1_score = SelectedClassesF1Score(selected_classes=[0, 1, 2])
        self.save_hyperparameters()

    def forward(self, input_ids):
        # in lightning, forward defines the prediction/inference actions
        output = self.model(**input_ids)
        
        if not self.use_sequence:
            last_hidden_state, pooler_output = output.last_hidden_state, output.pooler_output
            if self.disablefinetune:
                pooler_output = pooler_output.detach()
            x = self.dropout(pooler_output)
            logits = torch.sigmoid(self.fc(x))
            # logits = torch.sigmoid(self.fc(self.fc_0(x)))
        else:
            # logits = torch.sigmoid(output.logits)
            logits = output.logits
            # logits = torch.nn.functional.softmax(logits, dim=-1)
        return logits

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        logits = self.forward(x)
        # print(f'type of logits: {type(logits)}, logits.shape: {logits.shape} ,logits:{logits}')
        # print(f'type of y: {type(y)}, y.shape: {y.shape}, logits{y}')
        loss = self.loss(logits, y.float())
        # Logging to TensorBoard by default
        self.log('train_loss', loss)
        #
        for name, metric in self.metrics.items():
            if name.startswith('train_'):
                if not self.use_sequence:
                    self.log(f"{name}_step", metric(logits, y))
                else:
                    self.log(f"{name}_step", metric(torch.nn.functional.softmax(logits, dim=-1), y))

        return loss

    def training_epoch_end(self, outputs):
        for name, metric in self.metrics.items():
            if name.startswith('train_'):
                self.log(f"{name}_epoch", metric.compute())
                metric.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss(logits, y.float())
        self.log('val_loss', loss)

        for name, metric in self.metrics.items():
            if name.startswith('valid_'):
                if not self.use_sequence:
                    self.log(f"{name}_step", metric(logits, y))
                else:
                    self.log(f"{name}_step", metric(torch.nn.functional.softmax(logits, dim=-1), y))
        
        if self.val_selected_f1_score is not None:
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            self.val_selected_f1_score.update(probabilities, y)
    
    def validation_epoch_end(self, outs):
        for name, metric in self.metrics.items():
            if name.startswith('valid_'):
                self.log(f"{name}_epoch", metric.compute())
                metric.reset()
        
        if self.val_selected_f1_score is not None:
            selected_f1 = self.val_selected_f1_score.compute()
            self.log('val_custom_marco_f1', selected_f1)
            # 重置指标的状态，为下一个 epoch 准备
            self.val_selected_f1_score.reset()

    # def test_step(self, batch, batch_idx):
    #     x, y = batch
    #     logits = self.forward(x)
    #     logits = torch.nn.functional.softmax(logits, dim=-1)
    #     preds = torch.argmax(logits, dim=1)  # 获取预测标签
    #     batch_size = x.size(0)

    #     # 分割logits为大小为4的组
    #     # logits_groups = logits.view(-1, 3, logits.size(-1))
    #     preds_groups = preds.view(-1, 3)

    #     # 计算每组的平均logits
    #     # logits_mean = logits_groups.mean(dim=1)
    #     modes = torch.mode(preds_groups, dim=1).values

    #     # 由于每四个样本的标签相同，我们可以只取每组的第一个标签
    #     y_groups = y.view(-1, self.class_num)
    #     y_sample = y_groups[:, 0][0]

    #     # 对每组的平均logits应用度量标准
    #     for name, metric in self.metrics.items():
    #         if name.startswith('test_'):
    #             if not self.use_sequence:
    #                 self.log(f"{name}_step", metric(logits, y_sample))
    #             else:
    #                 if 'acc' in name:
    #                     self.log(f"{name}_step", metric(logits, y))
    #                 else:
    #                     # self.log(f"{name}_step", metric(torch.nn.functional.softmax(logits_mean, dim=-1), y_sample))
    #                     # print("=========================")
    #                     # print(name)
    #                     # print(f"modes shape is {modes.shape}")
    #                     # print(f"y_groups shape is {y_groups.shape}")
    #                     # print(f"y_sample shape is {y_sample.shape}")
    #                     # print("=========================")
    #                     self.log(f"{name}_step", metric(modes, y_sample))


    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)

        for name, metric in self.metrics.items():
            if name.startswith('test_'):
                if not self.use_sequence:
                    self.log(f"{name}_step", metric(logits, y))
                else:
                    self.log(f"{name}_step", metric(torch.nn.functional.softmax(logits, dim=-1), y))
        # return {'loss': loss, 'pred': pred}

    def test_epoch_end(self, outs):
        # log epoch metric
        for name, metric in self.metrics.items():
            if name.startswith('test_'):
                self.log(f"{name}_epoch", metric.compute())
                metric.reset()

    def configure_optimizers(self):

        if self.use_sequence and self.model_name in ["bert-base-uncased", "codebert-base", "microsoft/codebert-base"]:
            # Bert模型总共分层数量
            total_layers = len(self.model.bert.encoder.layer) # 例：BERT-base有12层
            # 决定从上往下要训练的层数量
            trainable_layers = 4
            # tune emb
            finetune_emb = True 

            # 冻结除了最后trainable_layers层之外的所有层
            for i, layer in enumerate(self.model.bert.encoder.layer):
                if i < total_layers - trainable_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
            
            for param in self.model.bert.embeddings.parameters():
                param.requires_grad = finetune_emb


            # emb_params = list(self.model.bert.embeddings.parameters())
            # encoder_trainable_params = list(filter(lambda p: p.requires_grad, self.model.bert.encoder.layer.parameters()))
            fc_params = list(self.model.bert.pooler.parameters()) + list(self.model.classifier.parameters())

            # base_lr = 5e-6
            base_lr = self.base_lr
            emb_lr = base_lr
            encoder_lr = base_lr
            fc_lr = base_lr

            weight_decay = 1e-2
            # weight_decay = 0
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                # 对embedding层单独设置学习率和权重衰减
                {
                    "params": [p for n, p in self.model.bert.embeddings.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
                    "weight_decay": weight_decay,    # 这里设置你想要的权重衰减
                    "lr": emb_lr    # 这里设置你想要的学习率
                },
                {
                    "params": [p for n, p in self.model.bert.embeddings.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
                    "weight_decay": 0.0,    # 对于bias和LayerNorm.weight不设置权重衰减
                    "lr": emb_lr    # 这里设置你想要的学习率
                },
                # 对其他层单独设置学习率和权重衰减
                {
                    "params": [p for n, p in self.model.bert.encoder.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
                    "weight_decay": weight_decay,    # 这里设置你想要的权重衰减
                    "lr": encoder_lr    # 这里设置你想要的学习率
                },
                {
                    "params": [p for n, p in self.model.bert.encoder.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
                    "weight_decay": 0.0,    # 对于bias和LayerNorm.weight不设置权重衰减
                    "lr": encoder_lr    # 这里设置你想要的学习率
                },
                {
                    'params': fc_params, 
                    'weight_decay': weight_decay,
                    'lr': fc_lr
                },
            ]

            optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
            # optimizer = torch.optim.AdamW([
            #     {'params': emb_params, 'lr': emb_lr, 'weight_decay': weight_decay},
            #     {'params': encoder_trainable_params, 'lr': encoder_lr, 'weight_decay': weight_decay},
            #     {'params': fc_params, 'lr': fc_lr, 'weight_decay': weight_decay},
            # ])

            warmup_epochs = 5
            def warmup_scheduler(epoch):
                if epoch < warmup_epochs:
                    # self.class_weights = torch.tensor([1.5, 1.5, 1.5, 1.5, 0.5]).to("cuda:0") # pytorch_2
                    # self.class_weights = torch.tensor([2.0, 2.0, 2.0, 1.0, 0.5]).to("cuda:0") # pytorch_3
                    # self.class_weights = torch.tensor([1.5, 1.5, 1.5, 1.0, 0.75]).to("cuda:0") # pytorch_8

                    
                    # self.class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]).to("cuda:0") # caffe_1
                    # self.class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]).to("cuda:0") # caffe_4
                    # self.class_weights = torch.tensor([1.2, 1.2, 1.2, 0.8, 0.75]).to("cuda:0") # caffe_4
                    # self.class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]).to("cuda:0") # caffe_6
                    # self.class_weights = torch.tensor([1.0, 1.5, 1.5, 1.0, 0.75]).to("cuda:0") # tensorflow
                    # self.class_weights = torch.tensor([2.0, 2.0, 2.0, 0.75, 0.75]).to("cuda:0") # caffe
                    # self.loss = nn.CrossEntropyLoss(weight=self.class_weights)
                    return float(epoch + 1) / warmup_epochs
                else:
                    # self.loss = nn.CrossEntropyLoss()
                    return 1

            total_epochs = self.trainer.max_epochs
            def decay_scheduler(epoch):
                return 1 - max(0, epoch + 1 - warmup_epochs) / (total_epochs - warmup_epochs)

            def warmup_decay_scheduler(epoch):
                if epoch < warmup_epochs:
                    return float(epoch + 1) / warmup_epochs
                else:
                    return 1 - max(0, epoch + 1 - warmup_epochs) / (total_epochs - warmup_epochs)

            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=warmup_scheduler
            )
            scheduler_d = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=decay_scheduler
            )
            scheduler_wd = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=warmup_decay_scheduler
            )

        else:
            optimizer = torch.optim.AdamW(self.parameters(), lr=2e-5, weight_decay=5e-2)
        

        lr_scheduler = {
            'scheduler': scheduler,
            'interval': 'epoch',  # 或者 'step'，以指定更新频率
            # 'monitor': 'val_loss',  # 指示调度器监控 val_loss 指标
            'frequency': 1       # 指定每多少个interval更新一次学习率
        }

        lr_scheduler_d = {
            'scheduler': scheduler_d,
            'interval': 'epoch',  # 或者 'step'，以指定更新频率
            # 'monitor': 'val_loss',  # 指示调度器监控 val_loss 指标
            'frequency': 1       # 指定每多少个interval更新一次学习率
        }
        lr_scheduler_d = {
            'scheduler': scheduler_wd,
            'interval': 'epoch',  # 或者 'step'，以指定更新频率
            # 'monitor': 'val_loss',  # 指示调度器监控 val_loss 指标
            'frequency': 1       # 指定每多少个interval更新一次学习率
        }
        return [optimizer], [lr_scheduler]
        # return [optimizer], []
    

        

