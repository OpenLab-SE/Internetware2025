import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import torchmetrics

import torch

from ..metrics.accuracy import MultiLabelAccuracy
from ..metrics.precision import MultiLabelPrecision
from ..metrics.recall import MultiLabelRecall

class TextCNN(pl.LightningModule):
    def __init__(self, num_classes: int, vocab_size:int, embedding_size:int=300, word_embeddings=None, dropout:float=0.5):
        super().__init__()

        in_channels = 1
        kernel_num = 100
        kernel_size = [3, 4, 5]

        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        if word_embeddings is not None:
            self.embeddings.weight = nn.Parameter(word_embeddings, requires_grad=True)
        # self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        self.convs1 = nn.ModuleList([nn.Conv2d(in_channels, kernel_num, (K, embedding_size)) for K in kernel_size])
        '''
        self.conv13 = nn.Conv2d(in_channels, kernel_num, (3, embedding_size))
        self.conv14 = nn.Conv2d(in_channels, kernel_num, (4, embedding_size))
        self.conv15 = nn.Conv2d(in_channels, kernel_num, (5, embedding_size))
        '''
        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(len(kernel_size)*kernel_num, num_classes)

        # self.loss = nn.BCELoss()
        self.loss = nn.CrossEntropyLoss()

        self.metrics = nn.ModuleDict()
        stage_name = ['train', 'valid', 'test']
        # for stage in stage_name:
        #     for k in range(1, 6):
        #         self.metrics[f"{stage}_acc_{k}"] = MultiLabelAccuracy(top_k=k)
        #         self.metrics[f"{stage}_precision_{k}"] = MultiLabelPrecision(top_k=k)
        #         self.metrics[f"{stage}_recall_{k}"] = MultiLabelRecall(top_k=k)
                # self.metrics[f"{stage}_f1_{k}"] = torchmetrics.F1(average=avg, num_classes=num_classes, top_k=k)

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

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, input_ids):
        # in lightning, forward defines the prediction/inference actions
        x = self.embeddings(input_ids) # (N, W, D)

        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        '''
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        '''
        x = self.dropout(x)  # (N, len(Ks)*Co)
        # logit = F.softmax(self.fc1(x), dim=0)  # (N, C)
        # logit = torch.sigmoid(self.fc1(x))  # (N, C)
        logit = self.fc1(x)  # (N, C)
        return logit

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        logits = self.forward(**x)
        loss = self.loss(logits, y.float())
        # Logging to TensorBoard by default
        self.log('train_loss', loss)
        
        for name, metric in self.metrics.items():
            if name.startswith('train_'):
                # self.log(f"{name}_step", metric(logits, y))
                self.log(f"{name}_step", metric(torch.nn.functional.softmax(logits, dim=-1), y))
        return loss

    def training_epoch_end(self, outputs):
        for name, metric in self.metrics.items():
            if name.startswith('train_'):
                self.log(f"{name}_epoch", metric.compute())
                metric.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(**x)
        loss = self.loss(logits, y.float())
        self.log('val_loss', loss)

        for name, metric in self.metrics.items():
            if name.startswith('valid_'):
                # self.log(f"{name}_step", metric(logits, y))
                self.log(f"{name}_step", metric(torch.nn.functional.softmax(logits, dim=-1), y))

    def validation_epoch_end(self, outs):
        for name, metric in self.metrics.items():
            if name.startswith('valid_'):
                self.log(f"{name}_epoch", metric.compute())
                metric.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(**x)

        for name, metric in self.metrics.items():
            if name.startswith('test_'):
                # self.log(f"{name}_step", metric(logits, y))
                self.log(f"{name}_step", metric(torch.nn.functional.softmax(logits, dim=-1), y))
        # return {'loss': loss, 'pred': pred}

    def test_epoch_end(self, outs):
        # log epoch metric
        for name, metric in self.metrics.items():
            if name.startswith('test_'):
                self.log(f"{name}_epoch", metric.compute())
                metric.reset()

    def configure_optimizers(self):
        # optimizer = torch.optim.SGD(self.parameters(), lr=0.1)
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
        return optimizer
