
import torch
from torch import nn
import numpy as np
from torch.nn import functional as F

import pytorch_lightning as pl
import torchmetrics

from ..metrics.accuracy import MultiLabelAccuracy
from ..metrics.precision import MultiLabelPrecision
from ..metrics.recall import MultiLabelRecall

class RCNN(pl.LightningModule):
    def __init__(self, num_classes: int, vocab_size:int, embedding_size:int=300, word_embeddings=None, dropout:float=0.5):
        super(RCNN, self).__init__()

        self.hidden_size = 64
        self.hidden_layers = 1
        self.hidden_size_linear = 64
        self.num_classes = num_classes
        self.embed_size = embedding_size
        self.vocab_size = vocab_size
        self.dropout_keep = dropout
        
        # Embedding Layer
        self.embeddings = nn.Embedding(vocab_size, self.embed_size, padding_idx=1)
        if word_embeddings is not None:
            self.embeddings.weight = nn.Parameter(word_embeddings, requires_grad=True)
        
        # Bi-directional LSTM for RCNN
        self.lstm = nn.LSTM(input_size = self.embed_size,
                            hidden_size = self.hidden_size,
                            num_layers = self.hidden_layers,
                            dropout = self.dropout_keep,
                            batch_first=True,
                            bidirectional = True)
        
        self.dropout = nn.Dropout(self.dropout_keep)
        
        # Linear layer to get "convolution output" to be passed to Pooling Layer
        self.W = nn.Linear(
            self.embed_size + 2*self.hidden_size,
            self.hidden_size_linear
        )
        
        # Tanh non-linearity
        self.tanh = nn.Tanh()
        
        # Fully-Connected Layer
        self.fc = nn.Linear(
            self.hidden_size_linear,
            self.num_classes
        )
        
        self.sigmoid = nn.Sigmoid()

        # self.loss = nn.BCELoss()
        self.loss = nn.CrossEntropyLoss()


        self.metrics = nn.ModuleDict()
        stage_name = ['train', 'valid', 'test']
        # for stage in stage_name:
        #     for k in range(1, 6):
        #         self.metrics[f"{stage}_acc_{k}"] = MultiLabelAccuracy(top_k=k)
        #         self.metrics[f"{stage}_precision_{k}"] = MultiLabelPrecision(top_k=k)
        #         self.metrics[f"{stage}_recall_{k}"] = MultiLabelRecall(top_k=k)
        
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

    def forward(self, input_ids):
        # x.shape = (seq_len, batch_size)
        embedded_sent = self.embeddings(input_ids)
        # embedded_sent.shape = (seq_len, batch_size, embed_size)

        lstm_out, (h_n,c_n) = self.lstm(embedded_sent)
        # lstm_out.shape = (seq_len, batch_size, 2 * hidden_size)
        
        input_features = torch.cat([lstm_out,embedded_sent], 2) # .permute(1,0,2)
        # final_features.shape = (batch_size, seq_len, embed_size + 2*hidden_size)
        
        linear_output = self.tanh(
            self.W(input_features)
        )
        # linear_output.shape = (batch_size, seq_len, hidden_size_linear)
        
        linear_output = linear_output.permute(0,2,1) # Reshaping fot max_pool
        
        max_out_features = F.max_pool1d(linear_output, linear_output.shape[2]).squeeze(2)
        # max_out_features.shape = (batch_size, hidden_size_linear)
        
        max_out_features = self.dropout(max_out_features)
        final_out = self.fc(max_out_features)
        # return self.sigmoid(final_out)
        return final_out

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        logits = self.forward(**x)
        loss = self.loss(logits, y.float())
        # Logging to TensorBoard by default
        self.log('train_loss', loss)
        print("loss: ", loss)
        
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
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
        return optimizer
