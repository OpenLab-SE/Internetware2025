import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torch.nn.utils.rnn as rnn
import pytorch_lightning as pl
import torchmetrics

import torch

from transformers import BertTokenizer, BertModel

from ..metrics.accuracy import MultiLabelAccuracy
from ..metrics.precision import MultiLabelPrecision
from ..metrics.recall import MultiLabelRecall

class BiLSTM(pl.LightningModule):
    def __init__(self, num_classes: int, vocab_size:int, embedding_size:int=300, word_embeddings=None, dropout:float=0.5):
        super().__init__()
        self.hidden_size = 64
        self.num_layers = 2
        self.num_directions = 2

        self.embeddings = nn.Embedding(vocab_size, embedding_size, padding_idx=1)
        if word_embeddings is not None:
            self.embeddings.weight = nn.Parameter(word_embeddings, requires_grad=True)
        self.lstm = nn.LSTM(embedding_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)
        # self.atten = AttentionLayer(self.hidden_size, self.attention_size, self.num_layers, self.num_directions)
        self.fc = nn.Linear(self.hidden_size * self.num_directions, num_classes)  # 2 for bidirection
        self.sigmoid = nn.Sigmoid()
        # self.softmax = nn.Softmax(dim=1)

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
        # in lightning, forward defines the prediction/inference actions
        # Set initial states
        h0 = torch.zeros(self.num_layers*2, input_ids.size(0), self.hidden_size, device=input_ids.device, requires_grad=True) # 2 for bidirection 
        c0 = torch.zeros(self.num_layers*2, input_ids.size(0), self.hidden_size, device=input_ids.device, requires_grad=True)
        
        batch = input_ids.shape[0]
        input_length = [input_ids.shape[1]] * batch
        pad_index = 1
        input_lengths = (input_ids != pad_index).sum(dim=1)

        x = self.embeddings(input_ids)

        # Forward propagate LSTM
        # print(x.shape)
        # packed_x = rnn.pack_padded_sequence(x, input_length, batch_first=True)
        packed_x = rnn.pack_padded_sequence(x, input_lengths.cpu(), batch_first=True, enforce_sorted=False)
        # out, (final_hidden_state, final_cell_state) = self.lstm(packed_x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        out, (final_hidden_state, final_cell_state) = self.lstm(packed_x)  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        # out, (final_hidden_state, final_cell_state) = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        
        # unpack
        # unpacked_out, unpacked_len = rnn.pad_packed_sequence(out, batch_first=True)
        unpacked_out, unpacked_len = rnn.pad_packed_sequence(out, batch_first=True, padding_value=1)

        # last_out = torch.zeros(x.shape[0], self.hidden_size*2, device=input_ids.device)
        # for each_data in range(len(unpacked_out)):
        #     last_out[each_data, :] = unpacked_out[each_data][-1, :]
        last_out = unpacked_out[range(len(unpacked_out)), input_lengths - 1]
        '''
        last_out = final_hidden_state.view(batch, self.num_layers, self.num_directions, self.hidden_size)
        last_out = last_out[:, 0, :, :].squeeze(1)
        last_out = last_out.view(batch, self.num_directions * self.hidden_size)
        '''
        # Decode the hidden state of the last time step
        # out = torch.sigmoid(self.fc(last_out))
        out = self.fc(last_out)
        return out

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        logits = self.forward(**x)
        loss = self.loss(logits, y.float())
        # Logging to TensorBoard by default
        self.log('train_loss', loss)
        print("======================loss", loss)
        
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
        optimizer = torch.optim.SGD(self.parameters(), lr=0.1)
        return optimizer

