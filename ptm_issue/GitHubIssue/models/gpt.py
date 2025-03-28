import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from torch import nn
from torch.utils.data import DataLoader, random_split
from transformers import GPT2ForSequenceClassification

from ..metrics.accuracy import MultiLabelAccuracy
from ..metrics.precision import MultiLabelPrecision
from ..metrics.recall import MultiLabelRecall

SEQUENCE_MODEL_CONFIG = {
    "gpt2": GPT2ForSequenceClassification, # https://huggingface.co/gpt2
    "gpt2-medium": GPT2ForSequenceClassification,
    "gpt2-large": GPT2ForSequenceClassification,
    "gpt2-xl": GPT2ForSequenceClassification,
    "microsoft/CodeGPT-small-py": GPT2ForSequenceClassification,
    "CodeGPT-small-py": GPT2ForSequenceClassification,
}


class Gpt(pl.LightningModule):
    def __init__(self, num_classes: int, base_lr: float=5e-5, model_name: str='gpt2', use_sequence: bool=False, disablefinetune: bool=False, local_model: bool=False):
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
            # Fix AssertionError: Cannot handle batch sizes > 1 if no padding token is defined.
            self.model.config.pad_token_id = self.model.config.eos_token_id 

            if disablefinetune:  # disable finetune for sequence model
                for name, param in self.model.named_parameters():
                    if 'classifier' in name: # classifier layer for bert base model
                        param.requires_grad = True
                    elif 'classification_head' not in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False


        self.hid_dim = self.model.config.hidden_size
        
        self.loss = nn.CrossEntropyLoss()
        
        # self.class_weights = torch.tensor([1.2, 1.2, 1.2, 0.8, 0.8]).to("cuda:0") # 
        # self.class_weights = torch.tensor([1.5, 1.5, 1.5, 0.75, 0.75]).to("cuda:0") # 
        # self.class_weights = torch.tensor([2.0, 2.0, 2.0, 0.75, 0.75]).to("cuda:0") # 
        # self.class_weights = torch.tensor([1.5, 3.0, 1.5, 1.0, 0.5]).to("cuda:0") # work for openpose
        # self.class_weights = torch.tensor([1.5, 1.5, 3.0, 1.0, 0.5]).to("cuda:0") # work for cyclegan
        # self.class_weights = torch.tensor([3.0, 4.0, 3.0, 1.0, 0.5]).to("cuda:0") # work for realtime
        # self.class_weights = torch.tensor([1.5, 1.5, 2.0, 0.75, 0.75]).to("cuda:0") # work for faceswap
        # self.class_weights = torch.tensor([1.25, 1.25, 1.5, 0.75, 0.75]).to("cuda:0") # work for faceswap
        # self.class_weights = torch.tensor([1.25, 1.25, 1.5, 1.0, 0.75]).to("cuda:0") # work for faceswap
        # self.class_weights = torch.tensor([1.5, 1.5, 1.5, 0.75, 1.0]).to("cuda:0") # work for deepfacelab
        # self.class_weights = torch.tensor([1.25, 1.5, 1.25, 0.75, 1.0]).to("cuda:0") # work for deepfacelab
        # self.loss = nn.CrossEntropyLoss(weight=self.class_weights)


        self.metrics = nn.ModuleDict()
        stage_name = ['train', 'valid', 'test']
        for stage in stage_name:
            for k in range(1, 3):
                self.metrics[f"{stage}_acc_{k}"] = MultiLabelAccuracy(top_k=k)
                self.metrics[f"{stage}_precision_{k}"] = torchmetrics.Precision(top_k=k)
                self.metrics[f"{stage}_recall_{k}"] = torchmetrics.Recall(top_k=k)
                self.metrics[f"{stage}_f1_marco_{k}"] = torchmetrics.F1(average='macro', num_classes=num_classes, top_k=k)
                self.metrics[f"{stage}_f1_marco_weight_{k}"] = torchmetrics.F1(average='weighted', num_classes=num_classes, top_k=k)
                self.metrics[f"{stage}_f1_mirco_{k}"] = torchmetrics.F1(average='micro', num_classes=num_classes, top_k=k)
        
        self.save_hyperparameters()

    def forward(self, input_ids):
        # in lightning, forward defines the prediction/inference actions
        # with torch.no_grad():
        # output = self.model(input_ids)
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
        # with torch.no_grad():
        #     print(torch.concat([torch.nn.functional.softmax(logits, dim=-1), y], dim=-1))
        loss = self.loss(logits, y.float())
        # Logging to TensorBoard by default
        self.log('train_loss', loss)
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

    def validation_epoch_end(self, outs):
        for name, metric in self.metrics.items():
            if name.startswith('valid_'):
                self.log(f"{name}_epoch", metric.compute())
                metric.reset()

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
        
        if self.use_sequence and self.model_name in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "CodeGPT-small-py", "microsoft/CodeGPT-small-py"]:
            # Do not modify
            num_decoder_layers = 12

            # Modify to control finetune layer
            decoder_layers_to_train = 4
            finetune_emb = True
            finetune_ln = True
            finetune_cls = True

            # 确保要微调的层数不超过实际层数
            decoder_layers_to_train = min(decoder_layers_to_train, num_decoder_layers)

            # 微调共享权重， 默认为True
            for param in self.model.transformer.wte.parameters():
                param.requires_grad = finetune_emb
            for param in self.model.transformer.wpe.parameters():
                param.requires_grad = finetune_emb
            
            # 微调解码器的最后decoder_layers_to_train层
            for i, layer in enumerate(self.model.transformer.h):
                for param in layer.parameters():
                    param.requires_grad = i >= (num_decoder_layers - decoder_layers_to_train)

            # 微调 layer norm 层 默认为True
            for param in self.model.transformer.ln_f.parameters():
                param.requires_grad = finetune_ln

            # 微调分类层, 默认为True
            for param in self.model.score.parameters():
                param.requires_grad = finetune_cls
            

            emb_params = list(self.model.transformer.wte.parameters()) + list(self.model.transformer.wpe.parameters())
            decoder_trainable_params = list(filter(lambda p: p.requires_grad, self.model.transformer.h.parameters()))
            ln_params = list(self.model.transformer.ln_f.parameters())
            fc_params = list(self.model.score.parameters())

            base_lr = self.base_lr
            emb_lr = base_lr
            decoder_lr = base_lr
            ln_lr = base_lr
            fc_lr = base_lr

            optimizer = torch.optim.AdamW([
                {'params': emb_params, 'lr': emb_lr, 'weight_decay':5e-2},
                {'params': decoder_trainable_params, 'lr': decoder_lr, 'weight_decay':5e-2},
                {'params': ln_params, 'lr': ln_lr, 'weight_decay':5e-2},
                {'params': fc_params, 'lr': fc_lr, 'weight_decay':5e-2},
            ])

        elif self.use_sequence and self.model_name in ["t5-base", "t5-large"]:
            # Do not modify
            num_encoder_layers = 12
            num_decoder_layers = 12

            # Modify to control finetune layer
            encoder_layers_to_train = 12
            decoder_layers_to_train = 12
            finetune_emb = True
            finetune_cls = True

            # 获取编码器和解码器的所有层
            encoder_layers = len(self.model.bert.transformer.encoder)
            decoder_layers = len(self.model.bert.transformer.decoder)
            
            # 确保要微调的层数不超过实际层数
            encoder_layers_to_train = min(encoder_layers_to_train, num_encoder_layers)
            decoder_layers_to_train = min(decoder_layers_to_train, num_decoder_layers)

            # 微调共享权重， 默认为True
            for params in self.model.bert.transformer.shared.parameters():
                param.requires_grad = finetune_emb
            
            # 微调编码器的最后encoder_layers_to_train层
            for i, layer in enumerate(self.model.bert.transformer.encoder):
                for param in layer.parameters():
                    param.requires_grad = i >= (num_encoder_layers - encoder_layers_to_train)
                    
            # 微调解码器的最后decoder_layers_to_train层
            for i, layer in enumerate(decoder_layers):
                for param in layer.parameters():
                    param.requires_grad = i >= (num_decoder_layers - decoder_layers_to_train)

            # 微调分类层, 默认为True
            for params in self.model.bert.classification_head.parameters():
                param.requires_grad = finetune_cls

            emb_params = self.model.bert.transformer.shared.parameters()
            encoder_trainable_params = list(filter(lambda p: p.requires_grad, self.model.bert.transformer.encoder.parameters()))
            decoder_trainable_params = list(filter(lambda p: p.requires_grad, self.model.bert.transformer.decoder.parameters()))
            fc_params = list(self.model.bert.classification_head.parameters())

            base_lr = self.base_lr
            emb_lr = base_lr
            encoder_lr = base_lr
            decoder_lr = base_lr
            fc_lr = base_lr

            optimizer = torch.optim.AdamW([
                {'params': emb_params, 'lr': emb_lr, 'weight_decay':5e-2},
                {'params': encoder_trainable_params, 'lr': encoder_lr, 'weight_decay':5e-2},
                {'params': decoder_trainable_params, 'lr': decoder_lr, 'weight_decay':5e-2},
                {'params': fc_params, 'lr': fc_lr, 'weight_decay':5e-2},
            ])
        else:
            # optimizer = torch.optim.Adam(self.parameters(), lr=2e-5)
            optimizer = torch.optim.AdamW(self.parameters(), lr=2e-5, weight_decay=5e-2)
        return optimizer
    