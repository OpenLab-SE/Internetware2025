import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2., reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha if alpha is not None else 1.0
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 输入 inputs 是您的模型的原始输出, 维度 (N, C) 其中 N 是 batch size 而 C 是类别数
        # targets 是实际标签, 维度 (N)
        if len(targets.shape) > 1 and targets.shape[1] > 1:
            targets = torch.argmax(targets, dim=1)

        # 计算 log_softmax，方便计算交叉熵损失
        log_probs = F.log_softmax(inputs, dim=-1)

        # 求出真实类别 targets 的 log_probs
        true_probs_log = log_probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)

        # 计算 Focal Loss
        focal_loss = -1 * (1 - true_probs_log.exp()).pow(self.gamma) * true_probs_log

        # 如果提供了 alpha，按类别进行加权
        if self.alpha is not None:
            if isinstance(self.alpha, (list, torch.Tensor)):
                # 使用 gather 来选择适当的 alpha
                alpha_factor = torch.tensor(self.alpha).to(inputs.device).gather(0, targets)
                print(f"alpha_factor is {alpha_factor}")
                focal_loss *= alpha_factor
            else:
                focal_loss *= self.alpha

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss