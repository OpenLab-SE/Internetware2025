import torch
from sklearn.metrics import f1_score
from torchmetrics import Metric


class SelectedClassesF1Score(Metric):
    def __init__(self, selected_classes, average='macro', dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.selected_classes = selected_classes
        self.average = average
        # 初始化用于保存预测和真实标签的缓冲区
        self.add_state("preds", default=[], dist_reduce_fx=None)
        self.add_state("targets", default=[], dist_reduce_fx=None)

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        # preds 和 targets 应当是在每个 validation step 产生的预测和标签

        # 预测通常是 logits 或者 softmax 概率
        # 通过 argmax 获得最可能的类别
        preds = torch.argmax(preds, dim=1)
        # print(preds)
        # 将 one-hot 编码的真实标签转换为整数标签
        if len(targets.shape) > 1 and targets.shape[1] > 1:
            targets = torch.argmax(targets, dim=1)

        # 转换到 CPU 并转换为 list 后存储
        self.preds.append(preds.cpu().tolist())
        self.targets.append(targets.cpu().tolist())

    def compute(self):
        # 将所有的预测和目标拼接在一起
        all_preds = [item for sublist in self.preds for item in sublist]
        all_targets = [item for sublist in self.targets for item in sublist]

        # print("===============Pred vs True label===================")
        # print(f"pred is {all_preds}")
        # print(f"true is {all_targets}")
        # print("===============Pred vs True label===================")

        # 使用 sklearn 的 f1_score 计算只关注特定类别的 F1 分数
        selected_f1 = f1_score(
            all_targets, all_preds, labels=self.selected_classes, average=self.average
        )
        return selected_f1
