import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn

# 假设你的标签是整数形式
labels = [0, 1, 2, 3]  # 类别标签
y_train = [0, 1, 1, 1, 2, 3, 3, 3, 3, 3]  # 训练数据中的标签

# 计算类别权重
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=y_train)
class_weights = torch.tensor(class_weights, dtype=torch.float)
print(f'class_weights: {class_weights}')
# 创建损失函数并应用权重
criterion = nn.CrossEntropyLoss(weight=class_weights)