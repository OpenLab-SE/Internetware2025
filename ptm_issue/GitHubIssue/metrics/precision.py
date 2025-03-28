import torch
from torchmetrics import Metric

class MultiLabelPrecision(Metric):
    def __init__(self, dist_sync_on_step=False, top_k: int = 1):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self._top_k = top_k
        self.add_state("correct", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        # Parameters
        preds : `torch.Tensor`, required.
            A tensor of preds of shape (batch_size, ..., num_classes).
        target : `torch.Tensor`, required.
            A tensor of integer class label of shape (batch_size, ...). It must be the same
            shape as the `preds` tensor without the `num_classes` dimension.
        """
        # preds, target = self.detach_tensors(preds, target)
        # preds, target = preds.cpu(), target.cpu()

        # Some sanity checks.
        num_classes = preds.size(-1)
        if target.dim() != preds.dim():
            raise ConfigurationError(
                "target must have dimension == preds.size() but "
                "found tensor of shape: {}".format(preds.size())
            )
        if (target >= num_classes).any():
            raise ConfigurationError(
                "A gold label passed to RecPrecision contains an id >= {}, "
                "the number of classes.".format(num_classes)
            )

        preds = preds.view(-1, num_classes)
        target = target.view(-1, num_classes).long()

        # Top K indexes of the preds (or fewer, if there aren't K of them).
        # Special case topk == 1, because it's common and .max() is much faster than .topk().
        if self._top_k == 1:
            top_k = preds.max(-1)[1].unsqueeze(-1)
        else:
            top_k = preds.topk(min(self._top_k, preds.shape[-1]), -1)[1]

        batch_size = preds.shape[0]
        preds_bin = torch.zeros(batch_size, num_classes, dtype=torch.long, device=target.device)
        for i in range(batch_size):
            preds_bin[i, top_k[i]] = 1

        correct_tensor = preds_bin & target

        # This is of shape (batch_size, ..., top_k).
        correct = torch.sum(correct_tensor, -1)
        
        self.correct += torch.sum(correct.float() / torch.full((batch_size,), self._top_k, dtype=torch.float32, device=target.device))
        self.total += batch_size

    def compute(self):
        if self.total > 1e-12:
            return self.correct.float() / self.total
        else:
            return torch.tensor(0, dtype=torch.float)