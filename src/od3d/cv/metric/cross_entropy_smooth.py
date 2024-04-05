
import torch
from od3d.cv.select import batched_index_select

class CrossEntropyLabelsSmoothed:
    def __init__(self, labels_smoothed, reduction='mean'):
        self.labels_smoothed = labels_smoothed  # C x C
        self.reduction = reduction

    def __call__(self, logits, labels):
        """Calculate cross entropy loss, apply label smoothing if needed.
        Args:
            logits: (BxC)
            labels: (Bx1)
        Returns:
            loss: (1,)
        """

        target_probs = batched_index_select(input=self.labels_smoothed, dim=0, index=labels)

        # BxC, BxC
        loss = (-target_probs * torch.log_softmax(logits, dim=1)).sum(dim=1) / target_probs.sum(dim=1)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss