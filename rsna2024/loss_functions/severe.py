"""
From

https://www.kaggle.com/code/junkoda/optimize-the-evaluation-metric/notebook

"""

import pandas as pd
import numpy as np
from typing import Sequence

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


# Custom Loss for this competition

class SevereLoss(_Loss):
    """
    For RSNA 2024
    criterion = SevereLoss()     # you can replace nn.CrossEntropyLoss
    loss = criterion(y_pred, y)
    """
    def __init__(self, device: str, temperature: float = 1.0, weights: Sequence[float] = np.array([6.084050632911392, 12.962531645569621, 14.38632911392405, 1.729113924050633])):
        """
        Use max if temperature = 0
        """
        super().__init__()
        self.t = temperature
        self.weights = weights
        self.device = device
        assert self.t >= 0
    
    def __repr__(self):
        return 'SevereLoss(t=%.1f)' % self.t

    def on_epoch_end(self):
        pass

    def forward(self, input: dict, target: dict) -> torch.Tensor:
        """
        Args:
          y_pred (Tensor[float]): logit             (batch_size, 3, 25)
          y      (Tensor[int]):   true label index  (batch_size, 25)
        """
        B = len(input['labels'])
        y_pred = input['labels'].reshape(B, 25, 3)
        y_pred.transpose(2, 1)
        y = target['labels'].to(self.device)

        assert y_pred.size(0) == y.size(0)
        assert y_pred.size(1) == 3 and y_pred.size(2) == 25
        assert y.size(1) == 25
        assert y.size(0) > 0
        
        slices = [slice(0, 5), slice(5, 15), slice(15, 25)] 
        w = 2 ** y  # sample_weight w = (1, 2, 4) for y = 0, 1, 2 (batch_size, 25)

        loss = F.cross_entropy(y_pred, y, reduction='none')  # (batch_size, 25)

        # Weighted sum of losses for spinal (:5), foraminal (5:15), and subarticular (15:25)
        wloss_sums = []
        for k, idx in enumerate(slices):
            wloss_sums.append((w[:, idx] * loss[:, idx]).sum())

        # any_severe_spinal
        #   True label y_max:      Is any of 5 spinal severe? true/false
        #   Prediction y_pred_max: Max of 5 spinal severe probabilities y_pred[:, 2, :5].max(dim=1)
        #   any_severe_spinal_loss is the binary cross entropy between these two.
        y_spinal_prob = y_pred[:, :, :5].softmax(dim=1)             # (batch_size, 3,  5)
        w_max = torch.amax(w[:, :5], dim=1)                         # (batch_size, )
        y_max = torch.amax(y[:, :5] == 2, dim=1).to(torch.float32)  # 0 or 1

        if self.t > 0:
            # Attention for the maximum value
            attn = F.softmax(y_spinal_prob[:, 2, :] / self.t, dim=1)  # (batch_size, 5)

            # Pick the sofmax among 5 severe=2 y_spinal_probs with attn
            y_pred_max = (attn * y_spinal_prob[:, 2, :]).sum(dim=1)   # weighted average among 5 spinal columns 
        else:
            # Exact max; this works too
            y_pred_max = y_spinal_prob[:, 2, :].amax(dim=1)

        loss_max = F.binary_cross_entropy(y_pred_max, y_max, reduction='none')
        wloss_sums.append((w_max * loss_max).sum())

        # See below about these numbers
        loss = (wloss_sums[0] / self.weights[0] +
                wloss_sums[1] / self.weights[1] + 
                wloss_sums[2] / self.weights[2] +
                wloss_sums[3] / self.weights[3]) / (4 * y.size(0))

        return loss
    

# Compute the weight global average
def compute_weight_global_average(train_filename: str) -> np.array:
    train = pd.read_csv(train_filename)
    weight_map = {'Normal/Mild': 1,
                'Moderate': 2,
                'Severe': 4,
                None: 0}

    w_sum = [0, ] * 4

    for r in train.iter_rows():
        w = np.array([weight_map[x] for x in r[1:]])  # array[int] (25, )
        assert len(w) == 25

        w_sum[0] += w[:5].sum()    # spinal
        w_sum[1] += w[5:15].sum()  # foraminal
        w_sum[2] += w[15:25].sum() # subarticular 
        w_sum[3] += w[:5].max()    # any_severe_spinal

    for k in range(4):
        w_sum[k] /= len(train)

    return w_sum
    # (6.084050632911392, 12.962531645569621, 14.38632911392405, 1.729113924050633)