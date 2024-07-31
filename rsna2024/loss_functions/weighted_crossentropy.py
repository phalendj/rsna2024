import torch.nn as nn
from typing import Sequence
import torch

class WeightedCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, device: str, weights: Sequence[float] = [1.0, 2.0, 4.0]):
        super().__init__(weight=torch.tensor(weights).to(device))
        self.device = device

    def on_epoch_end(self):
        pass

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = 0
        y = input['labels']
        t = target['labels'].to(self.device)
        N_LABELS = t.shape[-1]
        for col in range(N_LABELS):
            pred = y[:,col*3:col*3+3]
            gt = t[:, col]
            loss = loss + super().forward(pred, gt) / N_LABELS

        return loss