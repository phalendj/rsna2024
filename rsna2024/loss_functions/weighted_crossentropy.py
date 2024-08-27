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
    

class InstanceCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, device: str):
        super().__init__()
        self.device = device

    def on_epoch_end(self):
        pass

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        y = input['instance_labels']  # B, I, 2
        t = target['slice_classification'].to(self.device)  # B, I
        y = y.flatten(0,1)
        yt =t.flatten(0,1).long()
        return super().forward(y, yt)
    

class InstanceLevelCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, device: str):
        super().__init__()
        self.device = device

    def on_epoch_end(self):
        pass

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        y = input['instance_labels']  # Should be B, I, 10
        B, I, L = y.shape
        assert L == 10, f'yshape = {y.shape}'
        y = y.reshape(B, I, 5, 2).transpose(1, 2)  # Should be B, 5, I, 2
        t = target['level_slice_classification'].to(self.device)  # Should be B, 5, I
        assert t.shape[-2] == 5, f'tshape = {t.shape}, y.shape = {y.shape}'
        assert len(t.shape) == 3
        y = y.flatten(0,2)
        yt =t.flatten(0,2).long()
        return super().forward(y, yt)