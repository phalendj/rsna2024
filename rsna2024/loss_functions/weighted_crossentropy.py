import torch.nn as nn
from typing import Sequence
import torch

class WeightedCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, device: str, weights: Sequence[float] = [1.0, 2.0, 4.0]):
        super().__init__(weight=torch.tensor(weights).to(device))