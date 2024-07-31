from typing import Sequence
import logging
import torch
import torch.nn as nn

import heatmap
import weighted_crossentropy


logger = logging.getLogger(__name__)


class MixedLoss(nn.Module):
    def __init__(self, device: str, center_change: float, width_change:float, patch_size: int, weights: Sequence[float] = [1.0, 2.0, 4.0]):
        super().__init__()
        self.device = device
        self.center_change = center_change
        self.width_change = width_change
        self.heatmap_loss = heatmap.HeatmapLoss(device=device, patch_size=patch_size)
        self.wce_loss = weighted_crossentropy.WeightedCrossEntropyLoss(device=device, weights=weights)

        self.heatmap_weight = torch.tensor([1.0]).to(device)
        self.epoch = 0

    def on_epoch_end(self):
        self.epoch += 1
        # Fermi Function turn on
        self.heatmap_weight = torch.tensor([1.0/(1.0+torch.exp(self.epoch - self.center_change)/self.width_change)]).to(self.device)

        logger.info(f'Adjusting Heatmap weight for {self.epoch} to {self.heatmap_weight.cpu().item()}')


    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.heatmap_loss(input, target) * self.heatmap_weight + self.wce_loss(input, target)*(1.0-self.heatmap_weight)
        return loss
