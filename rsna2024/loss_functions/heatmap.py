import torch.nn as nn
import math
import torch

class HeatmapLoss(nn.Module):
    def __init__(self, device: str, patch_size):
        super().__init__()
        self.device = device
        self.patch_size = patch_size
        idx_map = torch.stack([torch.arange(self.patch_size)]*self.patch_size).to(device)
        self.idx_map = torch.stack([idx_map,idx_map.T]).view(1,1,2,self.patch_size,self.patch_size)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = 0
        mask_pred = input['masks']  # shape B, CH, H, W
        mask_true = target['centers'].to(self.device)
        B, CH, H, W = mask_pred.shape
        assert H == W
        assert H == self.patch_size
        s2 = torch.as_tensor([H/8]*CH)
        A = -1/(2*s2).to(self.device)
        K = 1/torch.sqrt(2*math.pi*s2).to(self.device)
        mask_pred = mask_pred*K.view(1,5,1,1)
        mask = self.idx_map - mask_true.view(-1,5,2,1,1)
        mask = torch.exp((A.view(-1,5,1,1,1)*mask*mask).sum(2))*K.view(-1,5,1,1)

        D = 1 - ((mask*mask_pred).sum())**2/((mask*mask).sum()*(mask_pred*mask_pred).sum())
        
        return D