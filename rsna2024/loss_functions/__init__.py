from . import severe as severe
from . import heatmap as heatmap
from . import mixed as mixedloss
from . import weighted_crossentropy as weighted_crossentropy


def create_loss(cfg, device):
    if cfg.name == 'WeightedCrossEntropy':
        return weighted_crossentropy.WeightedCrossEntropyLoss(device=device)
    elif cfg.name == 'SevereLoss':
        return severe.SevereLoss(device=device, temperature=0.0)
    elif cfg.name == 'HeatmapLoss':
        return heatmap.HeatmapLoss(device=device, patch_size=cfg.patch_size)
    elif cfg.name == 'MixedLoss':
        return mixedloss.MixedLoss(device=device, center_change=cfg.center_change, width_change=cfg.width_change, patch_size=cfg.patch_size)
    else:
        raise NotImplementedError