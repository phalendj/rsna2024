import loss_functions.severe as severe
import loss_functions.weighted_crossentropy as weighted_crossentropy


def create_loss(cfg, device):
    if cfg.name == 'WeightedCrossEntropy':
        return weighted_crossentropy.WeightedCrossEntropyLoss(device=device)
    elif cfg.name == 'SevereLoss':
        return severe.SevereLoss(device=device, temperature=0.0)
    else:
        raise NotImplementedError