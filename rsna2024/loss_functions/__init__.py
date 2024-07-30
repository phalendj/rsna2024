import severe
import weighted_crossentropy


def create_loss(cfg, device):
    if cfg.name == 'WeightedCrossEntropy':
        return weighted_crossentropy.WeightedCrossEntropyLoss(device=device)
    elif cfg.name == 'SevereLoss':
        return severe.SevereLoss(temperature=0.0)
    else:
        raise NotImplementedError