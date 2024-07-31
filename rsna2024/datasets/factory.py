import datasets.single_description as single_description
import datasets.augmentations as aug


def create_dataset(study_ids, mode, cfg):
    if cfg.name == 'SpinalCanalStenosisCenterDataset':
        cfg_aug = cfg.augmentations
        transform = aug.get_transform(train=(mode=='train'), cfg=cfg_aug)
        return single_description.SpinalCanalStenosisCenterDataset(study_ids=study_ids,
                                                                   image_size=cfg.image_size,
                                                                   channels=cfg.channels,
                                                                   mode=mode, 
                                                                   transform=transform
                                                                   )
    else:
        raise NotImplementedError