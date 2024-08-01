from . import single_description as single_description
from . import segmentation_single_image as segmentationsingle
from . import augmentations as aug


def create_dataset(study_ids, mode, cfg):
    if cfg.name == 'SingleSeriesCenterDataset':
        cfg_aug = cfg.augmentations
        transform = aug.get_transform(train=(mode=='train'), cfg=cfg_aug)
        return single_description.SingleSeriesCenterDataset(study_ids=study_ids,
                                                            image_size=cfg.image_size,
                                                            channels=cfg.channels,
                                                            conditions=cfg.conditions,
                                                            series_description=cfg.series_description,
                                                            mode=mode, 
                                                            transform=transform
                                                            )
    if cfg.name == 'SegmentationSingleImageDataset':
        return segmentationsingle.SegmentationSingleImageDataset(study_ids=study_ids,
                                                                 image_size=cfg.image_size,
                                                                 conditions=cfg.conditions,
                                                                 series_description=cfg.series_description,
                                                                 mode=mode, 
                                                                 aug_size=cfg.aug_size
                                                            )
    
    if cfg.name == 'SegmentationCenterDataset':
        cfg_aug = cfg.augmentations
        transform = aug.get_transform(train=(mode=='train'), cfg=cfg_aug)
        return segmentationsingle.SegmentationCenterDataset(study_ids=study_ids,
                                                                 image_size=cfg.image_size,
                                                                 channels=cfg.channels,
                                                                 conditions=cfg.conditions,
                                                                 series_description=cfg.series_description,
                                                                 mode=mode, 
                                                                 aug_size=cfg.aug_size
                                                            )

    else:
        raise NotImplementedError