from . import segmentation_single_image as segmentationsingle
from . import augmentations as aug
from . import level_cubes as level_cubes

def create_dataset(study_ids, mode, cfg):
    if cfg.name == 'SegmentationSingleImageDataset':
        cfg_aug = cfg.augmentations
        transform = aug.get_transform(train=(mode=='train'), cfg=cfg_aug)
        return segmentationsingle.SegmentationCenterDataset(study_ids=study_ids,
                                                                 image_size=cfg.image_size,
                                                                 conditions=cfg.conditions,
                                                                 channels=1,
                                                                 series_description=cfg.series_description,
                                                                 mode=mode, 
                                                                 aug_size=cfg.aug_size,
                                                                 transform=transform
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
                                                                 aug_size=cfg.aug_size,
                                                                 transform=transform,
                                                            )
    if cfg.name == 'SegmentationPredictedCenterDataset':
        cfg_aug = cfg.augmentations
        transform = aug.get_transform(train=(mode=='train'), cfg=cfg_aug)
        return segmentationsingle.SegmentationPredictedCenterDataset(study_ids=study_ids,
                                                                 image_size=cfg.image_size,
                                                                 channels=cfg.channels,
                                                                 conditions=cfg.conditions,
                                                                 series_description=cfg.series_description,
                                                                 generated_coordinate_file=cfg.center_file,
                                                                 mode=mode, 
                                                                 aug_size=cfg.aug_size,
                                                                 transform=transform,
                                                            )
    if cfg.name == 'LevelCubeDataset':
        cfg_aug = cfg.augmentations
        transform = aug.get_transform(train=(mode=='train'), cfg=cfg_aug)
        assert len(cfg.conditions) == 1, 'Only one condition for level cubes'
        return level_cubes.LevelCubeDataset(study_ids=study_ids,
                                            channels=cfg.channels,
                                            patch_size=cfg.subsize,
                                            condition=cfg.conditions[0],
                                            generated_coordinate_file=cfg.center_file,
                                            series_description=cfg.series_description,
                                            mode=mode, 
                                            transform=transform)
    if cfg.name == 'LevelCubeLeftRightDataset':
        cfg_aug = cfg.augmentations
        transform = aug.get_transform(train=(mode=='train'), cfg=cfg_aug)
        assert len(cfg.conditions) == 2
        assert ' '.join(cfg.conditions[0].split()[1:]) == ' '.join(cfg.conditions[1].split()[1:]), "Must be same type of condition"
        assert cfg.conditions[0].split()[0] == 'Left', "Left condition must be first"
        assert cfg.conditions[1].split()[0] == 'Right', "Right condition must be last"
        return level_cubes.LevelCubeLeftRightDataset(study_ids=study_ids,
                                                    channels=cfg.channels,
                                                    patch_size=cfg.subsize,
                                                    left_condition=cfg.conditions[0],
                                                    right_condition=cfg.conditions[1],
                                                    generated_coordinate_file=cfg.center_file,
                                                    series_description=cfg.series_description,
                                                    mode=mode, 
                                                    transform=transform)
    
    if cfg.name == 'AllLevelCubeDataset':
        cfg_aug = cfg.augmentations
        transform = aug.get_transform(train=(mode=='train'), cfg=cfg_aug)
        assert len(cfg.conditions) == 1, 'Only one condition for level cubes'
        return level_cubes.AllLevelCubeDataset(study_ids=study_ids,
                                            sagittal_t2_channels=cfg.sagittal_t2_channels,
                                            sagittal_t2_patch_size=cfg.sagittal_t2_subsize,
                                            sagittal_t1_channels=cfg.sagittal_t1_channels,
                                            sagittal_t1_patch_size=cfg.sagittal_t1_subsize,
                                            axial_t2_channels=cfg.axial_t2_channels,
                                            axial_t2_patch_size=cfg.axial_t2_subsize,
                                            condition=cfg.conditions[0],
                                            generated_coordinate_file=cfg.center_file,
                                            series_description=cfg.series_description,
                                            mode=mode, 
                                            transform=transform)
    if cfg.name == 'AllLevelCubeLeftRightDataset':
        cfg_aug = cfg.augmentations
        transform = aug.get_transform(train=(mode=='train'), cfg=cfg_aug)
        assert len(cfg.conditions) == 2
        assert ' '.join(cfg.conditions[0].split()[1:]) == ' '.join(cfg.conditions[1].split()[1:]), "Must be same type of condition"
        assert cfg.conditions[0].split()[0] == 'Left', "Left condition must be first"
        assert cfg.conditions[1].split()[0] == 'Right', "Right condition must be last"
        return level_cubes.AllLevelCubeLeftRightDataset(study_ids=study_ids,
                                                    sagittal_t2_channels=cfg.sagittal_t2_channels,
                                                    sagittal_t2_patch_size=cfg.sagittal_t2_subsize,
                                                    sagittal_t1_channels=cfg.sagittal_t1_channels,
                                                    sagittal_t1_patch_size=cfg.sagittal_t1_subsize,
                                                    axial_t2_channels=cfg.axial_t2_channels,
                                                    axial_t2_patch_size=cfg.axial_t2_subsize,
                                                    left_condition=cfg.conditions[0],
                                                    right_condition=cfg.conditions[1],
                                                    generated_coordinate_file=cfg.center_file,
                                                    series_description=cfg.series_description,
                                                    mode=mode, 
                                                    transform=transform)
    # if cfg.name == 'LevelCubeCropZoomDataset':
    #     cfg_aug = cfg.augmentations
    #     transform = aug.get_transform(train=(mode=='train'), cfg=cfg_aug)
    #     return level_cubes.LevelCubeCropZoomDataset(study_ids=study_ids,
    #                                         image_size=cfg.image_size,
    #                                         channels=cfg.channels,
    #                                         slice_size=cfg.subsize,
    #                                         conditions=cfg.conditions,
    #                                         series_description=cfg.series_description,
    #                                         mode=mode, 
    #                                         transform=transform)
    
    # if cfg.name == 'AllLevelCubeDataset':
    #     cfg_aug = cfg.augmentations
    #     transform = aug.get_transform(train=(mode=='train'), cfg=cfg_aug)
    #     return level_cubes.AllLevelCubeDataset(study_ids=study_ids,
    #                                            channels=cfg.channels,
    #                                            slice_size=cfg.subsize,
    #                                            conditions=cfg.conditions,
    #                                            mode=mode, 
    #                                            transform=transform)

    else:
        raise NotImplementedError