import datasets.single_description as single_description



def create_dataset(study_ids, mode, cfg):
    if cfg.name == 'SpinalCanalStenosisCenterDataset':
        return single_description.SpinalCanalStenosisCenterDataset(study_ids=study_ids,
                                                                   image_size=cfg.image_size,
                                                                   channels=cfg.channels,
                                                                   mode=mode, 
                                                                   transform=None
                                                                   )
    else:
        raise NotImplementedError