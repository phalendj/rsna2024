import logging
from tqdm import tqdm
import math
from pathlib import Path
from collections import OrderedDict
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import get_cosine_schedule_with_warmup

from hydra.core.hydra_config import HydraConfig

try:
    from datasets import load_train_files, LEVELS
    from utils import relative_directory
    import utils as rsnautils
    from datasets import factory as dsfactory
    import loss_functions as lffactory
    import loss_functions.official as officialloss
    import models
except:
    from ..datasets import load_train_files, LEVELS
    from ..utils import relative_directory
    from .. import utils as rsnautils
    from ..datasets import factory as dsfactory
    from .. import loss_functions as lffactory
    from ..loss_functions import official as officialloss
    from .. import models


logger = logging.getLogger(__name__)


def create_optimizer(cfg, model, nbatches):
    result = {}
    epochs = cfg.training.epochs
    grad_acc = cfg.training.grad_acc
    opt_cfg = cfg.optimization
    if opt_cfg.optimizer == 'AdamW':
        optimizer = AdamW(model.parameters(), lr=opt_cfg.learning_rate, weight_decay=opt_cfg.weight_decay)
        result['optimizer'] = optimizer

    if opt_cfg.scheduler == 'CosineWithWarmup':
        warmup_steps = nbatches // grad_acc
        num_total_steps = epochs * nbatches // grad_acc

        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=warmup_steps,
                                                    num_training_steps=num_total_steps,
                                                    num_cycles=opt_cfg.n_cycles)
        result['scheduler'] = scheduler

    return result


    logger.info('Generate instance numbers file')
    model_directory = Path(HydraConfig.get().runtime.output_dir)
    df, dfc, dfd = load_train_files(relative_directory, clean=False)
    if cfg.clean:
        rsnautils.set_clean(False)
        df_clean, __, __ = load_train_files(relative_directory, clean=True)
        df_clean_i = df_clean.set_index('study_id')
    else:
        df_clean_i = df.set_index('study_id')

    device = 'cuda:0'

    all_models = []
    for fold in range(5):
        model = models.create_model(cfg.model, fold=fold)
        fname = model_directory / (model.name() + f'_fold{fold}.pth')
        logger.info(f'Loading model from {fname}')
        model.load_state_dict(torch.load(fname))
        __ = model.eval()
        all_models.append(model.to(device))

    results = []
    autocast = torch.autocast('cuda', enabled=cfg.training.use_amp, dtype=torch.half) # you can use with T4 gpu. or newer
    w = torch.arange(30)
    for fold in cfg.training.use_folds:
        logger.info(f'Fold {fold}')
        df_valid = df.loc[df.fold == fold]
        valid_ds = dsfactory.create_dataset(study_ids=df_valid.study_id.unique(), mode='valid', cfg=cfg.dataset)
        valid_dl = DataLoader(valid_ds,
                            batch_size=cfg.training.batch_size,
                            shuffle=False,
                            pin_memory=True,
                            drop_last=False,
                            num_workers=cfg.training.workers)
        
        with torch.no_grad():
            for x, t in tqdm(valid_dl):
                if isinstance(x, tuple) or isinstance(x, list):
                    x1, x2, x3 = x
                    x1 = x1.to(device)
                    x2 = x2.to(device)
                    x3 = x3.to(device)
                    with autocast:
                        preds = [model(x1, x2, x3) for model in all_models]
                else:
                    x = x.to(device)
                    with autocast:
                        preds = [model(x) for model in all_models]
                study_ids = t['study_id'][:, 0].numpy()
                series_ids = t['series_id'][:, 0].numpy()
                tmp = []
                for i, study_id in enumerate(study_ids):
                    if study_id in df_clean_i.index:
                        fold = df_clean_i.loc[study_id, 'fold']
                        pred = preds[fold]
                        class_pred = pred['instance_labels'].softmax(dim=2)[i, :, 1].cpu()
                    else:
                        class_pred = torch.mean(torch.stack([pred['instance_labels'].softmax(dim=2)[i, :, 1].cpu() for pred in preds], dim=0), dim=0)
                    tmp.append(class_pred)

                class_pred = torch.stack(tmp, dim=0)
                loc = (class_pred*w).sum(dim=1)/class_pred.sum(dim=1)
                ind = torch.round(loc.float()).long()
                instance_numbers = [t['instance_numbers'][i, j].item() for i, j in enumerate(ind)]

                for study_id, series_id, z in zip(study_ids, series_ids, instance_numbers):
                    for lev in LEVELS:
                        results.append({'study_id': study_id, 'series_id': series_id, 'instance_number': z, 'condition': 'Spinal Canal Stenosis', 'level': lev, 'x': 0, 'y': 0})
    pred_center_df = pd.DataFrame(results)
    pred_center_df.to_csv(model_directory / 'predicted_label_coordinates.csv', index=False)
    
def train_one_fold(model, cfg, fold: int):
    df, __, __ = load_train_files(relative_directory=relative_directory, clean=rsnautils.CLEAN)

    val_losses = []
    train_losses = []
    df_train = df.loc[df.fold != fold]
    df_valid = df.loc[df.fold == fold]

    logger.info(f'Fold {fold}: Length of Training {len(df_train)}, Length of Valid {len(df_valid)}')

    train_ds = dsfactory.create_dataset(study_ids=df_train.study_id.unique(), mode='train', cfg=cfg.dataset)
    valid_ds = dsfactory.create_dataset(study_ids=df_valid.study_id.unique(), mode='valid', cfg=cfg.dataset)

    device = 'cuda:0'

    train_dl = DataLoader(
                            train_ds,
                            batch_size=cfg.training.batch_size,
                            shuffle=True,
                            pin_memory=True,
                            drop_last=True,
                            num_workers=cfg.training.workers
                            )
    
    valid_dl = DataLoader(
                            valid_ds,
                            batch_size=cfg.training.batch_size,
                            shuffle=False,
                            pin_memory=True,
                            drop_last=False,
                            num_workers=cfg.training.workers
                            )

    criterion = lffactory.create_loss(cfg=cfg.loss, device=device)

    best_loss = 1.e6
    es_step = 0

    optimization_details = create_optimizer(cfg=cfg, model=model, nbatches=len(train_dl))
    optimizer = optimization_details['optimizer']
    scheduler = optimization_details['scheduler']

    autocast = torch.autocast('cuda', enabled=cfg.training.use_amp, dtype=torch.half) # you can use with T4 gpu. or newer
    scaler = torch.GradScaler('cuda', enabled=cfg.training.use_amp, init_scale=4096)

    output_directory = Path(HydraConfig.get().runtime.output_dir)

    fname = output_directory / (model.name() + f'_fold{fold}.pth')

    GRAD_ACC = cfg.training.grad_acc
    MAX_GRAD_NORM = None

    model.to(device)

    EPOCHS = cfg.training.epochs
    for epoch in range(1, EPOCHS+1):
        logger.info(f'start epoch {epoch}')
        model.train()
        total_loss = 0
        with tqdm(train_dl, leave=True) as pbar:
            optimizer.zero_grad()
            for idx, (x, t) in enumerate(pbar):  
                if isinstance(x, dict):
                    x = {k: v.to(device) for k, v in x.items()}
                    with autocast:
                        y = model(x)
                        loss = criterion(y, t)
                        
                        total_loss += loss.item()
                        if GRAD_ACC > 1:
                            loss = loss / GRAD_ACC

                elif isinstance(x, tuple) or isinstance(x, list):
                    x1, x2, x3 = x
                    x1 = x1.to(device)
                    x2 = x2.to(device)
                    x3 = x3.to(device)
                    with autocast:
                        y = model(x1, x2, x3)
                        loss = criterion(y, t)
                        total_loss += loss.item()   
                        if GRAD_ACC > 1:
                            loss = loss / GRAD_ACC
                else:
                    x = x.to(device)
                    with autocast:
                        y = model(x)
                        loss = criterion(y, t)
                        
                        total_loss += loss.item()
                        if GRAD_ACC > 1:
                            loss = loss / GRAD_ACC

                if not math.isfinite(loss):
                    logger.info(f"Loss is {loss}, stopping training")
                    return

                pbar.set_postfix(
                    OrderedDict(
                        loss=f'{loss.item()*GRAD_ACC:.6f}',
                        lr=f'{optimizer.param_groups[0]["lr"]:.3e}'
                    )
                )
                scaler.scale(loss).backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM or 1e9)

                if (idx + 1) % GRAD_ACC == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    if scheduler is not None:
                        scheduler.step()                    

        train_loss = total_loss/len(train_dl)
        logger.info(f'train_loss:{train_loss:.6f}')
        train_losses.append(train_loss)
        total_loss = 0

        model.eval()
        with tqdm(valid_dl, leave=True) as pbar:
            with torch.no_grad():
                for idx, (x, t) in enumerate(pbar):
                    if isinstance(x, dict):
                        x = {k: v.to(device) for k, v in x.items()}
                        with autocast:
                            y = model(x)
                            loss = criterion(y, t)
                            
                            total_loss += loss.item()
                            if GRAD_ACC > 1:
                                loss = loss / GRAD_ACC

                    elif isinstance(x, tuple) or isinstance(x, list):
                        x1, x2, x3 = x
                        x1 = x1.to(device)
                        x2 = x2.to(device)
                        x3 = x3.to(device)
                        with autocast:
                            y = model(x1, x2, x3)
                            loss = criterion(y, t)
                            total_loss += loss.item()   
                    else:
                        x = x.to(device)
                        with autocast:
                            y = model(x)
                            loss = criterion(y, t)
                            total_loss += loss.item()   

        val_loss = total_loss/len(valid_dl)
        logger.info(f'val_loss:{val_loss:.6f}')
        val_losses.append(val_loss)
        if val_loss < best_loss:

            if device!='cuda:0':
                model.to('cuda:0')                

            logger.info(f'epoch:{epoch}, best weighted_logloss updated from {best_loss:.6f} to {val_loss:.6f}')
            best_loss = val_loss
            torch.save(model.state_dict(), fname)
            logger.info(f'{fname} is saved')
            es_step = 0

            if device!='cuda:0':
                model.to(device)

        else:
            es_step += 1
            if es_step >= cfg.training.early_stopping:
                logger.info('early stopping')
                break  

        criterion.on_epoch_end()

