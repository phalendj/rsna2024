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
    from datasets import load_train_files
    from utils import relative_directory
    from datasets import factory as dsfactory
    import loss_functions as lffactory
    import loss_functions.official as officialloss
except:
    from ..datasets import load_train_files
    from ..utils import relative_directory
    from ..datasets import factory as dsfactory
    from .. import loss_functions as lffactory
    from ..loss_functions import official as officialloss


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


def evaluate(model, cfg):
    df, __, __ = load_train_files(relative_directory=relative_directory)
    # load sample submission file
    device = 'cuda:0'

    model_predictions = []
    for fold in cfg.training.use_folds:
        logger.info(f'Evaluating Fold {fold}')
        try:
            output_directory = Path(HydraConfig.get().runtime.output_dir)
        except ValueError:
            output_directory = Path(cfg.load_directory)
        fname = output_directory / (model.name() + f'_fold{fold}.pth')
        logger.info(f'Loading Model from {fname}')
        model.load_state_dict(torch.load(fname))
        model.to(device)
        model.eval()

        df_valid = df.loc[df.fold == fold]
        valid_ds = dsfactory.create_dataset(study_ids=df_valid.study_id.unique(), mode='valid', cfg=cfg.dataset)
        valid_dl = DataLoader(
                            valid_ds,
                            batch_size=cfg.training.batch_size,
                            shuffle=False,
                            pin_memory=True,
                            drop_last=False,
                            num_workers=cfg.training.workers
                            )
        autocast = torch.autocast('cuda', enabled=cfg.training.use_amp, dtype=torch.half) # you can use with T4 gpu. or newer
        
        label_columns = valid_ds.labels
        N_LABELS = len(label_columns)
        with tqdm(valid_dl, leave=True) as pbar:
            with torch.no_grad():
                for idx, (x, t) in enumerate(pbar):
                    if isinstance(x, tuple) or isinstance(x, list):
                        x1, x2, x3 = x
                        x1 = x1.to(device)
                        x2 = x2.to(device)
                        x3 = x3.to(device)
                        with autocast:
                            y = model(x1, x2, x3)
                    else:
                        x = x.to(device)
                        with autocast:
                            y = model(x)
                    study_ids = t['study_id'][:, 0]
                    
                    for col in range(N_LABELS):
                        pred = y['labels'][:,col*3:col*3+3].softmax(dim=1).cpu().numpy()
                        lab = label_columns[col]
                        for i in range(len(study_ids)):
                            row = [str(study_ids[i].item()) + '_' + lab, pred[i, 0], pred[i, 1], pred[i, 2]]
                            model_predictions.append(row)
                            
    new_pred = pd.DataFrame(model_predictions, columns=['row_id', 'normal_mild', 'moderate', 'severe'])
    
    tmp = df.set_index('study_id').stack().reset_index(name='sample_weight').rename(columns={'level_1': 'location'})
    tmp = tmp[~tmp.location.map(lambda s: 'stratum' in s or 'fold' in s)]
    tmp['row_id'] = tmp.apply(lambda r: str(r.study_id) +'_' +r.location, axis=1)
    tmp['normal_mild'] = 0
    tmp.loc[tmp.sample_weight == 0, 'normal_mild'] = 1
    tmp['moderate'] = 0
    tmp.loc[tmp.sample_weight == 1, 'moderate'] = 1
    tmp['severe'] = 0
    tmp.loc[tmp.sample_weight == 2, 'severe'] = 1
    tmp['sample_weight'] = tmp.sample_weight.map({0: 1, 1: 2, 2: 4, -100: 0})
    
    submission = tmp.copy()
    del submission['study_id'], submission['location'], submission['sample_weight']
    submission['normal_mild'] = 1/3.0
    submission['moderate'] = 1/3.0
    submission['severe'] = 1/3.0

    submission = pd.concat([submission[~submission.row_id.isin(new_pred.row_id.unique())], new_pred])
    
    tmp_true = tmp.copy()
    del tmp_true['study_id'], tmp_true['location']

    LABELS = ['normal_mild','moderate','severe']
    tmp_true = tmp_true.sort_values(by='row_id')
    tmp_true = tmp_true[tmp_true[LABELS].max(axis=1) == 1].reset_index(drop=True)
    # tmp_preds2 = tmp_preds2.sort_values(by='row_id')
    submission = submission.sort_values(by='row_id')
    submission = submission[submission.row_id.isin(tmp_true.row_id)].reset_index(drop=True)
    tmp_true = tmp_true[tmp_true.row_id.isin(submission.row_id)].reset_index(drop=True)
    scr = officialloss.score(solution=tmp_true.copy(), submission=submission.copy(), row_id_column_name='row_id', any_severe_scalar=1.0)
    logger.info(f'Official CV score: {scr}')


def train_one_fold(model, cfg, fold: int):
    df, __, __ = load_train_files(relative_directory=relative_directory)

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
                if isinstance(x, tuple) or isinstance(x, list):
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
                    if isinstance(x, tuple) or isinstance(x, list):
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

