import logging
from tqdm import tqdm
import math
from pathlib import Path
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

from hydra.core.hydra_config import HydraConfig

from datasets import load_train_files
from utils import relative_directory
from datasets import factory as dsfactory
import loss_functions as lffactory


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
        warmup_steps = epochs/10 * nbatches // grad_acc
        num_total_steps = epochs * nbatches // grad_acc

        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=warmup_steps,
                                                    num_training_steps=num_total_steps,
                                                    num_cycles=opt_cfg.n_cycles)
        result['scheduler'] = scheduler

    return result



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
                            drop_last=True,
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
        print(f'start epoch {epoch}')
        model.train()
        total_loss = 0
        with tqdm(train_dl, leave=True) as pbar:
            optimizer.zero_grad()
            for idx, (x, t) in enumerate(pbar):  
                x = x.to(device)
                t = t['labels'].to(device)

                with autocast:
                    loss = 0
                    y = model(x)
                    N_LABELS = t.shape[-1]
                    for col in range(N_LABELS):
                        pred = y[:,col*3:col*3+3]
                        gt = t[:, col]
                        loss = loss + criterion(pred, gt) / N_LABELS

                    total_loss += loss.item()
                    if GRAD_ACC > 1:
                        loss = loss / GRAD_ACC

                if not math.isfinite(loss):
                    print(f"Loss is {loss}, stopping training")
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
        print(f'train_loss:{train_loss:.6f}')
        train_losses.append(train_loss)
        total_loss = 0

        model.eval()
        with tqdm(valid_dl, leave=True) as pbar:
            with torch.no_grad():
                for idx, (x, t) in enumerate(pbar):

                    x = x.to(device)
                    t = t['labels'].to(device)

                    with autocast:
                        loss = 0
                        y = model(x)
                        N_LABELS = t.shape[-1]
                        for col in range(N_LABELS):
                            pred = y[:,col*3:col*3+3]
                            gt = t[:]

                            loss = loss + criterion(pred, gt) / N_LABELS
                            y_pred = pred.float()

                        total_loss += loss.item()   

        val_loss = total_loss/len(valid_dl)
        print(f'val_loss:{val_loss:.6f}')
        val_losses.append(val_loss)
        if val_loss < best_loss:

            if device!='cuda:0':
                model.to('cuda:0')                

            print(f'epoch:{epoch}, best weighted_logloss updated from {best_loss:.6f} to {val_loss:.6f}')
            best_loss = val_loss
            torch.save(model.state_dict(), fname)
            print(f'{fname} is saved')
            es_step = 0

            if device!='cuda:0':
                model.to(device)

        else:
            es_step += 1
            if es_step >= cfg.training.early_stopping:
                print('early stopping')
                break  

