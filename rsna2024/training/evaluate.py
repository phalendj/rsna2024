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

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.ndimage import generate_binary_structure, binary_erosion

from hydra.core.hydra_config import HydraConfig

try:
    from datasets import load_train_files, load_test_files, LEVELS, CONDITIONS
    import utils as rsnautils
    from datasets import factory as dsfactory
    import loss_functions as lffactory
    import loss_functions.official as officialloss
    import models
except:
    from ..datasets import load_train_files, load_test_files, LEVELS, CONDITIONS
    from .. import utils as rsnautils
    from ..datasets import factory as dsfactory
    from .. import loss_functions as lffactory
    from ..loss_functions import official as officialloss
    from .. import models


logger = logging.getLogger(__name__)


def evaluate(model, cfg):
    df, __, __ = load_train_files(relative_directory=rsnautils.relative_directory, clean=cfg.clean)
    # load sample submission file
    device = 'cuda:0'

    model_predictions = []
    for fold in cfg.training.use_folds:
        logger.info(f'Evaluating Fold {fold}')
        if cfg.load_directory is None:
            output_directory = Path(HydraConfig.get().runtime.output_dir)
        else:
            output_directory = Path(cfg.load_directory)
        fname = output_directory / (model.name() + f'_fold{fold}.pth')
        logger.info(f'Loading Model from {fname}')
        model.load_state_dict(torch.load(fname, weights_only=True))
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
                    if isinstance(x, dict):
                        x = rsnautils.move_dict(x, device=device)
                        with autocast:
                            y = model(x)

                    elif isinstance(x, tuple) or isinstance(x, list):
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

                    try:
                        study_ids = t['study_id'].numpy()
                    except ValueError:
                        study_ids = t['study_id']
                    
                    for col in range(N_LABELS):
                        pred = y['labels'][:,col*3:col*3+3].softmax(dim=1).cpu().numpy()
                        lab = label_columns[col]
                        for i in range(len(study_ids)):
                            row = [str(study_ids[i].item()) + '_' + lab, pred[i, 0], pred[i, 1], pred[i, 2]]
                            model_predictions.append(row)
                            
    new_pred = pd.DataFrame(model_predictions, columns=['row_id', 'normal_mild', 'moderate', 'severe'])
    pcol = ['normal_mild', 'moderate', 'severe']
    totals = new_pred[pcol].sum(axis=1)
    for c in pcol:
        new_pred[c] /= totals
    fname = output_directory / 'oof.csv'
    new_pred.to_csv(fname, index=False)
    
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

    logger.info(f'Scoring for {len(submission)} submission vs {len(tmp_true)} true')

    scr = officialloss.score(solution=tmp_true.copy(), submission=submission.copy(), row_id_column_name='row_id', any_severe_scalar=1.0)
    logger.info(f'Official CV score: {scr}')


def predict(cfg):
    rsnautils.CLEAN = False
    logger.info('Generate predictions')
    if cfg.load_directory is None:
        model_directory = Path(HydraConfig.get().runtime.output_dir)
    else:
        model_directory = Path(cfg.load_directory)

    if cfg.mode == 'test':
        mode = 'test'
    else:
        mode = 'valid'

    if mode == 'valid':
        df, dfc, dfd = load_train_files(rsnautils.relative_directory, clean=False)
        if cfg.clean:
            rsnautils.set_clean((cfg.clean // 1000)*1000)
            df_clean, __, __ = load_train_files(rsnautils.relative_directory, clean=True)
            df_clean_i = df_clean.set_index('study_id')
        else:
            df_clean_i = df.set_index('study_id')
    else:
        dfd = load_test_files(rsnautils.relative_directory)
        df_clean_i = pd.DataFrame([[-1, -1]], columns=['study_id', 'fold']).set_index('study_id')
    device = 'cuda:0'

    all_models = []
    for fold in cfg.training.use_folds:
        model = models.create_model(cfg.model, fold=fold)
        fname = model_directory / (model.name() + f'_fold{fold}.pth')
        logger.info(f'Loading model from {fname}')
        model.load_state_dict(torch.load(fname, weights_only=True))
        __ = model.eval()
        all_models.append(model.to(device))

    autocast = torch.autocast('cuda', enabled=cfg.training.use_amp, dtype=torch.half) # you can use with T4 gpu. or newer
    valid_ds = dsfactory.create_dataset(study_ids=dfd.study_id.unique(), mode=mode, cfg=cfg.dataset)
    valid_dl = DataLoader(valid_ds,
                        batch_size=cfg.training.batch_size,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False,
                        num_workers=cfg.training.workers)

    model_predictions = []
        
    label_columns = valid_ds.labels
    N_LABELS = len(label_columns)
    with tqdm(valid_dl, leave=True) as pbar:
        with torch.no_grad():
            for idx, (x, t) in enumerate(pbar):
                try:
                    if isinstance(x, dict):
                        x = rsnautils.move_dict(x, device=device)
                        with autocast:
                            preds = [model(x) for model in all_models]

                    elif isinstance(x, tuple) or isinstance(x, list):
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

                    try:
                        study_ids = t['study_id'].numpy()
                    except ValueError:
                        study_ids = t['study_id']
                    
                    for i, study_id in enumerate(study_ids):
                        for col in range(N_LABELS):
                            if study_id in df_clean_i.index:
                                fold = df_clean_i.loc[study_id, 'fold']
                                pred = preds[fold]
                                pred = pred['labels'][i,col*3:col*3+3].softmax(dim=0).cpu().numpy()
                            else:
                                pred = torch.mean(torch.stack([pred['labels'][i,col*3:col*3+3].softmax(dim=0).cpu() for pred in preds], dim=0), dim=0).numpy()

                            lab = label_columns[col]
                            row = [str(study_id) + '_' + lab, pred[0], pred[1], pred[2]]
                            model_predictions.append(row)
                except Exception as e:
                    logger.exception(e)
                        
    new_pred = pd.DataFrame(model_predictions, columns=['row_id', 'normal_mild', 'moderate', 'severe']).fillna(0.33)
    pcol = ['normal_mild', 'moderate', 'severe']
    totals = new_pred[pcol].sum(axis=1)
    for c in pcol:
        new_pred[c] /= totals
    fname = model_directory / 'submission.csv'
    if mode == 'test':
        fname = 'submission.csv'
        if len(valid_ds.fails) > 0:
            fail_file = 'bad_data.csv'
            pd.DataFrame(list(valid_ds.fails), columns=['study_id', 'series_description', 'level']).to_csv(fail_file, index=False)
    logger.info(f'Writing result to {fname}')
    new_pred.to_csv(fname, index=False)


def generate_instance_numbers(cfg):
    logger.info('Generate instance numbers file')
    if cfg.load_directory is None:
        model_directory = Path(HydraConfig.get().runtime.output_dir)
    else:
        model_directory = Path(cfg.load_directory)

    if cfg.mode == 'test':
        mode = 'test'
    else:
        mode = 'valid'


    if mode == 'valid':
        df, dfc, dfd = load_train_files(rsnautils.relative_directory, clean=False)
        if cfg.clean:
            rsnautils.set_clean((cfg.clean // 1000)*1000)
            df_clean, __, __ = load_train_files(rsnautils.relative_directory, clean=True)
            df_clean_i = df_clean.set_index('study_id')
        else:
            df_clean_i = df.set_index('study_id')
    else:
        dfd = load_test_files(rsnautils.relative_directory)
        df_clean_i = pd.DataFrame([[-1, -1]], columns=['study_id', 'fold']).set_index('study_id')


    device = 'cuda:0'

    all_models = []
    for fold in cfg.training.use_folds:
        model = models.create_model(cfg.model, fold=fold)
        fname = model_directory / (model.name() + f'_fold{fold}.pth')
        logger.info(f'Loading model from {fname}')
        model.load_state_dict(torch.load(fname, weights_only=True))
        __ = model.eval()
        all_models.append(model.to(device))

    results = []
    autocast = torch.autocast('cuda', enabled=cfg.training.use_amp, dtype=torch.half) # you can use with T4 gpu. or newer
    w = torch.arange(30)
    valid_ds = dsfactory.create_dataset(study_ids=dfd.study_id.unique(), mode=mode, cfg=cfg.dataset)
    valid_dl = DataLoader(valid_ds,
                        batch_size=cfg.training.batch_size,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False,
                        num_workers=cfg.training.workers)

    with torch.no_grad():
        for x, t in tqdm(valid_dl):
            try:
                if isinstance(x, dict):
                    x = rsnautils.move_dict(x, device=device)
                    with autocast:
                        preds = [model(x) for model in all_models]

                elif isinstance(x, tuple) or isinstance(x, list):
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

                try:
                    study_ids = t['study_id'].numpy()
                except Exception:
                    study_ids = t['study_id']
            
                try:
                    series_ids = x['series_id'].cpu().numpy()
                except Exception:
                    series_ids = x['series_id']

                tmp = []
                for i, study_id in enumerate(study_ids):
                    if study_id in df_clean_i.index:
                        fold = df_clean_i.loc[study_id, 'fold']
                        pred = preds[fold]
                        v = pred['instance_labels']
                        if v.shape[-1] == 10:
                            v = v.reshape(len(study_ids), -1, 5, 2)
                        class_pred = v.softmax(dim=-1)[i, ..., 1].cpu()
                        if len(class_pred.shape) == 2:
                            class_pred = class_pred.transpose(-1,-2)
                    else:
                        v = preds[0]['instance_labels']
                        if v.shape[-1] == 10:
                            class_pred = torch.mean(torch.stack([pred['instance_labels'].reshape(len(study_ids), -1, 5, 2).softmax(dim=-1)[i, ..., 1].cpu() for pred in preds], dim=0), dim=0).transpose(-1,-2)
                        else:
                            class_pred = torch.mean(torch.stack([pred['instance_labels'].softmax(dim=-1)[i, ..., 1].cpu() for pred in preds], dim=0), dim=0)
                    tmp.append(class_pred)

                class_pred = torch.stack(tmp, dim=0)
                #print(class_pred.shape)
                loc = (class_pred*w).sum(dim=-1)/class_pred.sum(dim=-1)
                ind = torch.round(loc.float()).long()

                if len(ind.shape) == 1:
                    instance_numbers = [x['instance_numbers'].cpu()[i, j].item() for i, j in enumerate(ind)]

                    for study_id, series_id, z in zip(study_ids, series_ids, instance_numbers):
                        for lev in LEVELS:
                            results.append({'study_id': study_id, 'series_id': series_id, 'instance_number': z, 'condition': 'Spinal Canal Stenosis', 'level': lev, 'x': 0, 'y': 0})
                else:
                    for k, lev in enumerate(LEVELS):
                        instance_numbers = [x['instance_numbers'].cpu()[i, j].item() for i, j in enumerate(ind[:, k])]

                        for study_id, series_id, z in zip(study_ids, series_ids, instance_numbers):
                            results.append({'study_id': study_id, 'series_id': series_id, 'instance_number': z, 'condition': 'Spinal Canal Stenosis', 'level': lev, 'x': 0, 'y': 0})
            except Exception as e:
                logger.exception(e)
                pass

    pred_center_df = pd.DataFrame(results)
    fname = model_directory / 'predicted_label_coordinates.csv'
    if mode == 'test':
        fname = 'predicted_label_coordinates.csv'
        if len(valid_ds.fails) > 0:
            fail_file = 'bad_data.csv'
            pd.DataFrame(list(valid_ds.fails), columns=['study_id', 'series_description', 'level']).to_csv(fail_file, index=False)


    logger.info(f'Writing result to {fname}')
    pred_center_df.to_csv(fname, index=False)
    

def detect_peaks(image, size=10):
    """
    Takes an image and detect the peaks usingthe local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """

    # define an 8-connected neighborhood
    neighborhood = generate_binary_structure(2,2)

    #apply the local maximum filter; all pixel of maximal value 
    #in their neighborhood are set to 1
    local_max = maximum_filter(image, size=size)==image
    #local_max is a mask that contains the peaks we are 
    #looking for, but also the background.
    #In order to isolate the peaks we must remove the background from the mask.

    #we create the mask of the background
    background = (image==0)

    #a little technicality: we must erode the background in order to 
    #successfully subtract it form local_max, otherwise a line will 
    #appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    #we obtain the final mask, containing only peaks and a ring where the cut was applied, 
    #by removing the background from the local_max mask (xor operation)
    detected_peaks = local_max ^ eroded_background

    return detected_peaks


def corr2d(X, K):  #@save
    """Compute 2D cross-correlation."""
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y


def detect_best_levels(OUT, cut = 0.5):
    v = torch.sum(OUT, dim=0).cpu().numpy()
    v[v<cut] = 0
    mask = detect_peaks(v)
    
    # remove the ring
    X = torch.tensor(mask, dtype=torch.long)
    K = -torch.ones(size=(3,3))
    K[1,1] = 1
    Y = corr2d(X, K)
    
    # Now find the best looking points
    x, y = torch.where(Y==1)
    pts = torch.stack([x, y]).T.numpy() + 1  # Need one for padding
    max_i = 0
    max_val = 0
    if len(pts) > 5:
        for i in range(len(pts)-4):
            p = pts[i:i+5]
            test = np.sum([v[x0,y0] for x0, y0 in p])
            if test > max_val:
                max_i = i
                max_val = test
        pts = pts[max_i:max_i+5]
            
    return pts


def detect_single_levels(OUT, cut = 0.9):
    d = OUT.shape[0]
    pts = np.zeros((d, 2))
    for i in range(d):
        x, y = torch.where(OUT[i]>cut)
        pts[i, 0] = x.float().mean().item()
        pts[i, 1] = y.float().mean().item()

    return pts


def fill_coordinates(row, study, dft, df):
    # assert row.study_id == study.study_id
    series = study.get_series(row.series_id)
    stack = series.get_stack(row.instance_number)
    world_x, world_y, world_z = stack.get_world_coordinates(instance_number=row.instance_number, x=row.x, y=row.y)
    fold = -1 if df is None else df[df.study_id == study.study_id].iloc[0].fold
    condition = row.condition
    if df is None:
        tdft = dft[(dft.condition == condition) & (dft.level == row.level)]
    else:
        tdft = dft[dft.study_id.isin(df[df.fold != fold].study_id.unique()) & (dft.condition == condition) & (dft.level == row.level)]
    condition_lower = condition.lower().replace(' ', '_')
    CONDITION_2_SERIES_DESC = {'Spinal Canal Stenosis': 'Sagittal T2/STIR',
                                'Left Neural Foraminal Narrowing': 'Sagittal T1',
                                'Right Neural Foraminal Narrowing': 'Sagittal T1',
                                'Left Subarticular Stenosis': 'Axial T2',
                                'Right Subarticular Stenosis': 'Axial T2'}
    res = []
    for target_condition in CONDITIONS:
        if target_condition != condition:
            target = target_condition.lower().replace(' ', '_')
            x = tdft[f'world_x_{target}'] - tdft[f'world_x_{condition_lower}']
            y = tdft[f'world_y_{target}'] - tdft[f'world_y_{condition_lower}']
            z = tdft[f'world_z_{target}'] - tdft[f'world_z_{condition_lower}']
            x_offset, y_offset, z_offset = x.mean(), y.mean(), z.mean()
            projected_world_x = world_x + x_offset
            projected_world_y = world_y + y_offset
            projected_world_z = world_z + z_offset
            target_stack, target_series, dist = study.find_closest_stack_and_series(series_description=CONDITION_2_SERIES_DESC[target_condition], world_x=projected_world_x, world_y=projected_world_y, world_z=projected_world_z, required_in=True)
            if target_stack is not None:
                k, proj_x, proj_y = target_stack.get_pixel_from_world(world_x=projected_world_x, world_y=projected_world_y, world_z=projected_world_z)
                inum, px, py = target_stack.instance_numbers[k], proj_x, proj_y
                res.append({'study_id': row.study_id, 'series_id': target_series.series_id, 'instance_number': inum, 'condition': target_condition, 'level': row.level, 'x': px, 'y': py})
            
            
    return pd.DataFrame(res)



def generate_xy_values(cfg):
    """
    Given a file with only the instance numbers specified for Spinal Canal Stensosis, Fill out the XY coordinates for Spinal Canal, then translate to the rest of the points using a simple offset
    """
    logger.info('Generate XY coordinates from segmentation model')
    if cfg.load_directory is None:
        model_directory = Path(HydraConfig.get().runtime.output_dir)
    else:
        model_directory = Path(cfg.load_directory)
    
    device = 'cuda:0'

    if cfg.mode == 'test':
        mode = 'test'
    else:
        mode = 'valid'


    if mode == 'valid':
        df, dfc, dfd = load_train_files(rsnautils.relative_directory, clean=False)
        if cfg.clean:
            rsnautils.set_clean((cfg.clean // 1000)*1000)
            df_clean, __, __ = load_train_files(rsnautils.relative_directory, clean=True)
            df_clean_i = df_clean.set_index('study_id')
        else:
            df_clean_i = df.set_index('study_id')
    else:
        dfd = load_test_files(rsnautils.relative_directory)
        df_clean_i = pd.DataFrame([[-1, -1]], columns=['study_id', 'fold']).set_index('study_id')


    all_models = []
    for fold in cfg.training.use_folds:
        model = models.create_model(cfg.model, fold=fold)
        fname = model_directory / (model.name() + f'_fold{fold}.pth')
        logger.info(f'Loading model from {fname}')
        model.load_state_dict(torch.load(fname, weights_only=True))
        __ = model.eval()
        all_models.append(model.to(device))

    valid_ds = dsfactory.create_dataset(study_ids=dfd.study_id.unique(), mode=mode, cfg=cfg.dataset)

    old_dfc = pd.read_csv(cfg.dataset.center_file)

    results = []
    autocast = torch.autocast('cuda', enabled=cfg.training.use_amp, dtype=torch.half) # you can use with T4 gpu. or newer
    valid_dl = DataLoader(
                        valid_ds,
                        batch_size=cfg.training.batch_size,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False,
                        num_workers=cfg.training.workers
                        )
    with torch.no_grad():
        for x, t in tqdm(valid_dl):
            try:
                if isinstance(x, dict):
                    x = rsnautils.move_dict(x, device=device)
                    with autocast:
                        preds = [model(x) for model in all_models]
                elif isinstance(x, tuple) or isinstance(x, list):
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
                try:
                    study_ids = t['study_id'].numpy()
                except Exception:
                    study_ids = t['study_id']
            
                try:
                    series_ids = x['series_id'].cpu().numpy()
                except Exception:
                    series_ids = x['series_id']

                offsets = x['offsets'].cpu().numpy()
                scalings = x['scalings'].cpu().numpy()
                for i in range(len(study_ids)):
                    study_id = study_ids[i]
                    series_id = series_ids[i]
                    scaling = scalings[i]
                    offset = offsets[i]
                    if study_id in df_clean_i.index:
                        fold = df_clean_i.loc[study_id, 'fold']
                        OUT = preds[fold]['masks'][i]
                    else:
                        OUT = torch.mean(torch.stack([pred['masks'][i] for pred in preds], dim=0), dim=0)
                        min_values = OUT.flatten(1,2).min(dim=1)[0].view(OUT.shape[0],1,1)
                        max_values = OUT.flatten(1,2).max(dim=1)[0].view(OUT.shape[0],1,1)
                        OUT = (OUT - min_values)/(max_values - min_values)

                    # cut = 0.1
                    # pts = detect_best_levels(OUT.float(), cut=cut)
                    # while len(pts) < 5:
                    #     cut /= 2
                    #     pts = detect_best_levels(OUT.float(), cut=cut)
                    # pts = pts/scaling - offset
                    # pts = np.array(sorted(pts, key=lambda x: x[-1]))

                    pts = detect_single_levels(OUT)/scaling - offset
                    for k in range(5):
                        X, Y = pts[k]
                        lev = LEVELS[k]
                        results.append({'study_id': study_id, 'series_id': series_id, 'instance_number': np.nan, 'condition': 'Spinal Canal Stenosis', 'level': lev,  'x': X, 'y': Y})
            except Exception as e:
                logger.exception(e)
                pass

    pred_center_df = pd.DataFrame(results)
    pred_center_df = pred_center_df.drop('instance_number', axis=1).merge(old_dfc[['study_id', 'series_id', 'condition', 'level', 'instance_number']], on=['study_id', 'series_id', 'condition', 'level'])[['study_id', 'series_id', 'instance_number', 'condition', 'level', 'x', 'y']]
    fname = model_directory / 'predicted_center_coordinates.csv'
    if mode == 'test':
        fname = 'predicted_center_coordinates.csv'
        if len(valid_ds.fails) > 0:
            fail_file = 'bad_data.csv'
            pd.DataFrame(list(valid_ds.fails), columns=['study_id', 'series_description', 'level']).to_csv(fail_file, index=False)
    logger.info(f'Writing result to {fname}')
    pred_center_df.to_csv(fname, index=False)
    logger.info(f'Wrote output to {fname}')

    logger.info('Fill out dataframe')
    if mode == 'test':
        dft = pd.read_csv(f'{model_directory}/train_coordinates_translated.csv')
    else:
        dft = pd.read_csv(f'{cfg.directories.relative_directory}/train_coordinates_translated.csv')

    res = []
    fails = []
    for study in tqdm(valid_ds.studies):
        tmp = pred_center_df[pred_center_df.study_id == study.study_id]
        for row in tmp.itertuples():
            try:
                study.load()
                if mode == 'test':
                    filled_df = fill_coordinates(row, study=study, dft=dft, df=None)
                else:
                    filled_df = fill_coordinates(row, study=study, dft=dft, df=df)
                res.append(filled_df)
            except Exception as e:
                level_dict = {lev: i for i, lev in enumerate(LEVELS)}
                fails.append((study.study_id, 'Sagittal T2/STIR', level_dict[row.level]))
                logger.exception(e)
                logger.error(f'Error on {row}')
        study.unload()

    fname = model_directory / 'all_predicted_center_coordinates.csv'
    if mode == 'test':
        fname = 'all_predicted_center_coordinates.csv'
        fails += list(valid_ds.fails)

        if len(fails) > 0:
            fail_file = 'bad_data.csv'
            pd.DataFrame(fails, columns=['study_id', 'series_description', 'level']).to_csv(fail_file, index=False)
    logger.info(f'Writing result to {fname}')
    if len(res) > 0:
        temp_filler = pd.concat(res)
        full_pred_center = pd.concat([temp_filler, pred_center_df]).reset_index(drop=True)
        full_pred_center.to_csv(fname, index=False)
    else:
        print('No results')
        pred_center_df.to_csv(fname, index=False)

    logger.info(f'Wrote output to {fname}')
        