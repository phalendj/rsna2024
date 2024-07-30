import pandas as pd


CONDITIONS = [
    'Spinal Canal Stenosis', 
    'Left Neural Foraminal Narrowing', 
    'Right Neural Foraminal Narrowing',
    'Left Subarticular Stenosis',
    'Right Subarticular Stenosis'
]

LEVELS = [
    'L1/L2',
    'L2/L3',
    'L3/L4',
    'L4/L5',
    'L5/S1',
]

LABEL2ID = {'Normal/Mild': 0, 'Moderate':1, 'Severe':2}
ID2LABEL = {0: 'Normal/Mild', 1: 'Moderate', 2: 'Severe'}


def load_train_files(relative_directory: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads and returns tuple of train files (train_stratums, label_coordinates, series_descriptions)"""
    df = pd.read_csv(f'{relative_directory}/train_stratums.csv')

    df = df.fillna(-100)
    df = df.replace(LABEL2ID)
    
    dfc = pd.read_csv(f'{relative_directory}/train_label_coordinates.csv')
    dfd = pd.read_csv(f'{relative_directory}/train_series_descriptions.csv')

    # Drop cervical spine image
    # https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/514891
    dfd = dfd[dfd.series_id != 3892989905]
    dfc = dfc[dfc.series_id != 3892989905]

    # remove mislabeled instances
    # https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/521341#2931853
    bad_series_ids = set()
    for level in LEVELS:
        ldfc = dfc[(dfc.condition == 'Left Neural Foraminal Narrowing') & (dfc.level == level)]
        rdfc = dfc[(dfc.condition == 'Right Neural Foraminal Narrowing') & (dfc.level == level)]
        tmp = rdfc.merge(ldfc, how='inner', on=['study_id', 'series_id', 'instance_number'])
        bad_series_ids |= set(tmp.series_id.unique())

    dfd = dfd[~dfd.series_id.isin(bad_series_ids)]
    dfc = dfc[~dfc.series_id.isin(bad_series_ids)]

    return df, dfc, dfd


def load_test_files(relative_directory: str):
    dfd = pd.read_csv(f'{relative_directory}/test_series_descriptions.csv')
    return dfd


def create_column(condition, level):
    return condition.replace(' ', '_').lower() + '_' + level.replace('/', '_').lower()


