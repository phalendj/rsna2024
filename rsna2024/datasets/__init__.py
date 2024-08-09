import pandas as pd
import logging


pd.set_option("future.no_silent_downcasting", True)


logger = logging.getLogger(__name__)


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

def clean_coordinates(dfc):
    """
    Removes 17 bad points, all from the Spinal Canal Stenosis diagnosis
    """
    bad_ones = []
    for (study_id, series_id, condition), gdf in dfc.groupby(['study_id', 'series_id', 'condition']):
        if condition == 'Spinal Canal Stenosis':
            gdf = gdf.sort_values('level')
            m = gdf.x.median()
            if gdf.x.min() < m - 100.0:
                lev = gdf[gdf.x < m - 100].level.iloc[0]
                bad_ones.append((study_id, series_id, lev))
                
                # break
            elif gdf.x.max() > m + 100.0:
                lev = gdf[gdf.x > m + 100].level.iloc[0]
                bad_ones.append((study_id, series_id, lev))
                
            elif not gdf.y.is_monotonic_increasing:
                for i in range(1, len(gdf)):
                    if gdf.y.iloc[i-1] > gdf.y.iloc[i]:
                        lev = gdf.level.iloc[i]
                bad_ones.append((study_id, series_id, lev))

        ## TODO: For Foraminal Narrowing, we don't see random points, but we do see a lot of issues where the left and right levels are off by 1.  Not sure how to clean these up
        ## We should also ensure the left and right points are on the left and right of the images
        if condition == 'Left Neural Foraminal Narrowing' or condition == 'Right Neural Foraminal Narrowing':
            gdf = gdf.sort_values('level')
            m = gdf.x.median()
            if gdf.x.min() < m - 100.0:
                lev = gdf[gdf.x < m - 100].level.iloc[0]
                bad_ones.append((study_id, series_id, lev))
                # break
            elif gdf.x.max() > m + 100.0:
                lev = gdf[gdf.x > m + 100].level.iloc[0]
                bad_ones.append((study_id, series_id, lev))
            elif not gdf.y.is_monotonic_increasing:
                for i in range(1, len(gdf)):
                    if gdf.y.iloc[i-1] > gdf.y.iloc[i]:
                        lev = gdf.level.iloc[i]
                bad_ones.append((study_id, series_id, lev))
    #     if condition == 'Left Subarticular Stenosis' or condition == 'Right Subarticular Stenosis':
    #         display(gdf)
    #         break
    return dfc[~dfc.apply(lambda row: (row.study_id, row.series_id, row.level) in set(bad_ones), axis=1)].copy()
            
            

def clean_multiple(df, dfc, dfd, condition, number):
    tdfc = dfc[dfc.condition.map(lambda s: condition in s)]

    xdfc = tdfc.groupby('series_id').count()['instance_number']
    series_ids = xdfc[xdfc == number].index

    study_ids = dfd.loc[dfd.series_id.isin(series_ids), 'study_id'].unique()
    df = df[df.study_id.isin(study_ids)]
    dfc = dfc[dfc.study_id.isin(study_ids)]
    dfd = dfd[dfd.study_id.isin(study_ids)]
    return df, dfc, dfd



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

    # remove coordinates that are way off:
    dfc = clean_coordinates(dfc)

    # Only take data where all of what we need is present in a single series
    df, dfc, dfd = clean_multiple(df, dfc, dfd, condition='Spinal', number=5)
    df, dfc, dfd = clean_multiple(df, dfc, dfd, condition='Foraminal', number=10)
    df, dfc, dfd = clean_multiple(df, dfc, dfd, condition='Subarticular', number=10)
    return df, dfc, dfd


def load_test_files(relative_directory: str):
    dfd = pd.read_csv(f'{relative_directory}/test_series_descriptions.csv')
    return dfd


def create_column(condition, level):
    return condition.replace(' ', '_').lower() + '_' + level.replace('/', '_').lower()


