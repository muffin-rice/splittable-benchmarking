import pandas as pd
from pandas import DataFrame
from copy import deepcopy
import numpy as np
import ast

# dataframe helper functions (renaming cols, etc)
def open_fname_as_list_dicts(fname : str) -> [{}]:
    with open(fname, 'r') as f:
        s = f.read()
    l = []
    for d in s.split('\n'):
        try:
            l.append(ast.literal_eval(d))
        except:
            pass

    return l


def rename_fname_column(df2 : DataFrame) -> DataFrame:
    df = deepcopy(df2)
    df['video_index'] = df['fname'].apply(lambda x: x[-15:-11])
    df['frame_index'] = df['fname'].apply(lambda x: x[-10:])
    df = df.drop('fname', axis=1)
    #    df['fname'] = df['fname'].apply(lambda x : x[-15:])
    return df


def remove_empty_cols(df : DataFrame) -> DataFrame:
    return df.dropna(axis=1, how='all')


def count_missing_preds(df2 : DataFrame) -> DataFrame:
    df = deepcopy(df2)
    df['missing_preds_len'] = df['missing_preds'].apply(lambda x: len(x))
    df = df.drop('missing_preds', axis=1)

    return df


def count_objects(df : DataFrame, pred_prefix : str = 'pred_') -> int:
    x = 0
    for column in df:
        if pred_prefix in column:
            x += 1

    return x


def calculate_expected_misses(num_objects : int, refresh_fps : int, n : int) -> float:
    num_trials = n / refresh_fps
    lam = n / num_objects
    return refresh_fps * (refresh_fps - 1) / 2 / lam * num_trials


def calculate_deltas(df2 : DataFrame, pred_prefix : str = 'pred_') -> DataFrame:
    '''adds a column that calculates the iou decay from the tracker'''
    df = deepcopy(df2)
    for col in df:
        if pred_prefix not in col:
            continue

        df[f'delta_{col}'] = df[col].shift() - df[col]
        # df = df.drop(col, axis=1)

    return df


def calculate_object_first_accuracy(df2, pred_prefix : str = 'pred'):
    # TODO:
    # get object detection performance
    df = deepcopy(df2)
    d = {}
    for col in df:
        if pred_prefix not in col:
            continue

        d[col] = round(df[col].loc[df['compressor'].notna()].mean(), 4)

    return d


def calculate_deltas_in_dict(df2, delta_pred_prefix = 'delta_pred_') -> {str: (float, float, float, int)}:
    '''calculates the deltas stats and returns it in the {object_key : stats} format'''
    df = calculate_deltas(df2)
    d = {}
    df = df.loc[df['tracker']] # filter by tracker=True
    for col in df:
        if delta_pred_prefix not in col:
            continue

        d[col] = (df[col].mean(), df[col].abs().mean(), df[col].sum(), df[col].count())

    return d


def split_df_by_cat(df : DataFrame, mode: int = 1) -> (any, [str]):
    # mode 1 is mp4 files (video, split by that)
    if mode == 1:
        gb_df = df.groupby(df['fname'])
        return gb_df, list(gb_df.groups.keys())

    else:
        # mode 2 is png files, must split by the directory
        df['png_dirs'] = df['fname'].str.split('/').str[-2]
        gb_df = df.groupby(df['png_dirs'])

        return gb_df, list(gb_df.groups.keys())