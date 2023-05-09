import pandas as pd
from pandas import DataFrame
from copy import deepcopy
import numpy as np
import ast
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
from collections import defaultdict

VIRAT_COLS = ['object_id', 'duration', 'frame_num', 'x', 'y', 'w', 'h', 'class_name']

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
        df['png_dirs'] = df['fname'].str.split('/').str[-3]
        gb_df = df.groupby(df['png_dirs'])

        return gb_df, list(gb_df.groups.keys())

def get_num_tracked(df : DataFrame, prefix = 'p_d_', object_prefix = 'p_s_'):
    '''adds a row that shows the number of tracked objects per iteration'''
    all_objects = get_all_tracked_objects(df, prefix)

    def count_nonzero_objects(row):
        x = 0
        for object in all_objects:
            if math.isnan(row[f'{object_prefix}{object}']):
                continue

            x += 1

        return x

    def count_high_objects(row):
        x = 0
        for object in all_objects:
            if math.isnan(row[f'{object_prefix}{object}']):
                continue

            if row[f'{object_prefix}{object}'] < 0.5:
                continue

            x += 1

        return x

    df['num_tracked_objects'] = df.apply(count_nonzero_objects, axis=1)
    df['num_high_objects'] = df.apply(count_high_objects, axis=1)

def get_all_tracked_objects(df : DataFrame, prefix='p_d_') -> {str}:
    '''get the object numbers for objects tracked in this dataframe'''
    return [col.split('_')[-1] for col in df if col.startswith(prefix)]

def extract_frames(cap, vid_shape = (1280, 720), frame_limit=150, transpose_frame = False) -> (bool, np.ndarray):
    '''From a cv2 VideoCapture, return a stack of frame_limit subset of the video'''
    # get 15 frames from random starting point
    frame_limit = min(int(cap.get(7)), frame_limit)

    frames = []
    for i in range(frame_limit):
        success, frame = cap.read()
        if not success:
            if len(frames) > 0:
                break

            return False, np.array(frames)

        if transpose_frame:
            frame = frame.transpose(1,0,2)

        if vid_shape is None:
            frames.append(frame)
        else:
            frames.append(cv2.resize(frame, dsize = vid_shape))

    return True, np.array(frames)

def load_virat_video(df, data_dir, col_names = VIRAT_COLS):
    # return dataframe, numpy video
    base_fname = df['fname'].loc[0].split('/')[-1][:-4]

    annotation_fname = f'{data_dir}/annotations/{base_fname}.viratdata.objects.txt'

    if not os.path.isfile(annotation_fname):
        print('annotation not found')
        return

    annotation_df = pd.read_csv(annotation_fname, delimiter=' ', header=None, names=col_names)
    annotation_df = annotation_df.loc[annotation_df['class_name'] != 0]

    max_time = df['iter'].max() # no min time
    annotation_df = annotation_df.loc[annotation_df['frame_num'] <= max_time]

    video_fname = f'{data_dir}/videos/{base_fname}.mp4'

    cap = cv2.VideoCapture(video_fname)
    success, frames = extract_frames(cap, None, max_time)

    if not success:
        print('no success')
        return

    return frames, annotation_df

def add_boxes(ax, xyxy_boxes, color):
    for box in xyxy_boxes:
        ax.add_patch(patches.Rectangle((box[0], box[1]), box[2]- box[0], box[3] - box[1], edgecolor = color, facecolor = "None"))

def display_frame(timestep : int, pred_df, frames, annotation_df, tracked_numbers):
    df_slice = annotation_df.loc[annotation_df['frame_num'] == timestep]

    if tracked_numbers is None:
        pred_df_slice = pred_df.loc[pred_df['iter'] == timestep]
        ns = get_all_tracked_objects(pred_df)
        tracked_numbers = []
        for x in ns:
            if np.isnan(pred_df_slice[f'p_d_{x}']) or pred_df_slice[f'p_d_{x}'] == 0:
                continue

            tracked_numbers.append(x)

    if isinstance(tracked_numbers, int):
        tracked_numbers = [tracked_numbers]

    x0s = list(df_slice['x'])
    y0s = list(df_slice['y'])
    ws = list(df_slice['w'])
    hs = list(df_slice['h'])
    x1s = [x0s[j] + ws[j] for j in range(len(x0s))]
    y1s = [y0s[k] + hs[k] for k in range(len(y0s))]

    bb_list = list(zip(x0s, y0s, x1s, y1s))

    gt_object_ids = list(df_slice['object_id'])

    object_boxes = {object_id : bb for object_id, bb in zip(bb_list, gt_object_ids)}

    _, ax = plt.subplots(1, 1, figsize=(10, 10))

    ax.imshow(frames[timestep])
    add_boxes(ax, [box for k, box in object_boxes.items() if k in tracked_numbers], color='g')
    add_boxes(ax, [box for k, box in object_boxes.items() if k in tracked_numbers], color='r')

    plt.show()

def get_nonnan_rows(df, col):
    return df[col].loc[~df[col].isnull()]

def get_group(grouped, group_name):
    return remove_empty_cols(grouped.get_group(group_name))

def get_time_columns(df):
    return [x for x in df if 'time' in x]

# 1. get the decay of accuracy as it goes
# split when the tracker is reset, ie. when 'reset_tracker' is True
def get_accuracies_of_objects(df, objs, prefix = 'p_d_'):
    '''get acc decay of each object, split by each reset_tracker'''
    acc_lists = {}
    for obj in objs:
        obj_name = f'{prefix}{obj}'
        acc_lists[obj] = [[]]
        for i, row in df.iterrows():
            # if nan, make new row if there was stuff before
            if np.isnan(row[obj_name]):
                if acc_lists[obj][-1]: # new row if there has been existing things
                    acc_lists[obj].append([])

            # if not nan either append or start new row based on bool value
            else:
                if not np.isnan(row['reset_tracker']) and row['reset_tracker']:
                    if acc_lists[obj][-1]: # new row if existing thing
                        acc_lists[obj].append([])
                else:
                    acc_lists[obj][-1].append(row[obj_name])

        # check the last acc_list if it's empty
        # if it is empty, remove it
        if not acc_lists[obj][-1]:
            acc_lists[obj] = acc_lists[obj][:-1]

        acc_lists[obj] = [np.array(x) for x in acc_lists[obj]]

    return acc_lists

def get_average_decay(acc_list : np.array) -> float:
    '''helper function for get_acc_decay'''
    if acc_list[0] in (0, -1):
        return -1

    if (acc_list <= 0).any():
        filtered_acc = acc_list[acc_list > 0]
        return acc_list[0] / (filtered_acc.shape[0] + 1)
    
    return (acc_list[0] - acc_list[-1])/acc_list.shape[0]

# postprocess former function
def get_acc_decay(acc_lists : {int : [np.array]}) -> {int : [[float]]}:
    '''get average decay in acc by object per reset_tracker'''
    return {obj_id : [get_average_decay(acc_list) for acc_list in v] for obj_id, v in acc_lists.items()}

def get_improvement(acc_lists : [np.array]) -> []:
    '''helper function for get_improvement_from_reset'''
    l = []
    old_low = 0
    for acc_list in acc_lists:
        l.append(acc_list[0] - old_low)
        old_low = acc_list[-1]

    return l

def get_improvement_from_reset(acc_lists : {int : [np.array]}) -> {int : [float]}:
    '''get accuracy improvement by object per reset_tracker'''
    return {obj_id : get_improvement(v) for obj_id, v in acc_lists.items()}

def get_objs_in_row(df, i, all_objs, format_obj) -> {int}:
    gen_objs = set()
    high_objs = set()
    for obj_id in all_objs:
        curr_acc = df[format_obj(obj_id)].iloc[i]
        if not np.isnan(curr_acc):
            gen_objs.add(obj_id)
            if curr_acc > 0.5:
                high_objs.add(obj_id)

    return gen_objs, high_objs

def get_tracked_diff_from_reset(df : pd.DataFrame, all_objs, format_objs) -> [int]:
    '''get uptick in every num_high_objects during a reset_tracker
    also get decay in num_high_objects between reset_trackers
    2 things for decay in num_high_objects'''
    decay_missed_arr = []
    decay_lost_arr = []
    improvement_arr = []
    old_high = df['num_high_objects'].iloc[0]
    old_reset_index = 0
    for reset_index in get_nonnan_rows(df, 'reset_tracker').index:
        old_gen_objs, old_high_objs = get_objs_in_row(df, old_reset_index, all_objs, format_objs)
        gen_objs, high_objs = get_objs_in_row(df, reset_index-1, all_objs, format_objs)

        # objects that are in old_high but not in high_objs are "lost"
        if not high_objs.issubset(old_gen_objs):
            print(f'some error that the new high_objs is not in the object subset: {old_reset_index}, {reset_index}, {high_objs}, {old_gen_objs}')

        decay_lost_arr.append(len(old_high_objs) - len(high_objs))

        # objects that are in gen_objs but not in old_gen_objs are "missing"
        x=0
        for obj in gen_objs:
            if obj not in old_gen_objs:
                x += 1

        decay_missed_arr.append(x)

        old_reset_index = reset_index

        improvement_arr.append(df['num_high_objects'].iloc[reset_index] - df['num_high_objects'].iloc[reset_index-1])
        old_high = df['num_high_objects'].iloc[reset_index]

    return np.array(decay_missed_arr), np.array(decay_lost_arr), np.array(improvement_arr)

def get_first_instance_unhigh(df, col_to_check):
    df = df.fillna(0)
    try:
        x = df[df[col_to_check] < 0.5].index[0] - df.index[0]
    except:
        print(df)
        raise AssertionError

    return x

def iters_until_lost(df, all_objs, format_objs) -> {int : [int]}:
    # for every object that gets "lost", count the number of iters until it does so
    old_reset_index = 0
    lost_dict = defaultdict(list)

    for reset_index in get_nonnan_rows(df, 'reset_tracker').index:
        old_gen_objs, old_high_objs = get_objs_in_row(df, old_reset_index, all_objs, format_objs)
        gen_objs, high_objs = get_objs_in_row(df, reset_index - 1, all_objs, format_objs)

        # lost are the ones in old high but not in high
        for old_obj in old_high_objs:
            if old_obj not in high_objs:
                # get the iter at which it peers below 0.5
                lost_dict[old_obj].append(get_first_instance_unhigh(df[old_reset_index : reset_index], format_objs(old_obj)))

        old_reset_index = reset_index

    return lost_dict

def process_dataframe(df, missing_preds = True):
    # first get the acc deltas
    get_num_tracked(df)
    objs = get_all_tracked_objects(df)
    if missing_preds:
        df = count_missing_preds(df)

    acc_lists = get_accuracies_of_objects(df, objs, prefix='p_s_')

    acc_decay = get_acc_decay(acc_lists)
    acc_improvement = get_improvement_from_reset(acc_lists)
    tracked_missed, tracked_lost, tracked_improvement = get_tracked_diff_from_reset(df, objs, lambda x : f'p_s_{x}')
    lost_iters = iters_until_lost(df, objs, lambda x : f'p_s_{x}')

    for time_col in get_time_columns(df):
        print(f'{time_col}\t : {df[time_col].mean()}')

    return df, objs, acc_decay, acc_improvement, tracked_missed, tracked_lost, tracked_improvement, lost_iters