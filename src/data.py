from utils2 import *
import torch
from Logger import ConsoleLogger
from PIL import Image
import pandas as pd
from params import *
import time

class Dataset:
    def __init__(self, data_dir = PARAMS['DATA_DIR'], dataset = PARAMS['DATASET'], fps = PARAMS['FPS']):
        self.data_dir = data_dir
        self.logger = ConsoleLogger()
        self.simulated_fps = fps
        if dataset == 'latency':
            self.dataset = self.get_toy_dataloader()
        elif dataset == 'framelen':
            self.dataset = self.get_framelen_dataset()
        elif dataset == 'bdd':
            self.dataset = self.get_bdd_dataset()
        elif dataset == 'virat':
            self.dataset = self.get_virat_dataset()
        elif dataset == 'yc2':
            self.dataset = self.get_youcook2_dataset()
        elif dataset == 'phone':
            self.dataset = self.get_phone_dataset()
        elif dataset == 'kitti_toy':
            self.dataset = self.get_model_toy_dataset()
        elif dataset == 'kitti':
            self.dataset = self.get_kitti_dataset()
        elif dataset == 'davis':
            self.dataset = self.get_davis_dataset()
        else:
            raise NotImplementedError('No Dataset Found.')

    def get_dataset(self):
        '''in the video case, returns (byte_encoding, (frames_size, encoding_size), full_fname)
        for object detection, returns (image, orig_size, ((class, object_index),), (gt_boxes,), full_fname, new_video_bool)'''
        return self.dataset

    def open_fname(self, fname, cap = None, codec = 'avc1', frame_limit = PARAMS['FRAME_LIMIT'], shape = PARAMS['VIDEO_SHAPE'],
                   transpose = False):
        '''returns Success, (encoding as bytes, (original_size, size of encoding))'''
        self.logger.log_debug(f'Handling {fname}')
        if not cap:
            cap = cv2.VideoCapture(fname)

        success, frames = extract_frames(cap, frame_limit=frame_limit, transpose_frame=transpose, vid_shape=shape)

        if not success:
            self.logger.log_debug(f'OpenCV Failed on file {fname}')
            return False, None

        frames_byte_size = sys.getsizeof(frames) # frames should be in shape
        byte_frames = return_frames_as_bytes(frames, codec=codec)

        return True, (byte_frames, (frames_byte_size, len(byte_frames)))

    def get_toy_dataloader(self):
        # testing latency
        # 100 arrays split by intervals of 10k: [10k, ..., 1mil]
        # 30 trials each
        array_lens = np.arange(int(1e4), int(1e6 + 1e4), int(1e4), dtype=int)
        for array_len in array_lens:
            arr = np.zeros((array_len,), dtype=np.int64)
            arr_byte_len = sys.getsizeof(arr)
            for trial in range(30):
                yield arr, (arr_byte_len, arr_byte_len)

    def get_framelen_dataset(self):
        '''test framelen vs compression ratio
        using virat dataset because it is the most stable'''
        # 2 ->
        virat_dir = f'{self.data_dir}/VIRAT'
        fnames = [fname for fname in os.listdir(virat_dir) if
                  '.mp4' in fname]  # remove .DS_Store and other misc files

        frame_lens = range(2, 62, 2)
        for frame_len in frame_lens:
            for fname in fnames: # 10 videos inside virat
                full_fname = f'{virat_dir}/{fname}'
                cap = cv2.VideoCapture(full_fname)

                for i in range(10):  # load 10 random frames / video ==> 100 samples / frame length
                    ret, byte_info = self.open_fname(full_fname, cap=cap, frame_limit=frame_len)
                    if not ret:
                        continue

                    byte_encoding, (frames_size, encoding_size) = byte_info

                    yield byte_encoding, (frames_size, encoding_size), full_fname

    def get_bdd_dataset(self):
        fnames = [fname for fname in os.listdir(f'{self.data_dir}/bdd100k/videos/test') if '.mov' in fname] # remove .DS_Store and other misc files

        num_success = 0
        for fname in fnames:
            full_fname = f'{self.data_dir}/bdd100k/videos/test/{fname}'
            cap = cv2.VideoCapture(full_fname)

            for i in range(5):
                if num_success >= 700:  # 700 files x 5 iters/file = 7000 trials
                    break

                ret, byte_info = self.open_fname(full_fname, cap=cap)
                if not ret:
                    continue

                byte_encoding, (frames_size, encoding_size) = byte_info
                num_success+=1

                yield byte_encoding, (frames_size, encoding_size), full_fname

    def get_virat_dataset(self):
        virat_dir = f'{self.data_dir}/VIRAT'
        fnames = [fname for fname in os.listdir(virat_dir) if
                  '.mp4' in fname]  # remove .DS_Store and other misc files
        for fname in fnames:
            full_fname = f'{virat_dir}/{fname}'
            cap = cv2.VideoCapture(full_fname)

            for i in range(20): # load 20 random frames / video
                ret, byte_info = self.open_fname(full_fname, cap=cap)
                if not ret:
                    continue

                byte_encoding, (frames_size, encoding_size) = byte_info

                yield byte_encoding, (frames_size, encoding_size), full_fname

    def get_youcook2_dataset(self):
        yc2_dir = f'{self.data_dir}/YouCook2/raw_videos/validation'
        dirs = [f'{yc2_dir}/{x}' for x in sorted(os.listdir(yc2_dir)) if len(x)==3] # 3 digit codes usually, this is hardcoded but gets past .DS_Store files
        fnames = []
        for dir in dirs:
            fnames += [f'{dir}/{x}' for x in sorted(os.listdir(dir)) if '.webm' in x]

        for fname in fnames:
            cap = cv2.VideoCapture(fname)

            for i in range(70):
                ret, byte_info = self.open_fname(fname, cap=cap)
                if not ret:
                    continue

                byte_encoding, (frames_size, encoding_size) = byte_info

                yield byte_encoding, (frames_size, encoding_size), fname

    def get_phone_dataset(self):
        p_dir = f'{self.data_dir}/Phone'
        fnames = [x for x in os.listdir(p_dir) if '.MOV' in x]

        for fname in fnames:
            full_fname = f'{p_dir}/{fname}'
            cap = cv2.VideoCapture(full_fname)

            for i in range(60):
                ret, byte_info = self.open_fname(f'{self.data_dir}/Phone/{fname}', cap=cap, transpose=True)
                if not ret:
                    continue

                byte_encoding, (frames_size, encoding_size) = byte_info

                yield byte_encoding, (frames_size, encoding_size), full_fname


    def get_model_toy_dataset(self, col_names=PARAMS['KITTI_NAMES']):
        # uses the KITTI dataset for latency evals
        # returns in the format (img, img_size, (class, obj_id), bb_list, fname, frame_number)
        data_dir = f'{self.data_dir}/KITTI/data_tracking_image_2/training'
        videos = sorted([f'{data_dir}/image_02/{x}' for x in os.listdir(f'{data_dir}/image_02') if x.isnumeric()]) #dddd for video id
        video_labels = sorted([f'{data_dir}/label_02/{x}' for x in os.listdir(f'{data_dir}/label_02') if x[:-4].isnumeric()]) #dddd.txt for labels

        assert len(videos) == len(video_labels), f'{videos}, {video_labels}'

        time_per_frame = 1/self.simulated_fps
        SHAPES_TO_TEST = [(100, 200), (200, 400), (400, 600), (600, 800), (720, 1280), (1000, 1500)]

        for trial in range(50):
            for i, video in enumerate(videos):
                frames = sorted([f'{video}/{x}' for x in os.listdir(video) if '.jpg' in x or '.png' in x])
                label_df = pd.read_csv(video_labels[i], delimiter = ' ', header=None, names=col_names)
                time_since_previous_frame = time.time()
                for shape in SHAPES_TO_TEST:
                    for j, fname in enumerate(frames):
                        while time.time() - time_since_previous_frame < time_per_frame:
                            time.sleep(0.005)

                        time_since_previous_frame = time.time()

                        img = Image.open(fname)
                        if shape is None:
                            img = np.array(img)
                        else:
                            img = np.array(img.resize(shape))

                        df_slice = label_df.loc[label_df['timestep'] == j]

                        # get the object indices
                        object_ids = list(df_slice['object_i'])
                        # get class names
                        classes = list(df_slice['class_name'])
                        classes_id = [KITTI_CLASSES[x] for x in classes]

                        # create list of bounding boxes
                        x0s = list(df_slice['x0'])
                        y0s = list(df_slice['y0'])
                        x1s = list(df_slice['x1'])
                        y1s = list(df_slice['y1'])

                        bb_list = list(zip(x0s, y0s, x1s, y1s))

                        yield img, sys.getsizeof(img), (classes_id, object_ids), bb_list, fname, j == 0

                break

    def get_kitti_dataset(self, col_names=PARAMS['KITTI_NAMES'], shape=PARAMS['VIDEO_SHAPE']):
        # returns in the format (img, img_size, (class, obj_id), bb_list, fname, frame_number)
        data_dir = f'{self.data_dir}/KITTI/data_tracking_image_2/training'
        # sort for stability
        videos = sorted([f'{data_dir}/image_02/{x}' for x in os.listdir(f'{data_dir}/image_02') if
                         x.isnumeric()])  # dddd for video id
        video_labels = sorted([f'{data_dir}/label_02/{x}' for x in os.listdir(f'{data_dir}/label_02') if
                               x[:-4].isnumeric()])  # dddd.txt for labels

        assert len(videos) == len(video_labels), f'{videos}, {video_labels}'

        time_per_frame = 1 / self.simulated_fps

        for i, video in enumerate(videos):
            frames = sorted([f'{video}/{x}' for x in os.listdir(video) if '.jpg' in x or '.png' in x])
            label_df = pd.read_csv(video_labels[i], delimiter=' ', header=None, names=col_names)
            time_since_previous_frame = time.time()
            for j, fname in enumerate(frames):
                while time.time() - time_since_previous_frame < time_per_frame:
                    time.sleep(0.005)

                time_since_previous_frame = time.time()

                img = Image.open(fname)
                if shape is None:
                    img = np.array(img)
                else:
                    img = np.array(img.resize(shape))

                df_slice = label_df.loc[label_df['timestep'] == j]

                # get the object indices
                object_ids = list(df_slice['object_i'])
                # get class names
                classes = list(df_slice['class_name'])
                classes_id = [KITTI_CLASSES[x] for x in classes]

                # create list of bounding boxes
                x0s = list(df_slice['x0'])
                y0s = list(df_slice['y0'])
                x1s = list(df_slice['x1'])
                y1s = list(df_slice['y1'])

                bb_list = list(zip(x0s, y0s, x1s, y1s))

                yield img, sys.getsizeof(img), (classes_id, object_ids), bb_list, fname, j == 0

    def get_davis_dataset(self, shape=PARAMS['VIDEO_SHAPE'], frame_limit = PARAMS['FRAME_LIMIT']):
        data_dir = f'{self.data_dir}/DAVIS'
        video_categories = os.listdir(f'{data_dir}/JPEGImages/Full-Resolution')

        time_per_frame = 1 / self.simulated_fps

        for video_cat in video_categories:
            assert len(os.listdir(f'{data_dir}/JPEGImages/Full-Resolution/{video_cat}')) == \
                   len(os.listdir(f'{data_dir}/Annotations/Full-Resolution/{video_cat}'))
            vid_frame_nums = sorted([x for x in os.listdir(f'{data_dir}/JPEGImages/Full-Resolution/{video_cat}')
                                     if '.jpg' in x or '.png' in x])
            video_frames = [f'{data_dir}/JPEGImages/Full-Resolution/{video_cat}/{x}' for x in vid_frame_nums]

            mask_frame_nums = sorted([x for x in os.listdir(f'{data_dir}/Annotations/Full-Resolution/{video_cat}')
                                      if '.jpg' in x or '.png' in x])
            mask_frames = [f'{data_dir}/Annotations/Full-Resolution/{video_cat}/{x}' for x in mask_frame_nums]

            obj_classes = DAVIS_PASCAL_MAP[video_cat] # ordered in terms of RGB
            # obj_ids = tuple(range(len(obj_classes) + 1))

            time_since_previous_frame = time.time()
            for i in range(len(video_frames)):
                while time.time() - time_since_previous_frame < time_per_frame:
                    time.sleep(0.005)

                time_since_previous_frame = time.time()

                vid_frame = Image.open(video_frames[i])
                mask_frame = Image.open(mask_frames[i])
                if shape is None:
                    vid_frame = np.array(vid_frame)
                    mask_frame = np.array(mask_frame)
                else:
                    vid_frame = np.array(vid_frame.resize(shape))
                    mask_frame = np.array(mask_frame.resize(shape))

                assert len(mask_frame.shape) == 2

                obj_ids = np.unique(mask_frame)
                # mask_ids = np.zeros_like(mask_frame)

                yield vid_frame, sys.getsizeof(vid_frame), (obj_classes, obj_ids), mask_frame, video_frames[i], i==0

            break