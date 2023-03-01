from datetime import datetime

# constants, COCO constants defined below
LEVELS = {'DEBUG' : 0, 'INFO' : 1, 'ERROR' : 2, 'NOTHING' : 3}
# KITTI classes (1,2,3) match COCO, ignore Vans

CURR_DATE = datetime.now().strftime('%m-%d_%H:%M:%S')
DESIRED_CLASSES = {1,2,3}

PARAMS = {}
# in the offline case, client.py will have the server model as well
# otherwise
PARAMS['USE_NETWORK'] = False
PARAMS['HOST'] = '128.195.54.126' # localhost is '127.0.0.1', network host to use is '128.195.54.126'
PARAMS['PORT'] = 1234
PARAMS['SOCK_BUFFER_READ_SIZE'] = 16384

# params for directories + misc things
PARAMS['STATS_LOG_DIR'] = 'logs'
PARAMS['DATE_FORMAT'] = '%m/%d %H:%M:%S'
PARAMS['LOGGING_LEVEL'] = LEVELS['DEBUG']
PARAMS['FLUSH_LIMIT'] = 100
PARAMS['DEV_DIR'] = 'dev'
PARAMS['DATA_DIR'] = 'data'
PARAMS['LOG_STATS'] = True

# params for the dataset – used for classical compression methods
PARAMS['DATASET'] = 'kitti'
PARAMS['FRAME_LIMIT'] = 15
PARAMS['FPS'] = 30.
PARAMS['VIDEO_SHAPE'] = (1280,720)
# params for individual datasets
PARAMS['DAVIS_SCENES'] = [1,2,3,4,5,6,7,8,9,10]

# params for model yamls and paths
PARAMS['ENTROPIC_DIR'] = 'Models/entropic'
PARAMS['FASTER_RCNN_YAML'] = 'configs/coco2017/supervised_compression/entropic_student/faster_rcnn_splittable_resnet50-fp-beta0.08_fpn_from_faster_rcnn_resnet50_fpn.yaml'
PARAMS['DEEPLABV3_YAML'] = 'configs/pascal_voc2012/supervised_compression/entropic_student/deeplabv3_splittable_resnet50-fp-beta0.16_from_deeplabv3_resnet50.yaml'
# note: unusable due to lack of JIT
PARAMS['CLIENT_MODEL_PATH'] = 'Models/Split/model_2_client.pt'
PARAMS['SERVER_MODEL_PATH'] = 'Models/Split/model_2_client.pt'

# params for run type
PARAMS['RUN_TYPE'] = 'BB' # 'bounding box' (BB) vs 'segmentation mask' (SM) vs segmask after bb 'BBSM'
PARAMS['EVAL'] = True
PARAMS['BOX_LIMIT'] = 10 # max number of boxes to track, only applicable if EVAL is false

# params for the tracking
PARAMS['TRACKING'] = True # execute + evaluate a tracking algorithm; if false, evaluates the detector
PARAMS['TRACKER'] = 'MEDIANFLOW' # tracker algorithm
PARAMS['BBOX_SEG'] = None
PARAMS['CATCHUP_LIMIT'] = 15

# params for the object detection / segmentation
PARAMS['TASK'] = 'det' # det, seg, gt; gt means no predictions will happen
# TODO: combine run_type and task

# params for models
PARAMS['COMPRESSOR'] = 'model' # 'model' vs 'classical'
PARAMS['COMPRESSOR_DEVICE'] = 'cpu'
PARAMS['SERVER_DEVICE'] = 'cpu'
PARAMS['SMARTDET'] = True # parallelized detection/segmentation
PARAMS['MODEL_NAME'] = 'deeplabv3' # specific detection / segmentation model
PARAMS['STUDENT_YAML'] = PARAMS[f'{PARAMS["MODEL_NAME"].upper()}_YAML'] # yaml file for loading the student model if split

# params for detection 'refreshes'
PARAMS['BOX_REFRESH'] = 'fixed' # method to refresh bbox
PARAMS['REFRESH_ITERS'] = 10 # for fixed method, how many fixed iterations to refresh bb; setting iters to 1 makes detection run 100%

# params for when client requests files from the server
PARAMS['FILE_TRANSFER'] = False
PARAMS['DATA_PICKLE_FILES'] = None # pickle of a list of batches of fnames (ie fnames in the format of 'VIRAT/videos/...')
PARAMS['FILE_BATCH_SIZE'] = 5

try:
  from param_overrides import PARAM_OVERRIDES
  print('USING PARAM OVERRIDES')
  PARAMS.update(PARAM_OVERRIDES)
except ImportError:
  print('NO PARAM OVERRIDES DETECTED')
  pass

VIRAT_COLS = ['object_id', 'duration', 'frame_num', 'x', 'y', 'w', 'h', 'class_name']

VIRAT_CLASSES = {
 1:'person',
 2:'car',
 3:'vehicles',
 4:'object',
 5:'bike'
}

VIRAT_COCO_MAP = {
 1: 1,
 2: 3,
 3: 3,
 4: 0,
 5: 2
}

KITTI_COLS = ['timestep', 'object_i', 'class_name', '_1', '_2', '_3', 'x0', 'y0', 'x1', 'y1', '_4', '_5', '_6', '_7', '_8', '_9', '_10']

KITTI_CLASSES = {
 'DontCare' : -1,
 'Van': 0,
 'Cyclist' : 2,
 'Pedestrian' : 1,
 'Car' : 3,
 'Misc' : 4,
 'Truck' : 8,
 'Tram' : 10,
 'Person' : 1
}

COCO_CLASS_DICT = {
 0: u'__background__',
 1: u'person',
 2: u'bicycle',
 3: u'car',
 4: u'motorcycle',
 5: u'airplane',
 6: u'bus',
 7: u'train',
 8: u'truck',
 9: u'boat',
 10: u'traffic light',
 11: u'fire hydrant',
 12: u'stop sign',
 13: u'parking meter',
 14: u'bench',
 15: u'bird',
 16: u'cat',
 17: u'dog',
 18: u'horse',
 19: u'sheep',
 20: u'cow',
 21: u'elephant',
 22: u'bear',
 23: u'zebra',
 24: u'giraffe',
 25: u'backpack',
 26: u'umbrella',
 27: u'handbag',
 28: u'tie',
 29: u'suitcase',
 30: u'frisbee',
 31: u'skis',
 32: u'snowboard',
 33: u'sports ball',
 34: u'kite',
 35: u'baseball bat',
 36: u'baseball glove',
 37: u'skateboard',
 38: u'surfboard',
 39: u'tennis racket',
 40: u'bottle',
 41: u'wine glass',
 42: u'cup',
 43: u'fork',
 44: u'knife',
 45: u'spoon',
 46: u'bowl',
 47: u'banana',
 48: u'apple',
 49: u'sandwich',
 50: u'orange',
 51: u'broccoli',
 52: u'carrot',
 53: u'hot dog',
 54: u'pizza',
 55: u'donut',
 56: u'cake',
 57: u'chair',
 58: u'couch',
 59: u'potted plant',
 60: u'bed',
 61: u'dining table',
 62: u'toilet',
 63: u'tv',
 64: u'laptop',
 65: u'mouse',
 66: u'remote',
 67: u'keyboard',
 68: u'cell phone',
 69: u'microwave',
 70: u'oven',
 71: u'toaster',
 72: u'sink',
 73: u'refrigerator',
 74: u'book',
 75: u'clock',
 76: u'vase',
 77: u'scissors',
 78: u'teddy bear',
 79: u'hair drier',
 80: u'toothbrush'}

PASCAL_CLASS_DICT = {
 1: u'aeroplane',
 2: u'bicycle',
 3: u'bird',
 4: u'boat',
 5: u'bottle',
 6: u'bus',
 7: u'car ',
 8: u'cat',
 9: u'chair',
 10: u'cow',
 11: u'diningtable',
 12: u'dog',
 13: u'horse',
 14: u'motorbike',
 15: u'person',
 16: u'potted plant',
 17: u'sheep',
 18: u'sofa',
 19: u'train',
 20: u'tv/monitor',
}

PASCAL_COCO_MAP = {
 1: 5,
 2: 2,
 3: 15,
 4: 9,
 5: 40,
 6: 6,
 7: 3,
 8: 16,
 9: 57,
 10: 20,
 11: 61,
 12: 17,
 13: 18,
 14: 4,
 15: 1,
 16: 59,
 17: 19,
 18: 100, # sofa is unmatched
 19: 7,
 20: 63,
}

# TODO: implement the rest of the classes
# len is the num of objects, index is the obj class in pascal
DAVIS_PASCAL_MAP = {
 'bear': (0,12,), # bear is most similar to dog class
 'bike-packing': (0,2,15),
 'blackswan': (0,3,),
 'bmx-bumps': (0,15,14),
 'bmx-trees': (0,15,2),
 'boat': (0,4,),
 'boxing-fisheye': (0,15,),
 'breakdance': (0,15,),
}

ILSVRC_COCO_MAP = {

}