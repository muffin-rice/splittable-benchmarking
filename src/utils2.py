import os
import sys
import cv2
import numpy as np
from params import PARAMS, DESIRED_CLASSES
import torch
from copy import deepcopy
from skimage import color
from scipy.optimize import linear_sum_assignment

def default_detection_postprocessor(d):
    return

def load_model(model_config, device):
    if 'detection_model' not in model_config:
        from sc2bench.models.detection.registry import load_detection_model
        return load_detection_model(model_config, device)
    from sc2bench.models.detection.wrapper import get_wrapped_detection_model
    return get_wrapped_detection_model(model_config, device)

def get_student_model(yaml_file = PARAMS['STUDENT_YAML']):
    from torchdistill.common.yaml_util import load_yaml_file
    if yaml_file is None:
        return None

    config = load_yaml_file(os.path.expanduser(yaml_file))
    models_config = config['models']

    if PARAMS['MODEL_NAME'] == 'deeplabv3':
        models_config['student_model']['ckpt'] = \
            f'{PARAMS["ENTROPIC_DIR"]}/pascal_voc2012-deeplabv3_splittable_resnet50-fp-beta0.16_from_deeplabv3_resnet50.pt'
        models_config['student_model']['params']['backbone_config']['ckpt'] = \
            f'{PARAMS["ENTROPIC_DIR"]}/ilsvrc2012-splittable_resnet50-fp-beta0.16_from_resnet50.pt'
    elif PARAMS['MODEL_NAME'] == 'faster_rcnn':
        models_config['student_model']['ckpt'] = \
            f'{PARAMS["ENTROPIC_DIR"]}/coco2017-faster_rcnn_splittable_resnet50-fp-beta0.08_fpn_from_faster_rcnn_resnet50_fpn.pt'
        models_config['student_model']['params']['backbone_config']['ckpt'] = \
            f'{PARAMS["ENTROPIC_DIR"]}/ilsvrc2012-splittable_resnet50-fp-beta0.08_from_resnet50.pt'

    student_model_config = models_config['student_model'] if 'student_model' in models_config else models_config[
        'model']
    student_model = load_model(student_model_config, PARAMS['SERVER_DEVICE']).eval()

    return student_model

def get_tensor_size(tensor) -> int:
    # "primitive" types
    if tensor is None:
        return 0
    if type(tensor) is bytes:
        return len(tensor)
    if 'QuantizedTensor' in str(type(tensor)):
        return tensor.tensor.storage().__sizeof__()
    if 'Tensor' in str(type(tensor)):
        return tensor.storage().__sizeof__()

    if type(tensor) is dict:
        return sum(get_tensor_size(x) for x in tensor.values())

    if type(tensor) in (list, tuple):
        return sum(get_tensor_size(x) for x in tensor)

    return sys.getsizeof(tensor)

def move_data_list_to_device(data : (), device):
    new_data = []
    for d in data:
        if ('Tensor' in str(type(d))) or ('tensor' in str(type(d))):
            new_data.append(d.to(device))
        else:
            new_data.append(d)

    return new_data

def move_data_dict_to_device(data : {}, device):
    new_data = {}
    for k, v in data.items():
        if ('Tensor' in str(type(v))) or ('tensor' in str(type(v))):
            new_data[k] = v.to(device)
        else:
            new_data[k] = v

    return new_data

def encode_frame(frame : np.ndarray):
    '''uses cv2.imencode to encode a frame'''
    if frame.shape[2] != 3:
        frame = frame.transpose((1,2,0))

    if frame.dtype != 'uint8':
        frame *= 256
        # frame.astype('uint8')

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    success, frame_bytes = cv2.imencode('.jpg', frame, encode_param)
    if not success:
        raise ValueError('Encoding Failed')

    return frame_bytes

def decode_frame(encoded_frame):
    '''decodes the encode_frame and returns it as a float array (between 0 and 1) and 3xHxW'''
    return cv2.imdecode(encoded_frame, cv2.IMREAD_COLOR).transpose(2,0,1) / 256

def extract_frames(cap, vid_shape = (720, 1280), frame_limit=150, transpose_frame = False) -> (bool, np.ndarray):
    '''From a cv2 VideoCapture, return a stack of frame_limit subset of the video'''
    # get 15 frames from random starting point
    frame_limit = min(cap.get(7), frame_limit)

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
def return_frames_as_bytes(frames : np.ndarray, temp_dir = PARAMS['DEV_DIR'], codec='avc1', fps = PARAMS['FPS'],
                           shape = PARAMS['VIDEO_SHAPE'], frame_limit = PARAMS['FRAME_LIMIT']) -> bytes:
    '''From a numpy array of frames (nframes x h x w x 3), return the compressed video (with a specific codec) as bytes'''
    temp_fname = f'{temp_dir}/temp_{codec}.mp4'
    fourcc = cv2.VideoWriter_fourcc(*codec)

    out = cv2.VideoWriter(temp_fname, fourcc, fps, shape)
    for i in range(frame_limit):
        out.write(frames[i, ...])
    out.release()

    with open(temp_fname, 'rb') as f:
        byte_image = f.read()

    os.remove(temp_fname)

    return byte_image

def decode_bytes(byte_video, temp_dir = PARAMS['DEV_DIR']) -> cv2.VideoCapture:
    '''From bytes, return a cv2.VideoCapture'''
    temp_fname = f'{temp_dir}/temp.mp4'
    with open(temp_fname, 'wb') as f:
        f.write(byte_video)

    cap = cv2.VideoCapture(temp_fname)
    os.remove(temp_fname)

    return cap

def calculate_bb_iou(boxA, boxB):
    '''calculates the iou for x0y0x1y1 format, singular box, numpy'''
    # determine the (x, y)-coordinates of the intersection rectangle

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def calculate_mask_iou(maskA, maskB):
    intersection = np.logical_and(maskA, maskB).sum()
    union = np.logical_or(maskA, maskB).sum()

    return intersection / union

def map_xyxy_to_xyhw(xyxy_box):
    return np.array((xyxy_box[0], xyxy_box[1], xyxy_box[2] - xyxy_box[0], xyxy_box[3] - xyxy_box[1]))

def map_xyhw_to_xyxy(xyhw_box):
    xyxy_box = np.array((xyhw_box[0], xyhw_box[1], xyhw_box[2] + xyhw_box[0], xyhw_box[3] + xyhw_box[1]))
    if xyxy_box[2] < xyxy_box[0]:
        xyxy_box[0], xyxy_box[2] = xyxy_box[2], xyxy_box[0]

    if xyxy_box[3] < xyxy_box[1]:
        xyxy_box[1], xyxy_box[3] = xyxy_box[3], xyxy_box[1]

    return xyxy_box

def map_coco_outputs(outputs : {str : torch.Tensor}) -> ({int : {int : (int,)}}, [float]):
    '''Maps the model output (dict with keys boxes, labels, scores) to {class_label : {id : boxes}}'''
    boxes = outputs['boxes'].detach().numpy()
    labels = outputs['labels'].detach().numpy()
    scores = outputs['scores'].detach().numpy()
    # ignore scores for now

    high_scores = scores>0.4

    boxes, labels, scores = boxes[high_scores], labels[high_scores], scores[high_scores]

    d = {}
    curr_id = 0
    for i in range(labels.shape[0]):
        class_id = int(labels[i])
        if class_id not in DESIRED_CLASSES:
            continue

        if class_id in d:
            d[class_id][curr_id] = tuple(boxes[i])
        else:
            d[class_id] = {curr_id : tuple(boxes[i])}

        curr_id+=1

    return d, scores

def map_indexes(gt_objects : {int : any}, pred_objects : {int : any}, iou_calculator):
    index_mapping = {}
    if len(gt_objects) < len(pred_objects):
        for id_gt, gt_pred in gt_objects.items():  # gt boxes on the outside
            highscore, highj = -1, -1
            for id_pred, pred_box in pred_objects.items():
                if id_pred in index_mapping:
                    continue

                curr_score = iou_calculator(gt_pred, pred_box)

                if curr_score >= highscore:
                    highscore, highj, = curr_score, id_pred

            index_mapping[highj] = id_gt
    else:
        used_gt = set()
        for id_pred, pred in pred_objects.items():  # gt boxes on the outside
            highscore, highj = -1, -1
            for id_gt, gt_box in gt_objects.items():
                if id_gt in used_gt:
                    continue

                curr_score = iou_calculator(gt_box, pred)

                if curr_score >= highscore:
                    highscore, highj, = curr_score, id_gt

            index_mapping[id_pred] = highj
            used_gt.add(highj)

    return index_mapping

def map_bbox_ids(pred_boxes_allclass : {int : {int : (int,)}}, gt_boxes_allclass : {int : (int,)}) -> ({int: int}, {int: int}):
    '''maps indices of the predicted boxes to the ids of the gt boxes and returns the extraneous items
    input is in the format {class_id : {object_id : bbox_xyxy}}
    return dict in the form of {pred_index : gt_id} and {class_id : len(gt) - len(pred)}'''
    index_mapping = {}

    missing_objects = {} # for every class id, >0 means in gt but not in pred, <0 means in pred but not in gt
    unioned_classes = set(gt_boxes_allclass.keys()).union(set(pred_boxes_allclass.keys()))

    for class_id in unioned_classes:
        if class_id not in pred_boxes_allclass:
            missing_objects[class_id] = len(gt_boxes_allclass[class_id])
            continue

        if class_id not in gt_boxes_allclass:
            missing_objects[class_id] = -len(pred_boxes_allclass[class_id])
            continue

        gt_boxes, pred_boxes = gt_boxes_allclass[class_id], pred_boxes_allclass[class_id]
        missing_objects[class_id] = len(gt_boxes) - len(pred_boxes)

        index_mapping.update(map_indexes(gt_boxes, pred_boxes, calculate_bb_iou))

    return index_mapping, missing_objects

def remove_classes_from_pred(preds_with_classes : {int : {}}, add_clause = None, return_clause = None):
    '''returns the {object_id : pred} detections from a pred_with_classes and object_mapping {object_id : self}'''
    if not add_clause:
        add_clause = lambda x : True
    if not return_clause:
        return_clause = lambda x : False

    detections = {}
    object_id_mapping = {}
    for class_id, class_detection in preds_with_classes.items():
        for object_id, pred in class_detection.items():
            object_id_mapping[object_id] = object_id
            if add_clause(object_id):
                detections[object_id] = pred

            if return_clause(detections):
                return detections


    return detections, object_id_mapping

def separate_objects_in_mask(mask, num_objects, starting_id = 1, num_iters = 10) -> {int : np.array}:
    '''given a binary mask cluster the objects given the numobjects
    returns random id : mask'''
    assert len(np.unique(mask)) == 2, f'Mask is not binary: {mask}, {np.unique(mask)}'
    if num_objects == 1:
        return {starting_id : mask}

    mask_indices = np.array(tuple(x for x in zip(*np.where(mask))))
    start_inds = np.random.choice(np.arange(mask_indices.shape[0]), num_objects, replace=False)
    cluster_starts = mask_indices[start_inds] # num_objects x 2
    index_mask = np.indices((mask.shape[0], mask.shape[1])).transpose(1,2,0)

    assert num_iters > 1

    for i in range(num_iters):
        mask_distances = np.zeros((num_objects, mask.shape[0], mask.shape[1]))
        for j, cluster in enumerate(cluster_starts):
            mask_distances[j, :, :] = np.linalg.norm(index_mask - cluster, ord=2)

        # cluster mins is H x W array of the argmin
        cluster_mins = np.argmin(mask_distances, axis = 0) + 1
        cluster_mins = cluster_mins[mask]

        # find the average of the centroids
        for j in range(cluster_starts.shape[0]):
            cluster_starts[j, :] = np.mean(index_mask[cluster_mins == j+1], axis=[0,1])

    return {starting_id + i : cluster_mins[cluster_mins == i] for i in range(num_objects)}


def map_mask_ids(pred_mask_allclasses : {int : np.array}, gt_mask_classes : {int : {int : np.array}}):
    '''changes the mask id prediction itself, and returns the index mapping and missing detections
    see map_bbox_ids'''
    unioned_classes = set(pred_mask_allclasses.keys()).union(gt_mask_classes.keys())
    index_mapping = {}
    missing_objects = {}
    starting_object_id = 1

    for class_id in unioned_classes:
        assert class_id in pred_mask_allclasses
        assert 'array' in str(
            type(pred_mask_allclasses[class_id])), f'this should be an array: {pred_mask_allclasses[class_id]}'

        if not pred_mask_allclasses[class_id].any():
            # mask does not predict anything; either it's not there or missing pred
            pred_mask_allclasses.pop(class_id) # remove prediction
            if class_id in gt_mask_classes:
                missing_objects[class_id] = len(gt_mask_classes[class_id])

            continue

        if class_id not in gt_mask_classes:
            pred_mask_allclasses[class_id] = {-1 : pred_mask_allclasses[class_id]}
            missing_objects[class_id] = -len(pred_mask_allclasses[class_id])
            continue

        pred_mask_allclasses[class_id] = separate_objects_in_mask(pred_mask_allclasses[class_id],
                                                                  len(gt_mask_classes[class_id]),
                                                                  starting_id = starting_object_id)

        gt_mask, pred_mask = gt_mask_classes[class_id], pred_mask_allclasses[class_id]
        missing_objects[class_id] = len(gt_mask) - len(pred_mask)

        index_mapping.update(map_indexes(gt_mask, pred_mask, calculate_mask_iou))

    return index_mapping, missing_objects

def eval_predictions(gt : {int : any}, pred : {int : any}, object_id_mapping : {int : int},
                     metric_evaluator) -> ({int : float}, {int}):
    scores = {}

    for object_id in pred.keys():
        assert object_id in object_id_mapping

        gt_object_id = object_id_mapping[object_id]
        key_format = f'pred_{gt_object_id}'

        if gt_object_id not in gt: #in tracker but not in annotations
            scores[key_format] = -1
            continue

        scores[key_format] = metric_evaluator(gt[gt_object_id], pred[object_id])

    # check if there are any missing detections
    pred_object_ids = set(object_id_mapping.values())

    missing_detections = set()

    for gt_object_id in gt.keys():
        if gt_object_id not in pred_object_ids:
            missing_detections.add(gt_object_id)

    return scores, missing_detections

def eval_detections(gt_detections : {int : (int,)}, pred_detections : {int : (int,)},
                    object_id_mapping : {int : int}) -> ({int : float}, {int}):
    '''Detections are in the format of {object_id : [box]} and {pred_oid : gt_oid}
    Returns in the format of {object_id (from either) : score}'''
    return eval_predictions(gt_detections, pred_detections, object_id_mapping, calculate_bb_iou)


def rgb2lab(img):
    assert img.shape[2] == 3

    return color.rgb2lab(img)


def lab2rgb(img):
    return color.lab2rgb(img)


def separate_segmentation_mask(mask : np.array, OBJECT_LIMIT = 20) -> {int : np.array}:
    '''given a segmentation of pixels where each pixel value corresponds to a specific object,
    return {object_id : binary_mask} '''
    object_ids = np.unique(mask)
    assert len(object_ids) < OBJECT_LIMIT, 'too many objects in segmentation scene'

    return {int(object_id) : mask == object_id for object_id in object_ids}


def get_bbox_from_mask(binary_mask : np.array) -> [int]:
    '''given a 2D 0/1 binary input mask, output the xyhw bounding box'''

    assert len(binary_mask.shape) == 2, 'dimenisons of binary mask are incorrect'
    assert tuple(np.unique(binary_mask)) == (0,1)

    indices = np.where(binary_mask == 1)

    x_min, x_max = indices[0].min(), indices[0].max()
    y_min, y_max = indices[1].min(), indices[1].max()

    return [x_min, x_max - x_min, y_min, y_max - y_min]

def binarize_mask(mask : np.array) -> np.array:
    binary_mask = np.zeros_like(mask, dtype=bool)
    mask -= mask.min()
    mask /= mask.max()
    binary_mask[mask>0.5] = 1

    return binary_mask

def combine_masks(object_masks : {int : np.array}) -> np.array:
    initial_mask = None
    for object_id in object_masks.keys():
        if not initial_mask:
            initial_mask = object_masks[object_id].copy()

        else:
            initial_mask = np.logical_or(initial_mask, object_masks[object_id])

    return initial_mask

def eval_segmentation(gt_masks : {int : np.array}, pred_masks : {int : np.array}, object_id_mapping : {int : int}):
    '''evals in the format of class_id : score'''
    return eval_predictions(gt_masks, pred_masks, object_id_mapping, calculate_mask_iou)

def eval_segmentation_diffmasks(gt_masks : {int : np.array}):
    pass

def segment_subimg_mrf(subimg):
    height, width = subimg.shape[0], subimg.shape[1]

    # Define the MRF parameters
    n_states = 2
    smoothness = 1.0

    smooth_energy = 0
    for i in range(height):
        for j in range(width):
            if i < height - 1:
                smooth_energy += smoothness * np.sum(np.abs(subimg[i, j] - subimg[i + 1, j]))
            if j < width - 1:
                smooth_energy += smoothness * np.sum(np.abs(subimg[i, j] - subimg[i, j + 1]))

    # Optimize the MRF using the Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(np.array([[smooth_energy] * n_states]))

    # Get the optimized binary segmentation mask
    mask = np.zeros((height, width), dtype=np.int8)
    mask[row_ind[0]] = col_ind

    return mask

def segment_image_mrf(full_img : np.array, object_boxes : {int : [int]}):
    obj_masks = {}
    for object_id, bbox in object_boxes.items():
        subimg = full_img[bbox[0] : bbox[0] + bbox[2], bbox[1] : bbox[1] + bbox[3]].copy()
        mask = segment_subimg_mrf(subimg)

        obj_masks[object_id] = np.zeros((full_img.shape[0], full_img.shape[1]))
        obj_masks[object_id][bbox[0] : bbox[0] + bbox[2], bbox[1] : bbox[1] + bbox[3]] = mask

    return obj_masks

# def get_knn_reference(mask : np.array, img_lab : np.array) -> np.array:
#     assert len(mask.unique()) == 2
#
#     reference_image = img_lab[mask].mean(axis=(0,1))
#
#     return reference_image

def get_knn_references(masks : {int : np.array}, full_img : np.array) -> {int : np.array}:
    assert full_img.shape[2] == 3
    mask_references = {}
    img_lab = rgb2lab(full_img)
    for object_id, object_mask in masks.items():
        mask_references[object_id] = img_lab[object_mask].mean(axis=(0))
        assert mask_references[object_id].shape[0] == 3, f'Shape of mean is incorrect: {mask_references[object_id].shape}'

    return mask_references

def get_midbox_references(bboxes : {int : (int,)}, full_img : np.array, box_radius=5) -> {int : np.array}:
    assert full_img.shape[2] == 3
    box_references = {}
    igm_lab = rgb2lab(full_img)
    for object_id, bbox in bboxes.items():
        if (bbox[2] - bbox[0] < 10) or (bbox[3] - bbox[1] < 10):
            print(f'bbox too small: {full_img}')
            continue

        # midbox as reference
        midx, midy = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
        midbox = full_img[midx-box_radius : midx+box_radius, midy-box_radius : midy+box_radius]

        box_references[object_id] = midbox.mean(axis=0)
        assert box_references[object_id].shape[0] == 3, f'Shape of mean is incorrect: {box_references[object_id].shape}'

    return box_references

def segment_subimg_knn(subimg, bg_ref, object_ref, norm=1) -> np.array:
    '''segments a bounding box with a given subimg and a background reference'''
    bg_dists = (subimg - bg_ref) ** 2 / norm
    ref_dists = (subimg - object_ref) ** 2 / norm

    return ref_dists.sum(axis=-1) > bg_dists.sum(axis=-1)

def segment_image_knn(full_img: np.array, object_boxes: {int: [int]}, object_references: {int : np.array}, max_box_radius=5):
    '''for every box mapped as object_id : [bbox (XYWH)], return the cluster
    full_img should be H x W x 3'''
    assert full_img.shape[2] == 3

    img_lab = rgb2lab(full_img)
    image_normalization = img_lab.max(axis=(0, 1))

    # assume the edges of the image are the background and use as reference
    # TODO: make the background the actual background

    background_mask = np.zeros_like(img_lab, dtype=bool)
    background_mask[:5] = 1
    background_mask[-5:] = 1
    background_mask[:, -5:] = 1
    background_mask[:, :5] = 1

    ref_bg = img_lab[background_mask].reshape(-1, 3).mean(axis=0)

    obj_masks = {}

    for object_id, bbox in object_boxes.items():
        if bbox[2] < max_box_radius or bbox[3] < max_box_radius:
            continue

        # bounding box to segment
        subimg_lab = img_lab[bbox[0]: bbox[0] + bbox[2], bbox[1]: bbox[1] + bbox[3]]
        mask = segment_subimg_knn(subimg_lab, ref_bg, object_references[object_id], image_normalization)

        obj_masks[object_id] = np.zeros((full_img.shape[0], full_img.shape[1]))
        obj_masks[object_id][bbox[0]: bbox[0] + bbox[2], bbox[1]: bbox[1] + bbox[3]] = mask

    return obj_masks

def round_tracker_outputs(objects):
    return {k : v.astype(int) for k, v in objects.items()}