import torch
import numpy as np
from utils2 import *

class Evaler:
    def __init__(self, device, run_type, console_logger, stats_logger, tracking_box_limit):
        self.device = device

        self.object_gt_mapping = None
        self.console_logger = console_logger
        self.stats_logger = stats_logger
        self.run_type = run_type
        self.tracking_box_limit = tracking_box_limit

    def cast_obj_to_tensor(self, obj):
        if 'array' in str(type(obj)):
            return torch.from_numpy(obj).to(self.device)

        if 'dict' in str(type(obj)):
            return {k : torch.as_tensor(v, device = self.device) for k, v in obj.items()}

    def cast_scores_to_logs(self, scores : {}):
        log_dict = {}
        for k, v in scores.items():
            if torch.is_tensor(v):
                log_dict[k] = v.item()
            else:
                log_dict[k] = v

        return log_dict

    def calculate_mask_iou(self, maskA, maskB):
        if 'cuda' in self.device:
            intersection = torch.logical_and(maskA, maskB).sum()
            union = torch.logical_or(maskA, maskB).sum()

            return (intersection / union).item()

        # otherwise numpy arr
        intersection = np.logical_and(maskA, maskB).sum()
        union = np.logical_or(maskA, maskB).sum()

        return intersection / union

    def calculate_bb_iou(self, boxA, boxB):
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
        iou = interArea / (boxAArea + boxBArea - interArea)

        # return the intersection over union value
        if 'cuda' in self.device:
            return iou.item()

        return iou

    def eval_segmentation(self, gt_masks: {int: np.array}, pred_masks: {int: np.array}, object_id_mapping: {int: int}):
        '''evals in the format of class_id : score'''
        format_lambda = lambda object_id: f'p_s_{object_id}'
        self.console_logger.log_info('Evaluating segmentation')
        if 'cuda' in self.device:
            gt_masks = self.cast_obj_to_tensor(gt_masks)
            pred_masks = self.cast_obj_to_tensor(pred_masks)
            pred_scores, missing_preds = eval_predictions(gt_masks, pred_masks, object_id_mapping,
                                                          self.calculate_mask_iou, format_lambda)
            self.stats_logger.push_log({'missing_preds': missing_preds, **pred_scores})
            # self.stats_logger.push_log({'missing_preds': missing_preds, **self.cast_scores_to_logs(pred_scores)})
            return

        pred_scores, missing_preds = eval_predictions(gt_masks, pred_masks, object_id_mapping, self.calculate_mask_iou,
                                                      format_lambda)
        self.stats_logger.push_log({'missing_preds': missing_preds, **pred_scores})

    def eval_detections(self, gt_detections: {int: (int,)}, pred_detections: {int: (int,)},
                        object_id_mapping: {int: int}) -> ({int: float}, {int}):
        '''Detections are in the format of {object_id : [box]} and {pred_oid : gt_oid}
        Returns in the format of {object_id (from either) : score}'''
        format_lambda = lambda object_id: f'p_d_{object_id}'
        self.console_logger.log_info('Evaluating detections')
        if 'cuda' in self.device:
            gt_detections = self.cast_obj_to_tensor(gt_detections)
            pred_detections = self.cast_obj_to_tensor(pred_detections)
            pred_scores, missing_preds = eval_predictions(gt_detections, pred_detections, object_id_mapping,
                                                          self.calculate_mask_iou, format_lambda)
            self.stats_logger.push_log({'missing_preds': missing_preds, **pred_scores})
            # self.stats_logger.push_log({'missing_preds': missing_preds, **self.cast_scores_to_logs(pred_scores)})
            return

        pred_scores, missing_preds = eval_predictions(gt_detections, pred_detections, object_id_mapping, self.calculate_mask_iou,
                                                      format_lambda)
        self.stats_logger.push_log({'missing_preds': missing_preds, **pred_scores})

    def evaluate_predictions(self, pred, object_gt_mapping, pred_masks=None):
        if self.run_type == 'BB':
            self.console_logger.log_info('Evaluating detections')
            self.eval_detections(self.gt_as_pred, pred, object_gt_mapping)
        elif self.run_type == 'SM':
            self.console_logger.log_info('Evaluating masks')
            self.eval_segmentation(self.gt_as_pred, pred, object_gt_mapping)
        elif self.run_type == 'BBSM':  # pred is still in "bb" mode
            self.console_logger.log_debug('Evaluating detections and masks')
            self.eval_detections(self.gt_as_pred, pred, object_gt_mapping)
            self.eval_segmentation(self.gt_masks_as_pred, pred_masks, object_gt_mapping)

    def load_gt_bb(self, class_info, gt):
        self.gt_preds = get_gt_detections(class_info, gt)  # class : {object : info}
        self.gt_as_pred = get_gt_detections_as_pred(class_info, gt)  # {object : info}

    def load_gt_sm(self, class_info, gt):
        self.gt_preds = get_gt_masks(class_info, gt)  # class : {object : info}
        self.gt_as_pred = get_gt_masks_as_pred(class_info, gt)  # {object : info}

    def load_gt_bbsm(self, class_info, gt):
        self.gt_preds = get_gt_dets_from_mask(class_info, gt)  # class : {object : info}
        self.gt_as_pred = get_gt_dets_from_mask_as_pred(class_info, gt)  # {object : info}
        self.gt_masks_as_pred = get_gt_masks_as_pred(class_info, gt)

    def load_gt(self, class_info, gt):
        # get the gt detections in {object : info} for eval
        if self.run_type == 'BB':
            self.load_gt_bb(class_info, gt)
        elif self.run_type == 'SM':
            self.load_gt_sm(class_info, gt)
        elif self.run_type == 'BBSM':
            self.load_gt_bbsm(class_info, gt)

        if self.run_type == 'BB':
            self.console_logger.log_debug(f'num gt_detections : {len(self.gt_as_pred)}')
        elif self.run_type == 'SM' or self.run_type == 'BBSM':
            self.console_logger.log_debug(f'Got gt mask; type {type(self.gt_preds)} with len {len(self.gt_preds)}')