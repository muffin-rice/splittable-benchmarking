'''This file contains the split versions (in ClientModel and ServerModel) of
configs/coco2017/supervised_compression/ghnd-bq/faster_rcnn_resnet50-bq1ch_fpn_from_faster_rcnn_resnet50_fpn.yaml'''

import warnings
import torch
from torch import Tensor, nn
from typing import List, Tuple
from collections import OrderedDict
from copy import deepcopy

# ==================================================== Client ====================================================

class ClientModel(nn.Module):
    '''client model for faster_rcnn_resnet50-bq1ch_fpn_from_faster_rcnn_resnet50_fpn.yaml'''

    def __init__(self, student_model2):
        super(ClientModel, self).__init__()
        student_model = deepcopy(student_model2)
        self.transform = student_model.transform
        self.encoder = student_model.backbone.body.bottleneck_layer.encoder
        self.training = student_model.training
        self.compressor = student_model.backbone.body.bottleneck_layer.compressor

    def remove_degenerate_boxes(self, images, targets):
        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    torch._assert(
                        False,
                        "All bounding boxes should have positive height and width."
                        f" Found invalid box {degen_bb} for target at index {target_idx}.",
                    )

    def transform_forward(self, images, targets=None):
        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training mode")
            else:
                for target in targets:
                    boxes = target["boxes"]
                    if isinstance(boxes, torch.Tensor):
                        torch._assert(
                            len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                            f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.",
                        )
                    else:
                        torch._assert(False, f"Expected target boxes to be of type Tensor, got {type(boxes)}.")

        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)

        return images, targets, original_image_sizes, images.tensors.shape[-2:]

    def encoder_forward(self, z):
        z = self.encoder(z)
        if self.compressor is not None:
            z = self.compressor(z)
        return z

    def forward(self, x):
        images, targets, original_image_sizes, secondary_image_size = self.transform_forward(x)
        self.remove_degenerate_boxes(images, targets)
        features = self.encoder_forward(images.tensors)
        return (features, targets), (images.image_sizes, original_image_sizes, secondary_image_size)

# ==================================================== SERVER ====================================================

# RPN Helper functions used for redefining RPN Forward
# RPN Forward currently takes in the images as input (while only using the shapes); by re-defining the function,
# we avoid using the images as input. However, our custom re-definition must have these non-built in functions
# from https://github.com/pytorch/vision/blob/main/torchvision/models/detection/rpn.py
def concat_box_prediction_layers(box_cls: List[Tensor], box_regression: List[Tensor]) -> Tuple[Tensor, Tensor]:
    box_cls_flattened = []
    box_regression_flattened = []
    # for each feature level, permute the outputs to make them be in the
    # same format as the labels. Note that the labels are computed for
    # all feature levels concatenated, so we keep the same representation
    # for the objectness and the box_regression
    for box_cls_per_level, box_regression_per_level in zip(box_cls, box_regression):
        N, AxC, H, W = box_cls_per_level.shape
        Ax4 = box_regression_per_level.shape[1]
        A = Ax4 // 4
        C = AxC // A
        box_cls_per_level = permute_and_flatten(box_cls_per_level, N, A, C, H, W)
        box_cls_flattened.append(box_cls_per_level)

        box_regression_per_level = permute_and_flatten(box_regression_per_level, N, A, 4, H, W)
        box_regression_flattened.append(box_regression_per_level)
    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    box_cls = torch.cat(box_cls_flattened, dim=1).flatten(0, -2)
    box_regression = torch.cat(box_regression_flattened, dim=1).reshape(-1, 4)
    return box_cls, box_regression

def permute_and_flatten(layer: Tensor, N: int, A: int, C: int, H: int, W: int) -> Tensor:
    layer = layer.view(N, -1, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2)
    layer = layer.reshape(N, -1, C)
    return layer


def rpn_forward(self, features, targets, image_sizes, secondary_image_size):
    features = list(features.values())
    objectness, pred_bbox_deltas = self.head(features)

    # modify anchor generator to not use images
    anchors = self.anchor_generator(image_sizes, secondary_image_size, features)

    num_images = len(anchors)
    num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
    num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
    objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)
    # apply pred_bbox_deltas to anchors to obtain the decoded proposals
    # note that we detach the deltas because Faster R-CNN do not backprop through
    # the proposals
    proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
    proposals = proposals.view(num_images, -1, 4)

    # modified to use image_sizes
    boxes, scores = self.filter_proposals(proposals, objectness, image_sizes, num_anchors_per_level)

    losses = {}
    if self.training:
        if targets is None:
            raise ValueError("targets should not be None")
        labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)
        regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)
        loss_objectness, loss_rpn_box_reg = self.compute_loss(
            objectness, pred_bbox_deltas, labels, regression_targets
        )
        losses = {
            "loss_objectness": loss_objectness,
            "loss_rpn_box_reg": loss_rpn_box_reg,
        }
    return boxes, losses


def anchor_forward(self, image_sizes, tensorshapes, feature_maps: List[Tensor]) -> List[Tensor]:
    grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]
    image_size = tensorshapes
    dtype, device = feature_maps[0].dtype, feature_maps[0].device
    strides = [
        [
            torch.empty((), dtype=torch.int64, device=device).fill_(image_size[0] // g[0]),
            torch.empty((), dtype=torch.int64, device=device).fill_(image_size[1] // g[1]),
        ]
        for g in grid_sizes
    ]
    self.set_cell_anchors(dtype, device)
    anchors_over_all_feature_maps = self.grid_anchors(grid_sizes, strides)
    anchors: List[List[torch.Tensor]] = []
    for _ in range(len(image_sizes)):
        anchors_in_image = [anchors_per_feature_map for anchors_per_feature_map in anchors_over_all_feature_maps]
        anchors.append(anchors_in_image)
    anchors = [torch.cat(anchors_per_image) for anchors_per_image in anchors]
    return anchors


class ServerModel(nn.Module):
    def __init__(self, student_model2):
        super().__init__()
        student_model = deepcopy(student_model2)
        self.backbone = student_model.backbone

        # change bottleneck_layer to only decode (encoder in ClientModel)
        self.backbone.body.bottleneck_layer.encoder = nn.Identity()
        self.backbone.body.bottleneck_layer.compressor = nn.Identity()
        self.backbone.body.bottleneck_layer.encode = lambda z: {'z': z}

        # change RPN and anchor generator
        self.rpn = student_model.rpn
        self.rpn.anchor_generator.forward = lambda image_sizes, tensorshapes, feature_maps: anchor_forward(
            self.rpn.anchor_generator, image_sizes, tensorshapes, feature_maps)
        self.rpn.forward = lambda features, targets, image_sizes, secondary_image_size: rpn_forward(self.rpn, features,
                                                                                                    targets,
                                                                                                    image_sizes,
                                                                                                    secondary_image_size)

        self.roi_heads = student_model.roi_heads
        self.postprocess = student_model.transform.postprocess
        self._has_warned = student_model._has_warned
        self.eager_outputs = student_model.eager_outputs

    def forward(self, features, targets, image_sizes, original_image_sizes, secondary_image_size):
        '''image sizes '''
        features = self.backbone(features)

        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        proposals, proposal_losses = self.rpn(features, targets, image_sizes, secondary_image_size)
        detections, detector_losses = self.roi_heads(features, proposals, image_sizes, targets)
        detections = self.postprocess(detections, image_sizes, original_image_sizes)  # type: ignore[operator]

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        else:
            return self.eager_outputs(losses, detections)