import torch
import torch.nn.functional as F
import numpy as np

from typing import Tuple, Any, Union


def xywh2xyxy(xywh:np.ndarray):
    """
    Convert bounding box coordinates from (xywh) to (xyxy)
    """
    xy, wh = np.split(xywh, 2, axis=-1)
    return np.concatenate((xy - wh / 2, xy + wh / 2), axis=-1)

def xyxy2xywh(xyxy:np.ndarray):
    """
    Convert bounding box coordinates from (xyxy) to (xywh)
    """
    xy_lt, xy_rb = np.split(xyxy, 2, axis=-1)
    return np.concatenate(((xy_lt + xy_rb) / 2, xy_rb - xy_lt), axis=-1)

def pad_to(x:torch.Tensor, stride:int=None, shape:Tuple[int,int]=None):
    """
    Pads an image with zeros to make it divisible by stride
    (Pads both top/bottom and left/right evenly) or pads to
    specified shape.

    Args:
        x (Tensor): image tensor of shape (..., h, w)
        stride (optional, int): stride of model
        shape (optional, Tuple[int,int]): shape to pad image to
    """
    h, w = x.shape[-2:]

    if stride is not None:
        h_new = h if h % stride == 0 else h + stride - h % stride
        w_new = w if w % stride == 0 else w + stride - w % stride
    elif shape is not None:
        h_new, w_new = shape

    t, b = int((h_new-h) / 2), int(h_new-h) - int((h_new-h) / 2)
    l, r = int((w_new-w) / 2), int(w_new-w) - int((w_new-w) / 2)
    pads = (l, r, t, b)

    x_padded = F.pad(x, pads, "constant", 0)

    return x_padded, pads

def unpad(x:torch.Tensor, pads:tuple):
    l, r, t, b = pads
    return x[..., t:-b, l:-r]

def pad_xyxy(xyxy:Union[np.ndarray, torch.Tensor], pads:Tuple[int, int, int, int], im_size:Tuple[int, int]=None, return_norm:bool=False):
    """
    Add padding to the bounding boxes based on image padding

    Args:
        xyxy: The bounding boxes in the format of `(x_min, y_min, x_max, y_max)`.
            if `im_size` is provided, assume this is normalized coordinates
        pad: The padding added to the image in the format
            of `(left, right, top, bottom)`.
        im_size: The size of the original image in the format of `(height, width)`.
        return_norm: Whether to return normalized coordinates
    """
    l, r, t, b = pads
    if return_norm and im_size is None:
        raise ValueError("im_size must be provided if return_norm is True")
    
    if im_size is not None:
        h, w = im_size
        hpad, wpad = h+b+t, w+l+r
    
    if isinstance(xyxy, np.ndarray):
        xyxy_unnorm = xyxy * np.array([w, h, w, h], dtype=xyxy.dtype) if im_size else xyxy
        padded = xyxy_unnorm + np.array([l, t, l, t], dtype=xyxy.dtype)
        if return_norm:
            padded /= np.array([wpad, hpad, wpad, hpad], dtype=xyxy.dtype)
        return padded
    
    xyxy_unnorm = xyxy * torch.tensor([w, h, w, h], dtype=xyxy.dtype, device=xyxy.device) if im_size else xyxy
    padded = xyxy_unnorm + torch.tensor([l, t, l, t], dtype=xyxy.dtype, device=xyxy.device)
    if return_norm:
        padded /= torch.tensor([wpad, hpad, wpad, hpad], dtype=xyxy.dtype, device=xyxy.device)
    return padded

def pad_xywh(xywh:Union[np.ndarray, torch.Tensor], pads:Tuple[int, int, int, int], im_size:Tuple[int, int]=None, return_norm:bool=False):
    """
    Add padding to the bounding boxes based on image padding

    Args:
        xywh: The bounding boxes in the format of `(x, y, w, h)`.
            if `im_size` is provided, assume this is normalized coordinates
        pad: The padding added to the image in the format
            of `(left, right, top, bottom)`.
        im_size: The size of the original image in the format of `(height, width)`.
        return_norm: Whether to return normalized coordinates
    """
    l, r, t, b = pads
    if return_norm and im_size is None:
        raise ValueError("im_size must be provided if return_norm is True")
    
    if im_size is not None:
        h, w = im_size
        hpad, wpad = h+b+t, w+l+r

    if isinstance(xywh, np.ndarray):
        xywh_unnorm = xywh * np.array([w, h, w, h], dtype=xywh.dtype) if im_size else xywh
        padded = xywh_unnorm + np.array([l, t, 0, 0], dtype=xywh.dtype)
        if return_norm:
            padded /= np.array([wpad, hpad, wpad, hpad], dtype=xywh.dtype)
        return padded
    
    xywh_unnorm = xywh * torch.tensor([w, h, w, h], dtype=xywh.dtype, device=xywh.device) if im_size else xywh
    padded = xywh_unnorm + torch.tensor([l, t, 0, 0], dtype=xywh.dtype, device=xywh.device)
    if return_norm:
        padded /= torch.tensor([wpad, hpad, wpad, hpad], dtype=xywh.dtype, device=xywh.device)
    return padded

def unpad_xyxy(xyxy:Union[np.ndarray, torch.Tensor], pads:Tuple[int, int, int, int]):
    """
    Remove padding from the bounding boxes based on image padding

    Args:
        pad: The padding added to the image in the format
            of `(left, right, top, bottom)`.
    """
    l, r, t, b = pads
    if isinstance(xyxy, np.ndarray):
        return xyxy - np.array([l, t, l, t], dtype=xyxy.dtype)
    return xyxy - torch.tensor([l, t, l, t], dtype=xyxy.dtype, device=xyxy.device)

def box_iou_batch(gt_boxes: np.ndarray, pred_boxes: np.ndarray) -> np.ndarray:
    """
    Compute Intersection over Union (IoU) of two sets of bounding boxes -
        `gt_boxes` and `pred_boxes`. Both sets
        of boxes are expected to be in `(xyxy)` format.

    Args:
        gt_boxes (np.ndarray): 2D `np.ndarray` representing ground-truth boxes.
            `shape = (N, 4)` where `N` is number of true objects.
        pred_boxes (np.ndarray): 2D `np.ndarray` representing detection boxes.
            `shape = (M, 4)` where `M` is number of detected objects.

    Returns:
        np.ndarray: Pairwise IoU of boxes from `gt_boxes` and `pred_boxes`.
            `shape = (N, M)` where `N` is number of true objects and
            `M` is number of detected objects.
    """

    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    area_true = box_area(gt_boxes.T)
    area_detection = box_area(pred_boxes.T)

    top_left = np.maximum(gt_boxes[:, None, :2], pred_boxes[:, :2])
    bottom_right = np.minimum(gt_boxes[:, None, 2:], pred_boxes[:, 2:])

    area_inter = np.prod(np.clip(bottom_right - top_left, a_min=0, a_max=None), 2)
    return area_inter / (area_true[:, None] + area_detection - area_inter)

def non_max_suppression(predictions: np.ndarray, iou_threshold: float = 0.5) -> np.ndarray:
    """
    Perform Non-Maximum Suppression (NMS) on object detection predictions.

    Args:
        predictions (np.ndarray): An array of object detection predictions in
            the format of `(x_min, y_min, x_max, y_max, score)`
            or `(x_min, y_min, x_max, y_max, score, class)`.
        iou_threshold (float, optional): The intersection-over-union threshold
            to use for non-maximum suppression.

    Returns:
        np.ndarray: A boolean array indicating which predictions to keep after n
            on-maximum suppression.

    Raises:
        AssertionError: If `iou_threshold` is not within the
            closed range from `0` to `1`.
    """
    assert 0 <= iou_threshold <= 1, (
        "Value of `iou_threshold` must be in the closed range from 0 to 1, "
        f"{iou_threshold} given."
    )
    rows, columns = predictions.shape

    # add column #5 - category filled with zeros for agnostic nms
    if columns == 5:
        predictions = np.c_[predictions, np.zeros(rows)]

    # sort predictions column #4 - score
    sort_index = np.flip(predictions[:, 4].argsort())
    predictions = predictions[sort_index]

    boxes = predictions[:, :4]
    categories = predictions[:, 5]
    ious = box_iou_batch(boxes, boxes)
    ious = ious - np.eye(rows)

    keep = np.ones(rows, dtype=bool)

    for index, (iou, category) in enumerate(zip(ious, categories)):
        if not keep[index]:
            continue

        # drop detections with iou > iou_threshold and
        # same category as current detections
        condition = (iou > iou_threshold) & (categories == category)
        keep = keep & ~condition

    return keep[sort_index.argsort()]

### Data validation for Detections class
def validate_xyxy(xyxy:Any) -> None:
    expected_shape = "(_, 4)"
    actual_shape = str(getattr(xyxy, "shape", None))
    is_valid = isinstance(xyxy, np.ndarray) and xyxy.ndim == 2 and xyxy.shape[1] == 4
    if not is_valid:
        raise ValueError(
            f"xyxy must be a 2D np.ndarray with shape {expected_shape}, but got shape "
            f"{actual_shape}"
        )

def validate_mask(mask:Any, n:int) -> None:
    expected_shape = f"({n}, H, W)"
    actual_shape = str(getattr(mask, "shape", None))
    is_valid = mask is None or (
        isinstance(mask, np.ndarray) and len(mask.shape) == 3 and mask.shape[0] == n
    )
    if not is_valid:
        raise ValueError(
            f"mask must be a 3D np.ndarray with shape {expected_shape}, but got shape "
            f"{actual_shape}"
        )


def validate_class_id(class_id:Any, n:int) -> None:
    expected_shape = f"({n},)"
    actual_shape = str(getattr(class_id, "shape", None))
    is_valid = class_id is None or (
        isinstance(class_id, np.ndarray) and class_id.shape == (n,)
    )
    if not is_valid:
        raise ValueError(
            f"class_id must be a 1D np.ndarray with shape {expected_shape}, but got "
            f"shape {actual_shape}"
        )


def validate_confidence(confidence:Any, n:int) -> None:
    expected_shape = f"({n},)"
    actual_shape = str(getattr(confidence, "shape", None))
    is_valid = confidence is None or (
        isinstance(confidence, np.ndarray) and confidence.shape == (n,)
    )
    if not is_valid:
        raise ValueError(
            f"confidence must be a 1D np.ndarray with shape {expected_shape}, but got "
            f"shape {actual_shape}"
        )


def validate_tracker_id(tracker_id:Any, n:int) -> None:
    expected_shape = f"({n},)"
    actual_shape = str(getattr(tracker_id, "shape", None))
    is_valid = tracker_id is None or (
        isinstance(tracker_id, np.ndarray) and tracker_id.shape == (n,)
    )
    if not is_valid:
        raise ValueError(
            f"tracker_id must be a 1D np.ndarray with shape {expected_shape}, but got "
            f"shape {actual_shape}"
        )

def validate_detections_fields(
    xyxy: Any,
    mask: Any,
    class_id: Any,
    confidence: Any,
    tracker_id: Any
) -> None:
    validate_xyxy(xyxy)
    n = len(xyxy)
    validate_mask(mask, n)
    validate_class_id(class_id, n)
    validate_confidence(confidence, n)
    validate_tracker_id(tracker_id, n)
