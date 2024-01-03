import torch
import torch.nn.functional as F

def bbox_iou(box1:torch.Tensor, box2:torch.Tensor, xywh:bool=True, eps:float=1e-10):
    """
    Calculate IoU between two bounding boxes

    Args:
        box1: (Tensor) with shape (..., 1 or n, 4)
        box2: (Tensor) with shape (..., n, 4)
        xywh: (bool) True if box coordinates are in (xywh) else (xyxy)

    Returns:
        iou: (Tensor) with IoU
    """
    if xywh:
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, dim=-1), box2.chunk(4, dim=-1)
        b1_x1, b1_y1, b1_x2, b1_y2 = x1 - w1 / 2, y1 - h1 / 2, x1 + w1 / 2, y1 + h1 / 2
        b2_x1, b2_y1, b2_x2, b2_y2 = x2 - w2 / 2, y2 - h2 / 2, x2 + w2 / 2, y2 + h2 / 2
    else:
        (b1_x1, b1_y1, b1_x2, b1_y2), (b2_x1, b2_y1, b2_x2, b2_y2) = box1.chunk(4, dim=-1), box2.chunk(4, dim=-1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1

    intersection = (torch.minimum(b1_x2, b2_x2) - torch.maximum(b1_x1, b2_x1)).clamp(min=0) * \
                   (torch.minimum(b1_y2, b2_y2) - torch.maximum(b1_y1, b2_y1)).clamp(min=0)
    
    union = w1 * h1 + w2 * h2 - intersection + eps

    iou = intersection / union

    return iou

def df_loss(pred_box_dist:torch.Tensor, targets:torch.Tensor):
    """
    Sum of left and right DFL losses
    """
    target_left = targets.long()
    target_right = target_left + 1
    weight_left = target_right - targets
    weight_right = 1 - weight_left

    dfl_left = F.cross_entropy(pred_box_dist, target_left.view(-1), reduction='none').view(target_left.shape) * weight_left
    dfl_right = F.cross_entropy(pred_box_dist, target_right.view(-1), reduction='none').view(target_right.shape) * weight_right

    return torch.mean(dfl_left + dfl_right, dim=-1, keepdim=True)

