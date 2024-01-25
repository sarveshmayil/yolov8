import torch
import torchvision

from model.modules.utils import xywh2xyxy

from typing import List

def nms(preds:torch.Tensor, confidence_thresh:float=0.25, iou_thresh:float=0.45) -> List[torch.Tensor]:
    """
    Non-Maximum Suppression for predicted boxes and classes

    Args:
        preds (Tensor): Predictions from model of shape (bs, 4 + num_classes, num_boxes)
        confidence_thresh (float, optional): Confidence threshold. Defaults to 0.25
        iou_thresh (float, optional): IoU threshold. Defaults to 0.45

    Returns:
        List[Tensor]: list of tensors of shape (num_boxes, 6) containing boxes with
            (x1, y1, x2, y2, confidence, class)
    """
    b, nc, _ = preds.shape
    nc -= 4

    # max confidence score among boxes
    xc = preds[:,4:].amax(dim=1) > confidence_thresh

    # (b, 4+nc, a) -> (b, a, 4+nc)
    preds = preds.transpose(-1, -2)

    preds[..., :4] = xywh2xyxy(preds[..., :4])

    out = [torch.zeros((0,6), device=preds.device)] * b

    for i, x in enumerate(preds):
        # take max cls confidence score
        # only consider predictions with confidence > confidence_thresh
        x = x[xc[i]]

        # If there are no remaining predictions, move to next image
        if not x.shape[0]:
            continue

        box, cls = x.split((4, nc), dim=1)

        confidences, cls_idxs = cls.max(dim=1, keepdim=True)
        x = torch.cat((box, confidences, cls_idxs.float()), dim=1)

        keep_idxs = torchvision.ops.nms(x[:,:4], x[:,4], iou_thresh)

        out[i] = x[keep_idxs]

    return out
