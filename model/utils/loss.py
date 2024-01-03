import torch
import torch.nn as nn

from model.modules import make_anchors, dist2bbox, bbox2dist
from .metrics import bbox_iou, df_loss


class BaseLoss:
    def __init__(self, device:str):
        self.device = device

    def compute_loss(self, batch:torch.Tensor, preds:torch.Tensor):
        raise NotImplementedError


class BboxLoss(BaseLoss):
    def __init__(self, reg_max:int, device:str, use_dfl:bool=False):
        super().__init__(device)

        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def compute_loss(self, pred_box_dist:torch.Tensor, pred_boxes:torch.Tensor, target_boxes:torch.Tensor, anchor_points:torch.Tensor, target_score:torch.Tensor):
        weight = target_score.sum(dim=-1).unsqueeze(dim=-1)
        iou = bbox_iou(pred_boxes, target_boxes, xywh=False)
        iou_loss = ((1 - iou) * weight).sum() / max(target_score.sum(), 1)

        if self.use_dfl:
            gt_ltrb = bbox2dist(target_boxes, anchor_points, self.reg_max)
            dfl_loss = df_loss(pred_box_dist.view(-1, self.reg_max+1), gt_ltrb*weight)
        else:
            dfl_loss = torch.tensor(0.0).to(self.device)

        return iou_loss, dfl_loss
