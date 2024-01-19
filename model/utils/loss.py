import torch
import torch.nn as nn

from model.modules import make_anchors, dist2bbox, bbox2dist, xywh2xyxy
from .metrics import bbox_iou, df_loss
from .tal import TaskAlignedAssigner

from typing import Dict, Tuple


class BaseLoss:
    def __init__(self, device:str):
        self.device = device

    def compute_loss(self, batch:torch.Tensor, preds:torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class BboxLoss(BaseLoss):
    def __init__(self, reg_max:int, device:str, use_dfl:bool=False):
        super().__init__(device)

        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def compute_loss(
        self,
        pred_box_dist:torch.Tensor,
        pred_boxes:torch.Tensor,
        target_boxes:torch.Tensor,
        anchor_points:torch.Tensor,
        target_scores:torch.Tensor,
        mask:torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        weight = target_scores.sum(dim=-1)[mask].unsqueeze(dim=-1)
        iou = bbox_iou(pred_boxes[mask], target_boxes[mask], xywh=False)

        target_scores_sum = max(target_scores.sum(), 1)
        iou_loss = ((1 - iou) * weight).sum() / target_scores_sum

        if self.use_dfl:
            gt_ltrb = bbox2dist(target_boxes, anchor_points, self.reg_max)
            dfl_loss = df_loss(pred_box_dist[mask].view(-1, self.reg_max+1), gt_ltrb[mask]) * weight
            dfl_loss = dfl_loss.sum() / target_scores_sum
        else:
            dfl_loss = torch.tensor(0.0).to(self.device)

        return iou_loss, dfl_loss


class DetectionLoss(BaseLoss):
    def __init__(self, model, device:str):
        super().__init__(device)
        
        detect_head = model.model[-1]

        self.nc = detect_head.nc
        self.n_outputs = detect_head.n_outputs
        self.reg_max = detect_head.reg_max
        self.stride = detect_head.stride

        self.loss_gains = model.loss_gains

        # Projects predicted boxes to different scales
        self.proj = torch.arange(self.reg_max, device=self.device, dtype=torch.float)

        # Sigmoid + Binary Cross Entropy Loss (-log(sigmoid(x)))
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

        self.bbox_loss = BboxLoss(self.reg_max-1, self.device, use_dfl=True)

        self.tal_assigner = TaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0).to(self.device)

    def preprocess(self, targets:torch.Tensor, batch_size:int, scale_tensor:torch.Tensor) -> torch.Tensor:
        """
        Preprocesses target boxes to match predicted boxes batch size
        """
        # No bboxes in image
        if targets.shape[0] == 0:
            return torch.zeros(batch_size, 0, 5, device=self.device)
        
        im_idxs = targets[:,0]
        _, counts = im_idxs.unique(return_counts=True)

        # Row idxs correspond to predicted idxs, col idxs correspond to targets
        out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
        for i in range(batch_size):
            mask = (im_idxs == i)
            n_matches = mask.sum()
            if n_matches > 0:
                # Add cls and bbox targets to output at matching indices
                out[i, :n_matches] = targets[mask, 1:]

        # Convert boxes from xywh to xyxy
        out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def decode_bbox(self, anchor_points:torch.Tensor, pred_box_dist:torch.Tensor):
        """
        Decodes bounding box coordinates from anchor points and predicted
        box distribution. Returns bounding box coordinates in (xyxy) format.
        """
        b, a, c = pred_box_dist.shape  # (batch, anchors, channels)
        # Reshape to (batch, anchors, 4, reg_max) then softmax
        # along reg_max dim and mul by (reg_max,) -> (b,a,4)
        pred_boxes = pred_box_dist.view(b, a, 4, c//4).softmax(dim=3) @ self.proj
        return dist2bbox(pred_boxes, anchor_points, xywh=False)

    def compute_loss(self, batch:Dict[str,torch.Tensor], preds:torch.Tensor):
        pred_box_dist, pred_cls = torch.cat(
            [xi.view(preds[0].shape[0], self.n_outputs, -1) for xi in preds], dim=2
        ).split((4*self.reg_max, self.nc), dim=1)

        pred_cls = pred_cls.permute(0, 2, 1).contiguous()
        pred_box_dist = pred_box_dist.permute(0, 2, 1).contiguous()

        batch_size = pred_box_dist.shape[0]
        im_size = torch.tensor(preds[0].shape[2:], device=self.device) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(preds, self.stride)
        pred_boxes = self.decode_bbox(anchor_points, pred_box_dist)  # (b, h*w, 4) in (xyxy)

        # (batch_idx, cls, xywh box)
        targets = torch.cat((batch['batch_idx'].view(-1,1), batch['cls'].view(-1,1), batch['bboxes']), dim=1).to(self.device)
        targets = self.preprocess(targets, batch_size, scale_tensor=im_size[[1,0,1,0]])
        # cls, xyxy box
        gt_cls, gt_boxes = targets.split((1,4), dim=2)
        gt_mask = gt_boxes.sum(dim=2, keepdim=True) > 0  # mask to filter out (0,0,0,0) boxes (just used to pad tensor)

        _, target_boxes, target_scores, mask = self.tal_assigner(
            pred_cls.detach().sigmoid(), pred_boxes.detach() * stride_tensor, anchor_points * stride_tensor, gt_cls, gt_boxes, gt_mask
        )
        
        cls_loss = self.bce_loss(pred_cls, target_scores).sum() / max(target_scores.sum(), 1)

        if mask.sum() > 0:
            iou_loss, dfl_loss = self.bbox_loss.compute_loss(
                pred_box_dist, pred_boxes, target_boxes/stride_tensor, anchor_points, target_scores, mask
            )
        else:
            iou_loss = torch.tensor(0.0).to(self.device)
            dfl_loss = torch.tensor(0.0).to(self.device)

        loss = self.loss_gains['cls']*cls_loss + self.loss_gains['iou']*iou_loss + self.loss_gains['dfl']*dfl_loss

        return loss * batch_size
