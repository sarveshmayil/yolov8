import torch
import torch.nn as nn

from .metrics import bbox_iou

from typing import Tuple


def anchors_in_gt_boxes(anchor_points:torch.Tensor, gt_boxes:torch.Tensor, eps:float=1e-8):
    """
    Returns mask for positive anchor centers that are in GT boxes

    Args:
        anchor_points (Tensor): Anchor points of shape (n_anchors, 2)
        gt_boxes (Tensor): GT boxes of shape (batch_size, n_boxes, 4)
        
    Returns:
        mask (Tensor): Mask of shape (batch_size, n_boxes, n_anchors)
    """
    n_anchors = anchor_points.shape[0]
    batch_size, n_boxes, _ = gt_boxes.shape
    lt, rb = gt_boxes.view(-1, 1, 4).chunk(chunks=2, dim=2)
    box_deltas = torch.cat((anchor_points.unsqueeze(0) - lt, rb - anchor_points.unsqueeze(0)), dim=2).view(batch_size, n_boxes, n_anchors, -1)
    return torch.amin(box_deltas, dim=3) > eps

def select_highest_iou(mask:torch.Tensor, ious:torch.Tensor, num_max_boxes:int):
    """
    Select GT box with highest IoU for each anchor

    Args:
        mask (Tensor): Mask of shape (batch_size, num_max_boxes, n_anchors)
        ious (Tensor): IoU of shape (batch_size, num_max_boxes, n_anchors)

    Returns:
        target_gt_box_idxs (Tensor): Index of GT box with highest IoU for each anchor of shape (batch_size, n_anchors)
        fg_mask (Tensor): Mask of shape (batch_size, n_anchors) where 1 indicates positive anchor
        mask (Tensor): Mask of shape (batch_size, num_max_boxes, n_anchors) where 1 indicates positive anchor
    """
    # sum over n_max_boxes dim to get num GT boxes assigned to each anchor
    # (batch_size, num_max_boxes, n_anchors) -> (batch_size, n_anchors)
    fg_mask = mask.sum(dim=1)

    if fg_mask.max() > 1:
        # If 1 anchor assigned to more than one GT box, select the one with highest IoU
        max_iou_idx = ious.argmax(dim=1)  # (batch_size, n_anchors)

        # mask for where there are more than one GT box assigned to anchor
        multi_gt_mask = (fg_mask.unsqueeze(1) > 1).expand(-1, num_max_boxes, -1)  # (batch_size, num_max_boxes, n_anchors)

        # mask for GT box with highest IoU
        max_iou_mask = torch.zeros_like(mask, dtype=torch.bool)
        max_iou_mask.scatter_(dim=1, index=max_iou_idx.unsqueeze(1), value=1)

        mask = torch.where(multi_gt_mask, max_iou_mask, mask)
        fg_mask = mask.sum(dim=1)

    target_gt_box_idxs = mask.argmax(dim=1)  # (batch_size, n_anchors)
    return target_gt_box_idxs, fg_mask, mask

class TaskAlignedAssigner(nn.Module):
    """
    Task-aligned assigner for object detection  
    """
    def __init__(self, topk:int=10, num_classes:int=80, alpha:float=1.0, beta:float=6.0, eps:float=1e-8, device:str='cuda'):
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.device = device

        self.bg_idx = num_classes  # no object (background)

    @torch.no_grad()
    def forward(
        self,
        pred_scores:torch.Tensor,
        pred_boxes:torch.Tensor,
        anchor_points:torch.Tensor,
        gt_labels:torch.Tensor,
        gt_boxes:torch.Tensor,
        gt_mask:torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Assignment works in 4 steps:
        1. Compute alignment metric between all predicted bboxes (at all scales) and GT
        2. Select top-k bbox as candidates for each GT
        3. Limit positive sample's center in GT (anchor-free detector can only predict positive distances)
        4. If anchor box is assigned to multiple GT, select the one with highest IoU

        Args:
            pred_scores (Tensor): Predicted scores of shape (batch_size, num_anchors, num_classes)
            pred_boxes (Tensor): Predicted boxes of shape (batch_size, num_anchors, 4)
            anchor_points (Tensor): Anchor points of shape (num_anchors, 2)
            gt_labels (Tensor): GT labels of shape (batch_size, num_max_boxes, 1)
            gt_boxes (Tensor): GT boxes of shape (batch_size, num_max_boxes, 4)
            gt_mask (Tensor): GT mask of shape (batch_size, num_max_boxes, 1)

        Returns:
            target_labels (Tensor): Target labels of shape (batch_size, num_anchors)
            target_boxes (Tensor): Target boxes of shape (batch_size, num_anchors, 4)
            target_scores (Tensor): Target scores of shape (batch_size, num_anchors, num_classes)
        """
        num_max_boxes = gt_boxes.shape[1]

        # If there are no GT boxes, all boxes are background
        if num_max_boxes == 0:
            return (torch.full_like(pred_scores[..., 0], self.bg_idx).to(self.device),
                    torch.zeros_like(pred_boxes).to(self.device),
                    torch.zeros_like(pred_scores).to(self.device))
        
        mask, align_metrics, ious = self.get_positive_mask(
            pred_scores, pred_boxes, anchor_points, gt_labels, gt_boxes, gt_mask
        )

        # Select GT box with highest IoU for each anchor
        target_gt_box_idxs, fg_mask, mask = select_highest_iou(mask, ious, num_max_boxes)

        target_labels, target_boxes, target_scores = self.get_targets(gt_labels, gt_boxes, target_gt_box_idxs, fg_mask)

        # Normalize
        align_metrics *= mask
        positive_align_metrics = align_metrics.amax(dim=-1, keepdim=True)  # (batch_size, num_max_boxes)
        positive_ious = (ious * mask).amax(dim=-1, keepdim=True)  # (batch_size, num_max_boxes)
        align_metrics_norm = (align_metrics * positive_ious / (positive_align_metrics + self.eps)).amax(dim=-2).unsqueeze(-1)
        target_scores = target_scores * align_metrics_norm

        return target_labels, target_boxes, target_scores, fg_mask.bool()
        
    def get_positive_mask(self, pred_scores, pred_boxes, anchor_points, gt_labels, gt_boxes, gt_mask):
        mask_anchors_in_gt = anchors_in_gt_boxes(anchor_points, gt_boxes)

        alignment_metrics, ious = self.get_alignment_metric(pred_scores, pred_boxes, gt_labels, gt_boxes, mask_anchors_in_gt * gt_mask)

        topk_mask = self.get_topk_candidates(alignment_metrics, mask=gt_mask.expand(-1, -1, self.topk))

        # merge masks (batch_size, num_max_boxes, n_anchors)
        mask = topk_mask * mask_anchors_in_gt * gt_mask

        return mask, alignment_metrics, ious

    def get_alignment_metric(self, pred_scores, pred_boxes, gt_labels, gt_boxes, mask):
        """
        Compute alignment metric
        """
        batch_size, n_anchors, _ = pred_scores.shape
        num_max_boxes = gt_boxes.shape[1]

        ious = torch.zeros((batch_size, num_max_boxes, n_anchors), dtype=pred_boxes.dtype, device=pred_boxes.device)
        box_cls_scores = torch.zeros((batch_size, num_max_boxes, n_anchors), dtype=pred_scores.dtype, device=pred_scores.device)

        batch_idxs = torch.arange(batch_size).unsqueeze_(1).expand(-1, num_max_boxes).to(torch.long)  # (bs, num_max_boxes)
        class_idxs = gt_labels.squeeze(-1).to(torch.long)  # (bs, num_max_boxes)

        # Scores for each grid for each GT cls
        box_cls_scores[mask] = pred_scores[batch_idxs, :, class_idxs][mask]  # (bs, num_max_boxes, num_anchors)

        masked_pred_boxes = pred_boxes.unsqueeze(1).expand(-1, num_max_boxes, -1, -1)[mask]  # (bs, num_max_boxes, 1, 4)
        masked_gt_boxes = gt_boxes.unsqueeze(2).expand(-1, -1, n_anchors, -1)[mask]  # (bs, 1, num_anchors, 4)
        ious[mask] = bbox_iou(masked_gt_boxes, masked_pred_boxes, xywh=False).squeeze(-1).clamp_(min=0)

        alignment_metrics = box_cls_scores.pow(self.alpha) * ious.pow(self.beta)

        return alignment_metrics, ious
    
    def get_topk_candidates(self, alignment_metrics:torch.Tensor, mask:torch.Tensor):
        """
        Select top-k candidates for each GT
        """
        # (batch_size, num_max_boxes, topk)
        topk_metrics, topk_idxs = torch.topk(alignment_metrics, self.topk, dim=-1, largest=True)

        # Take max of topk alignment metrics, only take those that are positive
        # make same dimension as topk_idxs
        if mask is None:
            mask = (topk_metrics.amax(dim=-1, keepdim=True) > self.eps).expand_as(topk_idxs)
        
        # Fill values that have negative alignment metric with 0 idx
        topk_idxs.masked_fill_(~mask, 0)

        counts = torch.zeros(alignment_metrics.shape, dtype=torch.int8, device=topk_idxs.device)  # (batch_size, num_max_boxes, n_anchors)
        increment = torch.ones_like(topk_idxs[:,:,:1], dtype=torch.int8, device=topk_idxs.device)  # (batch_size, num_max_boxes, 1)

        for i in range(self.topk):
            counts.scatter_add_(dim=-1, index=topk_idxs[:,:,i:i+1], src=increment)

        # If more than 1, filter out
        counts.masked_fill_(counts > 1, 0)

        return counts.to(alignment_metrics.dtype)
    
    def get_targets(self, gt_labels:torch.Tensor, gt_boxes:torch.Tensor, target_gt_box_idx:torch.Tensor, mask:torch.Tensor):
        """
        Get target labels, bboxes, scores for positive anchor points.

        Args:
            gt_labels (Tensor): GT labels of shape (batch_size, num_max_boxes, 1)
            gt_boxes (Tensor): GT boxes of shape (batch_size, num_max_boxes, 4)
            target_gt_box_idx (Tensor): Index of GT box with highest IoU for each anchor of shape (batch_size, n_anchors)
            mask (Tensor): Mask of shape (batch_size, num_max_boxes, n_anchors) where 1 indicates positive (foreground) anchor

        Returns:
            target_labels (Tensor): Target labels for each positive anchor of shape (batch_size, num_anchors)
            target_boxes (Tensor): Target boxes for each positive anchor of shape (batch_size, num_anchors, 4)
            target_scores (Tensor): Target scores for each positive anchor of shape (batch_size, num_anchors, num_classes)
        """
        batch_size, num_max_boxes, _ = gt_boxes.shape
        _, num_anchors = target_gt_box_idx.shape

        batch_idxs = torch.arange(batch_size, device=gt_labels.device).unsqueeze(-1)

        target_gt_box_idx = target_gt_box_idx + batch_idxs * num_max_boxes

        target_labels = gt_labels.long().flatten()[target_gt_box_idx]  # (batch_size, num_anchors)
        target_labels.clamp_(min=0, max=self.num_classes)

        # (batch_size, max_num_boxes, 4) -> (batch_size, num_anchors, 4)
        target_boxes = gt_boxes.view(-1, 4)[target_gt_box_idx]  # (batch_size, num_anchors, 4)

        # One hot encode (equivalent to doing F.one_hot())
        target_scores = torch.zeros((batch_size, num_anchors, self.num_classes), dtype=torch.int64, device=target_labels.device)
        target_scores.scatter_(dim=2, index=target_labels.unsqueeze(-1), value=1)

        scores_mask = mask.unsqueeze(-1).expand(-1, -1, self.num_classes)  # (batch_size, num_anchors, num_classes)
        target_scores = torch.where(scores_mask > 0, target_scores, 0)

        return target_labels, target_boxes, target_scores


        


    