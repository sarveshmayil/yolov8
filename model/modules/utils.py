import torch
import torch.nn as nn


__all__ = ('autopad', 'dist2bbox', 'init_weights', 'make_anchors', 'xywh2xyxy', 'xyxy2xywh')


def autopad(kernel_size:int, padding:int=None):
    """
    Calculate padding size automatically
    """
    if padding is None:
        return kernel_size // 2 if isinstance(kernel_size, int) else [k // 2 for k in kernel_size]
    return padding

def dist2bbox(distance:torch.Tensor, anchor_points:torch.Tensor, xywh:bool=True, dim:int=-1):
    """
    Transform distance in (ltrb) to bounding box (xywh) or (xyxy)
    """

    lt, rb = torch.chunk(distance, 2, dim=dim)
    xy_lt = anchor_points - lt
    xy_rb = anchor_points + rb

    if xywh:
        center = (xy_lt + xy_rb) / 2
        wh = xy_rb - xy_lt
        return torch.cat((center, wh), dim=dim)
    
    return torch.cat((xy_lt, xy_rb), dim=dim)

def bbox2dist(bbox:torch.Tensor, anchor_points:torch.Tensor, reg_max:int):
    """
    Transform bounding box (xyxy) to distance (ltrb)
    """
    xy_lt, xy_rb = torch.chunk(bbox, 2, dim=-1)
    lt = anchor_points - xy_lt
    rb = xy_rb - anchor_points
    return torch.cat((lt, rb), dim=-1).clamp(max=reg_max-0.01)

def init_weights(model:nn.Module):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in (nn.ReLU, nn.SiLU):
            m.inplace = True

def make_anchors(feats:torch.Tensor, strides:torch.Tensor):
    anchor_points = []
    stride_tensor = []

    device = feats[0].device
    dtype = feats[0].dtype

    for i, stride in enumerate(strides):
        h, w = feats[i].shape[-2:]
        yv, xv = torch.meshgrid(torch.arange(h).to(device=device, dtype=dtype) + 0.5,
                                torch.arange(w).to(device=device, dtype=dtype) + 0.5)

        # (x,y) coordinates of center of each cell in grid
        anchor_points.append(torch.stack((xv, yv), dim=-1).view(-1, 2))
        stride_tensor.append(torch.full((h*w,1), stride).to(device=device, dtype=dtype))

    anchor_points = torch.cat(anchor_points, dim=0)
    stride_tensor = torch.cat(stride_tensor, dim=0)

    return anchor_points, stride_tensor

def xywh2xyxy(xywh:torch.Tensor):
    """
    Convert bounding box coordinates from (xywh) to (xyxy)
    """
    xy, wh = torch.chunk(xywh, 2, dim=-1)
    return torch.cat((xy - wh / 2, xy + wh / 2), dim=-1)

def xyxy2xywh(xyxy:torch.Tensor):
    """
    Convert bounding box coordinates from (xyxy) to (xywh)
    """
    xy_lt, xy_rb = torch.chunk(xyxy, 2, dim=-1)
    return torch.cat(((xy_lt + xy_rb) / 2, xy_rb - xy_lt), dim=-1)
