import torch
import torch.nn as nn


__all__ = ('autopad', 'dist2bbox', 'init_weights')


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
