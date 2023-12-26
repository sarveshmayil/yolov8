import torch


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