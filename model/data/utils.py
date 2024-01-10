import torch
import torch.nn.functional as F

def pad_to(x:torch.Tensor, stride:int):
    """
    Pads an image with zeros to make it divisible by stride
    (Pads both top/bottom and left/right evenly)

    Args:
        x (Tensor): image tensor of shape (..., h, w)
        stride (int): stride of model
    """
    h, w = x.shape[-2:]

    h_new = h if h % stride == 0 else h + stride - h % stride
    w_new = w if w % stride == 0 else w + stride - w % stride

    t, b = int((h_new-h) / 2), int(h_new-h) - int((h_new-h) / 2)
    l, r = int((w_new-w) / 2), int(w_new-w) - int((w_new-w) / 2)
    pads = (l, r, t, b)

    x_padded = F.pad(x, pads, "constant", 0)

    return x_padded, pads

def unpad(x:torch.Tensor, pads:tuple):
    l, r, t, b = pads
    return x[..., t:-b, l:-r]