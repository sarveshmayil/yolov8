import torch
import torch.nn as nn

from .utils import autopad

__all__ = ('Conv')

class Conv(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int=3, stride:int=1, padding:int=None, bias:bool=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=autopad(kernel_size, padding), bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x:torch.Tensor):
        return self.act(self.bn(self.conv(x)))