import torch
import torch.nn as nn
from math import log

from .conv import Conv
from .block import DFL
from .utils import dist2bbox

from typing import List

__all__ = ('DetectionHead')

class DetectionHead(nn.Module):
    stride:torch.Tensor
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, n_classes:int=80, in_channels:List[int]=[]):
        super().__init__()
        self.nc = n_classes
        self.n_layers = len(in_channels)
        self.reg_max = 16
        self.n_outputs = 4*self.reg_max + self.nc
        self.stride = torch.zeros(self.n_layers)

        c2 = max((16, in_channels[0]//4, self.reg_max*4))
        c3 = max(in_channels[0], min(self.nc, 100))

        self.box_convs = nn.ModuleList(
            nn.Sequential(
                Conv(c, c2, kernel_size=3),
                Conv(c2, c2, kernel_size=3),
                nn.Conv2d(c2, 4*self.reg_max, kernel_size=1)
            ) for c in in_channels
        )

        self.cls_convs = nn.ModuleList(
            nn.Sequential(
                Conv(c, c3, kernel_size=3),
                Conv(c3, c3, kernel_size=3),
                nn.Conv2d(c3, self.nc, kernel_size=1)
            ) for c in in_channels
        )

        self.dfl = DFL(in_channels=self.reg_max)

    def forward(self, x:List[torch.Tensor]):
        shape = x[0].shape  # (batch, channels, height, width)
        for i in range(self.n_layers):
            x[i] = torch.cat((self.box_convs[i](x[i]), self.cls_convs[i](x[i])), dim=1)

        # If training, return predicted box, class prediction
        if self.training:
            return x
        
        x = torch.cat([xi.view(shape[0], self.n_outputs, -1) for xi in x], dim=2)
        box, cls = x.split((4*self.reg_max, self.nc), dim=1)
        bbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), dim=1) * self.strides

        out = torch.cat((bbox, torch.sigmoid(cls)), dim=1)
        return out, x
    
    def _bias_init(self) -> None:
        """
        Initialize biases for Conv2d layers

        Must set stride before calling this method
        """
        if self.stride is None:
            raise ValueError('stride is not set')
        
        for b_list, c_list, s in zip(self.box_convs, self.cls_convs, self.stride):
            b_list[-1].bias.data[:] = 1.0
            c_list[-1].bias.data[:self.nc] = log(5/(self.nc*(640/s)**2))
