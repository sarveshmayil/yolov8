import torch
import torch.nn as nn

from .conv import Conv
from .block import DFL

from typing import List

class DetectionHead(nn.Module):
    def __init__(self, n_classes:int=80, in_channels:List[int]=[]):
        super().__init__()
        self.nc = n_classes
        self.n_layers = len(in_channels)
        self.reg_max = 16

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
