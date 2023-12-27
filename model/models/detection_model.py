import yaml
import torch
import torch.nn as nn

from model.models.base_model import BaseModel
from model.modules.conv import Conv
from model.modules.block import SPPF, C2f
from model.modules.head import DetectionHead

from model.modules.utils import parse_config

class DetectionModel(BaseModel):
    def __init__(self, config:str, in_channels:int, n_classes=None):
        super().__init__()

        config = config if isinstance(config, dict) else yaml.safe_load(open(config, 'r'))

        self.model, self.save_idxs = parse_config(config)

        detect_head = self.model[-1]
        s = 256
        detect_head.strides = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, in_channels, s, s))])


class Backbone(nn.Module):
    def __init__(self, in_channels:int, depth, width, max_channels):
        super().__init__()

        self.model = nn.Sequential(
            # 0 - P1
            # (640,640,3) -> (320,320,64)
            Conv(in_channels, 64, kernel_size=3, stride=2),

            # 1 - P2
            # (320,320,64) -> (160,160,128)
            Conv(64, 128, kernel_size=3, stride=2),
            C2f(128, 128, n=max(round(3*depth), 1), shortcut=True),

            # 3 - P3
            # (160,160,128) -> (80,80,256)
            Conv(128, 256, kernel_size=3, stride=2),
            C2f(256, 256, n=max(round(6*depth), 1), shortcut=True),

            # 5 - P4
            # (80,80,256) -> (40,40,512)
            Conv(256, 512, kernel_size=3, stride=2),
            C2f(512, 512, n=max(round(6*depth), 1), shortcut=True),

            # 7 - P5
            # (40,40,512) -> (20,20,1024)
            Conv(512, 1024, kernel_size=3, stride=2),
            C2f(1024, 1024, n=max(round(3*depth), 1), shortcut=True),

            # (20,20,1024) -> (20,20,1024)
            SPPF(1024, 1024, kernel_size=5)
        )

    def forward(self, x:torch.Tensor):
        return self.model(x)
