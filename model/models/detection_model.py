import torch
import torch.nn as nn

from model.models.base_model import BaseModel
from model.modules.conv import Conv
from model.modules.block import SPPF, C2f
from model.modules.head import DetectionHead

class DetectionModel(BaseModel):
    def __init__(self, config:str, in_channels:int, n_classes=None):
        super().__init__()

        depth = 0.33
        width = 0.25
        max_channels = 1024

        ### Backbone
        # 0 - P1
        # (640,640,3) -> (320,320,64,w)
        self.conv0 = Conv(in_channels, 64, kernel_size=3, stride=2)

        # 1 - P2
        # (320,320,64,w) -> (160,160,128,w)
        self.conv1 = Conv(64, 128, kernel_size=3, stride=2)
        self.c2f2 = C2f(128, 128, n=max(round(3*depth), 1), shortcut=True)

        # 3 - P3
        # (160,160,128,w) -> (80,80,256,w)
        self.conv3 = Conv(128, 256, kernel_size=3, stride=2)
        self.c2f4 = C2f(256, 256, n=max(round(6*depth), 1), shortcut=True)

        # 5 - P4
        # (80,80,256,w) -> (40,40,512,w)
        self.conv5 = Conv(256, 512, kernel_size=3, stride=2)
        self.c2f6 = C2f(512, 512, n=max(round(6*depth), 1), shortcut=True)

        # 7 - P5
        # (40,40,512,w) -> (20,20,1024,w)
        self.conv7 = Conv(512, 1024, kernel_size=3, stride=2)
        self.c2f8 = C2f(1024, 1024, n=max(round(3*depth), 1), shortcut=True)

        # (20,20,1024,w) -> (20,20,1024,w,r)
        self.sppf9 = SPPF(1024, 1024, kernel_size=5)

        ### Head
        # (20,20,1024,w,r) -> (40,40,1024,w,r)
        self.upsample10 = nn.Upsample(size=None, scale_factor=2, mode='nearest')

        # (40,40,512+1024,w,1+r) -> (40,40,512,w)
        self.c2f12 = C2f(512+1024, 512, n=max(round(3*depth), 1))

        # (40,40,512,w) -> (80,80,512,w)
        self.upsample13 = nn.Upsample(size=None, scale_factor=2, mode='nearest')

        # (80,80,256+512) -> (80,80,256)
        self.c2f15 = C2f(256+512, 256, n=max(round(3*depth), 1))

        # (80,80,256) -> (40,40,256)
        self.conv16 = Conv(256, 256, kernel_size=3, stride=2)

        # (40,40,512+256) -> (40,40,512)
        self.c2f18 = C2f(512+256, 512, n=max(round(3*depth), 1))

        # (40,40,512) -> (20,20,512)
        self.conv19 = Conv(512, 512, kernel_size=3, stride=2)

        # (20,20,1024+512) -> (20,20,1024)
        self.c2f21 = C2f(1024+512, 1024, n=max(round(3*depth), 1))

        self.detect = DetectionHead(n_classes=n_classes, in_channels=[256, 512, 1024])

    def predict(self, x: torch.Tensor, *args, **kwargs):
        # Backbone
        out4 = self.c2f4(self.conv3(self.c2f2(self.conv1(self.conv0(x)))))
        out6 = self.c2f6(self.conv5(out4))
        out9 = self.sppf9(self.c2f8(self.conv7(out6)))

        # Head
        out12 = self.c2f12(torch.cat((out6, self.upsample10(out9)), dim=1))

        out15 = self.c2f15(torch.cat((out4, self.upsample13(out12)), dim=1))

        out18 = self.c2f18(torch.cat((out12, self.conv16(out15)), dim=1))

        out21 = self.c2f21(torch.cat((out9, self.conv19(out18)), dim=1))

        return self.detect([out15, out18, out21])


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
