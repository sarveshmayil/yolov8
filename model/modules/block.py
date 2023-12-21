import torch
import torch.nn as nn

from .conv import Conv

class SPPF(nn.Module):
    "Spatial Pyramid Pooling - Fast"
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int=5, hidden_size:int=None):
        super().__init__()
        if hidden_size is None:
            hidden_size = in_channels // 2
        self.conv1 = Conv(in_channels, hidden_size, kernel_size=1, stride=1)
        self.conv2 = Conv(hidden_size*4, out_channels, kernel_size=1, stride=1)
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x:torch.Tensor):
        x = self.conv1(x)
        y1 = self.maxpool(x)
        y2 = self.maxpool(y1)
        return self.conv2(torch.cat((x, y1, y2, self.maxpool(y2)), dim=1))
    

class C2f(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, n=1, shortcut:bool=False, g=1, e=0.5, hidden_size:int=None):
        super().__init__()
        if isinstance(hidden_size, int):
            self.hidden_size = hidden_size
        else:
            self.hidden_size = int(out_channels * e)

        self.conv1 = Conv(in_channels, 2*self.hidden_size, kernel_size=1, stride=1)
        self.conv2 = Conv((n+2)*self.hidden_size, out_channels, kernel_size=1, stride=1)
        self.conv3 = Conv(out_channels, int(out_channels * e), kernel_size=1, stride=1)


class DFL(nn.Module):
    """Module for Distribution Focal Loss"""

    def __init__(self, in_channels:int=16):
        super().__init__()
        self.in_channels = in_channels
        self.conv = nn.Conv2d(in_channels, out_channels=1, kernel_size=1, bias=False).requires_grad_(False)
        x = torch.arange(in_channels, dtype=torch.float)
        self.conv.weight.data = nn.Parameter(x.view(1, in_channels, 1, 1))

    def forward(self, x:torch.Tensor):
        b, C, a = x.shape  # (batch, channels, anchors)
        return self.conv(x.view(b, 4, self.in_channels, a).transpose(2,1).softmax(dim=1)).view(b, 4, a)