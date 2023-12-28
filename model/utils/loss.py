import torch
import torch.nn as nn

from model.models import BaseModel


class BaseLoss:
    def __init__(self, device:str):
        self.device = device

    def compute_loss(self, batch:torch.Tensor, preds:torch.Tensor):
        raise NotImplementedError
