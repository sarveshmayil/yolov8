from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from typing import Union

class BaseModel(nn.Module, ABC):
    def __init__(self):
        super(BaseModel, self).__init__()

    def load(self, weights:Union[dict, nn.Module]):
        model = weights if isinstance(weights, nn.Module) else weights['model']
        state_dict = model.float().state_dict()
        self.load_state_dict(state_dict)

    def forward(self, x:torch.Tensor, *args, **kwargs):
        return self.predict(x, *args, **kwargs)

    @abstractmethod
    def predict(self, x:torch.Tensor, *args, **kwargs):
        raise NotImplementedError

    def save(self, path:str):
        torch.save(self.state_dict(), path)