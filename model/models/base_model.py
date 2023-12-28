from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from model.modules import Conv, C2f, SPPF, DetectionHead

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

    def predict(self, x:torch.Tensor, *args, **kwargs):
        return self._predict(x, *args, **kwargs)
    
    def _predict(self, x:torch.Tensor, *args, **kwargs):
        outputs = []
        for module in self.model:
            if module.f != -1:
                x = outputs[module.f] if isinstance(module.f, int) else [x if i == -1 else outputs[i] for i in module.f]
                if isinstance(x, list) and not isinstance(module, DetectionHead):
                    x = torch.cat(x, dim=1)
            x = module(x)

            if module.i in self.save_idxs:
                outputs.append(x)
            else:
                outputs.append(None)

        return x

    def save(self, path:str):
        torch.save(self.state_dict(), path)