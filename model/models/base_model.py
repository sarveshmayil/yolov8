import torch
import torch.nn as nn

from model.modules import Conv, C2f, SPPF, DetectionHead
from model.utils.loss import BaseLoss

from typing import Union

class BaseModel(nn.Module):
    model:nn.ModuleList
    save_idxs:set
    loss_fn:BaseLoss

    def __init__(self, device='cpu'):
        super().__init__()

        self.device = device

        self.model = None
        self.save_idxs = set()

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
            # If not just using previous module output
            if module.f != -1:
                # Get list of inputs for module
                x = outputs[module.f] if isinstance(module.f, int) else [x if i == -1 else outputs[i] for i in module.f]
                # Don't concat if module is DetectionHead (it takes in a list)
                if isinstance(x, list) and not isinstance(module, DetectionHead):
                    x = torch.cat(x, dim=1)
            x = module(x)

            outputs.append(x if module.i in self.save_idxs else None)

        return x
    
    def loss(self, batch:torch.Tensor):
        preds = self.forward(batch)
        return self.loss_fn.compute_loss(batch, preds)

    def save(self, path:str):
        torch.save(self.state_dict(), path)