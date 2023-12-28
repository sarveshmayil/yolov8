import yaml
import torch

from .base_model import BaseModel

from model.misc import parse_config

class DetectionModel(BaseModel):
    def __init__(self, config:str):
        super().__init__()

        config = config if isinstance(config, dict) else yaml.safe_load(open(config, 'r'))
        in_channels = config.get('in_channels', 3)

        self.model, self.save_idxs = parse_config(config)

        detect_head = self.model[-1]
        s = 256
        detect_head.strides = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, in_channels, s, s))])
