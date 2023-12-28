import torch
import torch.nn as nn

from .modules import Conv, C2f, SPPF, DetectionHead

from typing import Tuple

def parse_config(config_dict:dict, verbose=False) -> Tuple[nn.Module, set]:
    depth, width, max_channels = config_dict['scale']

    channels = [config_dict.get('in_channels', 3)]

    modules = []
    save_idxs = set()

    for i, (module, f, r, args) in enumerate(config_dict['backbone']+config_dict['head']):
        module = getattr(torch.nn, module[3:]) if module.startswith('nn.') else globals()[module]
        if module in (Conv, C2f, SPPF):
            c_in = channels[f] if isinstance(f, int) else sum([channels[idx] for idx in f])
            c_out = args[0]

            if module == C2f:
                args = [c_in, c_out, max(round(r*depth), 1), *args[1:]]
            else:
                args = [c_in, c_out, *args[1:]]

        elif module in (DetectionHead,):
            args.append([channels[idx] for idx in f])

        m_ = module(*args)
        modules.append(m_)
        m_.i, m_.f = i, f

        save_idxs.update([f] if isinstance(f, int) else f)

        # Remove initial channel amount
        # (only needed for first Conv layer)
        if i == 0:
            channels = []
        channels.append(c_out)

    return nn.Sequential(*modules), save_idxs