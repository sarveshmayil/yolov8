import logging

import torch
import torch.nn as nn

from .modules import Conv, C2f, SPPF, DetectionHead

from typing import Tuple

def parse_config(config_dict:dict, verbose=False) -> Tuple[nn.Module, set]:
    if verbose:
        log = logging.getLogger("yolo")
        logging.basicConfig(level=logging.INFO)
    
    depth, width, max_channels = config_dict['scale']

    num_classes = config_dict['num_classes']

    channels = [config_dict.get('in_channels', 3)]

    modules = []
    save_idxs = set()

    if verbose:
        log.info(f'{"idx":>4} | {"Module Type":>14} | {"Input idx(s)":>12} | Args')
        log.info('-'*60)

    # Loop through backbone and head layers
    for i, (module, f, r, args) in enumerate(config_dict['backbone']+config_dict['head']):
        module = getattr(torch.nn, module[3:]) if module.startswith('nn.') else globals()[module]
        if module in (Conv, C2f, SPPF):
            # Get input/output channel sizes
            c_in = channels[f] if isinstance(f, int) else sum([channels[idx] for idx in f])
            c_out = args[0]

            if c_out != num_classes:
                c_out = int(min(c_out, max_channels) * width)

            if module == C2f:
                args = [c_in, c_out, max(round(r*depth), 1), *args[1:]]
            else:
                args = [c_in, c_out, *args[1:]]

        elif module in (DetectionHead,):
            args.append([channels[idx] for idx in f])

        if verbose:
            log.info(f'{i:>4} | {module.__name__:>14} | {str(f):>12} | {args}')

        m_ = module(*args)
        modules.append(m_)
        m_.i, m_.f = i, f

        save_idxs.update([f] if isinstance(f, int) else f)

        # Remove initial channel amount
        # (only needed for first Conv layer)
        if i == 0:
            channels = []
        channels.append(c_out)

    save_idxs.remove(-1)

    if verbose:
        log.info(f' Will save at indices: {save_idxs}')

    return nn.Sequential(*modules), save_idxs