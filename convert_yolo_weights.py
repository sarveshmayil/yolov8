import torch
import torch.nn as nn
import argparse

from model.models.detection_model import DetectionModel
import ultralytics
from ultralytics.nn.modules import Conv, C2f, SPPF, Detect


def get_args():
    parser = argparse.ArgumentParser('Convert YOLO weights to new model weights')
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to YOLO weights'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to model config'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        required=False,
        help='Path to output model weights'
    )
    return parser.parse_args()

def convert_yolo(yolo_model, model):
    module_count = 0
    for module in yolo_model.model:
        if (module.__module__.startswith('torch.nn')):
            model.model[module_count] = module
            module_count += 1
        if isinstance(module, Conv):
            convert_conv(module, model.model[module_count])
            module_count += 1
        elif isinstance(module, C2f):
            convert_c2f(module, model.model[module_count])
            module_count += 1
        elif isinstance(module, SPPF):
            convert_sppf(module, model.model[module_count])
            module_count += 1
        elif isinstance(module, Detect):
            convert_detect(module, model.model[module_count])
            module_count += 1

def convert_conv(yolo_conv, conv):
    conv.conv = yolo_conv.conv
    conv.bn = yolo_conv.bn
    conv.act = yolo_conv.act

def convert_c2f(yolo_c2f, c2f):
    c2f.hidden_size = yolo_c2f.c
    convert_conv(yolo_c2f.cv1, c2f.conv1)
    convert_conv(yolo_c2f.cv2, c2f.conv2)
    for i, bottleneck in enumerate(yolo_c2f.m):
        convert_bottleneck(bottleneck, c2f.bottlenecks[i])

def convert_sppf(yolo_sppf, sppf):
    convert_conv(yolo_sppf.cv1, sppf.conv1)
    convert_conv(yolo_sppf.cv2, sppf.conv2)
    sppf.maxpool = yolo_sppf.m

def convert_bottleneck(yolo_bottleneck, bottleneck):
    convert_conv(yolo_bottleneck.cv1, bottleneck.conv1)
    convert_conv(yolo_bottleneck.cv2, bottleneck.conv2)
    bottleneck.residual = yolo_bottleneck.add

def convert_dfl(yolo_dfl, dfl):
    dfl.in_channels = yolo_dfl.c1
    dfl.conv = yolo_dfl.conv

def convert_detect(yolo_detect, detect):
    detect.stride = yolo_detect.stride
    detect.anchors = yolo_detect.anchors
    detect.strides = yolo_detect.strides
    detect.nc = yolo_detect.nc
    detect.n_layers = yolo_detect.nl
    detect.reg_max = yolo_detect.reg_max
    detect.n_outputs = yolo_detect.no

    for i, module in enumerate(yolo_detect.cv2):
        for j, conv in enumerate(module):
            if isinstance(conv, nn.Conv2d):
                detect.box_convs[i][j] = conv
            else:
                convert_conv(conv, detect.box_convs[i][j])

    for i, module in enumerate(yolo_detect.cv3):
        for j, conv in enumerate(module):
            if isinstance(conv, nn.Conv2d):
                detect.cls_convs[i][j] = conv
            else:
                convert_conv(conv, detect.cls_convs[i][j])

    convert_dfl(yolo_detect.dfl, detect.dfl)

if __name__ == '__main__':
    args = get_args()

    model = DetectionModel(args.config)

    yolo_dict = torch.load(args.model, map_location='cpu')
    yolo_model = yolo_dict['model']

    # module_count = 0
    # for i, module in enumerate(yolo_model.model):
    #     if (module.__module__.startswith('torch.nn')) or isinstance(module, (Conv, C2f, SPPF, Detect)):
    #         print(i, type(module), module_count, type(model.model[module_count]))
    #         module_count += 1

    convert_yolo(yolo_model, model)

    if args.output is not None:
        print("Saving model at", args.output)
        torch.save(model.state_dict(), args.output)
