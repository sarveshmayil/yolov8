import argparse

import torch.onnx

from model.models.detection_model import DetectionModel


def get_args():
    parser = argparse.ArgumentParser(description='Export model to ONNX/TensorRT')
    parser.add_argument(
        '--config',
        type=str,
        default='model/config/models/yolov8n.yaml',
        help='path to model config file'
    )
    parser.add_argument(
        '--weights',
        type=str,
        default='model/weights/yolov8n.pt',
        help='path to weights file'
    )

    parser.add_argument(
        '--output',
        '-o',
        type=str,
        default='model/weights/yolov8n.onnx',
        help='path to output ONNX file'
    )

    parser.add_argument(
        '--device',
        '-d',
        type=str,
        default='cuda',
        help='device to use'
    )

    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='verbose ONNX export'
    )
    parser.add_argument(
        '--im-size',
        type=int,
        default=None,
        help='Set value for fixed image size'
    )

    return parser.parse_args()


def to_onnx(args):
    device = torch.device(args.device)
    model = DetectionModel(args.config, device=device)
    model.load(torch.load(args.weights))
    model.eval()

    size = args.im_size if args.im_size is not None else 640
    dummy_input = torch.randn(1, 3, size, size, device=device)

    dyn_axes = {'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}}

    if args.im_size is None:
        dyn_axes['input'].update({2: 'height', 3: 'width'})

    torch.onnx.export(model,
                      dummy_input,
                      args.output,
                      input_names=['input'],
                      output_names=['output'],
                      export_params=True,
                      dynamic_axes=dyn_axes,
                      verbose=args.verbose)


if __name__  == "__main__":
    args = get_args()
    to_onnx(args)
    print(f"Successfully exported model to {args.output}")
