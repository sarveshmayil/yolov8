import os
import argparse
import torch
import cv2
from matplotlib import colormaps as cm

from model.models.detection_model import DetectionModel
from model.data.dataset import Dataset
from model.data.detections import Detections

from torch.utils.data import DataLoader


def get_args():
    parser = argparse.ArgumentParser(description='YOLOv8 model inference')
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='path to model config file'
    )
    parser.add_argument(
        '--weights',
        type=str,
        default=None,
        help='path to weights file'
    )
    parser.add_argument(
        '--onnx',
        type=str,
        default=None,
        help='path to ONNX file'
    )

    dataset_args = parser.add_argument_group('Dataset')
    dataset_args.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='path to dataset config file'
    )
    dataset_args.add_argument(
        '--dataset-mode',
        type=str,
        default='val',
        help='dataset mode - (train/val/test)'
    )

    parser.add_argument(
        '--device',
        '-d',
        type=str,
        default='cuda',
        help='device to run inference on'
    )

    parser.add_argument(
        '--visualize',
        '-v',
        action='store_true',
        help='visualize inference results'
    )

    parser.add_argument(
        '--save',
        '-s',
        action='store_true',
        help='save inference results'
    )

    return parser.parse_args()

def main(args):
    device = torch.device(args.device)
    model = DetectionModel(args.config, device=device)
    model.load(torch.load(args.weights))
    model.eval()
    model.mode = 'eval'

    dataset = Dataset(args.dataset, mode=args.dataset_mode)
    dataloader = DataLoader(dataset, batch_size=dataset.batch_size, shuffle=True, collate_fn=Dataset.collate_fn)

    if args.visualize:
        cmap = cm['jet']

    if args.save:
        save_path = os.path.join(os.path.dirname(args.dataset), dataset.config['path'], 'results', args.dataset_mode)
        os.makedirs(save_path, exist_ok=True)

    for batch in dataloader:
        with torch.no_grad():
            preds = model(batch['images'].to(device))

        for i in range(len(preds)):
            detections = Detections.from_yolo(preds[i])

            if args.save:
                detections.save(os.path.join(save_path, batch['ids'][i]+'.txt'), pads=batch['padding'][i], im_size=batch['orig_shapes'][i])

            if args.visualize:
                image = batch['images'][i].detach().cpu().numpy().transpose((1, 2, 0))
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                detections.view(image, classes_dict=dataset.config['names'], cmap=cmap)
                cv2.imshow('annotations', image)
                cv2.waitKey(0)

    cv2.destroyAllWindows()

def main_onnx(args):
    import onnx, onnxruntime
    from model.utils.ops import nms

    model = onnx.load(args.onnx)
    onnx.checker.check_model(model)

    ort_session = onnxruntime.InferenceSession(args.onnx, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    dataset = Dataset(args.dataset, mode=args.dataset_mode)
    dataloader = DataLoader(dataset, batch_size=dataset.batch_size, shuffle=True, collate_fn=Dataset.collate_fn)

    if args.visualize:
        cmap = cm['jet']

    if args.save:
        save_path = os.path.join(os.path.dirname(args.dataset), dataset.config['path'], 'results', args.dataset_mode)
        os.makedirs(save_path, exist_ok=True)

    for batch in dataloader:
        ort_inputs = {ort_session.get_inputs()[0].name: batch['images'].cpu().numpy()}
        ort_outs = ort_session.run(None, ort_inputs)[0]
        preds = nms(torch.from_numpy(ort_outs))

        for i in range(len(preds)):
            detections = Detections.from_yolo(preds[i])

            if args.save:
                detections.save(os.path.join(save_path, batch['ids'][i]+'.txt'), pads=batch['padding'][i], im_size=batch['orig_shapes'][i])

            if args.visualize:
                image = batch['images'][i].detach().cpu().numpy().transpose((1, 2, 0))
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                detections.view(image, classes_dict=dataset.config['names'], cmap=cmap)
                cv2.imshow('annotations', image)
                cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    args = get_args()
    if args.onnx is not None:
        main_onnx(args)
    elif args.config is not None and args.weights is not None:
        main(args)
    else:
        raise ValueError('Invalid arguments, must provide either --onnx or both --config and --weights')
    