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
        default='model/config/models/yolov8n.yaml',
        help='path to model config file'
    )
    parser.add_argument(
        '--weights',
        type=str,
        default='yolov8n_mine.pt',
        help='path to weights file'
    )

    dataset_args = parser.add_argument_group('Dataset')
    dataset_args.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='path to dataset config file'
    )
    dataset_args.add_argument(
        '--dataset_mode',
        type=str,
        default='val',
        help='dataset mode - (train/val/test)'
    )

    parser.add_argument(
        '--device',
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
    model = DetectionModel(args.config)
    model.load_state_dict(torch.load(args.weights))
    model.eval()
    model.mode = 'eval'

    dataset = Dataset(args.dataset, mode=args.dataset_mode)
    dataloader = DataLoader(dataset, batch_size=dataset.batch_size, shuffle=True, collate_fn=Dataset.collate_fn)

    if args.visualize:
        num_classes = len(dataset.config['names'])
        cmap = cm['jet']

    if args.save:
        save_path = os.path.join(os.path.dirname(args.dataset), dataset.config['path'], 'results', args.dataset_mode)
        os.makedirs(save_path, exist_ok=True)

    for batch in dataloader:
        with torch.no_grad():
            preds = model(batch['images'])

        for i in range(len(preds)):
            detections = Detections.from_yolo(preds[i])

            if args.save:
                detections.save(os.path.join(save_path, batch['ids'][i]+'.txt'), pads=batch['padding'][i], im_size=batch['orig_shapes'][i])

            if args.visualize:
                image = batch['images'][i].detach().cpu().numpy().transpose((1, 2, 0))
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                for j in range(len(detections)):
                    x1, y1, x2, y2 = detections.xyxy[j].astype(int)
                    cls = detections.class_id[j]
                    confidence = detections.confidence[j]
                    label = dataset.config['names'][cls] + f' {confidence:.2f}'

                    cls_color = cmap(cls/num_classes, bytes=True)[:3]
                    cls_color = (int(cls_color[0]), int(cls_color[1]), int(cls_color[2]))

                    # draw bounding box
                    cv2.rectangle(image, (x1, y1), (x2, y2), cls_color, 2)

                    # draw label
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    cv2.rectangle(image, (x1, y1), (x1+w, y1-h), cls_color, -1)
                    cv2.putText(image, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

                cv2.imshow('annotations', image)
                cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    args = get_args()
    main(args)
    