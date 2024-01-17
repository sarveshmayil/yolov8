import argparse
import yaml
import os
import cv2
import numpy as np
from matplotlib import colormaps as cm


def get_args():
    parser = argparse.ArgumentParser(description='Display annotations')

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='path to dataset config file'
    )
    parser.add_argument(
        '--dataset_mode',
        type=str,
        default='val',
        help='dataset mode - (train/val/test)'
    )
    parser.add_argument(
        '--labels',
        '-l',
        type=str,
        default='labels/val',
        help='relative path to label directory'
    )

    return parser.parse_args()

def visualize_dataset(args):
    config_dict = yaml.safe_load(open(args.config, 'r'))

    im_dir = os.path.join(os.path.dirname(args.config), config_dict['path'], config_dict[args.dataset_mode])
    label_dir = os.path.join(os.path.dirname(args.config), config_dict['path'], args.labels)

    num_classes = len(config_dict['names'])

    cmap = cm['jet']

    for im_file in os.listdir(im_dir):
        id = os.path.splitext(im_file)[0]
        image = cv2.imread(os.path.join(im_dir, im_file))
        annotations = open(os.path.join(label_dir, id+'.txt'), 'r').readlines()

        H, W = image.shape[:2]

        for ann in annotations:
            ann = ann.strip('\n').split(' ')
            cls = int(ann[0])
            label = config_dict['names'][cls]

            # box provided in xywh format
            xywh = np.array(ann[1:5], dtype=float)
            xywh = xywh * [W, H, W, H]
            xywh = xywh.astype(int)

            x1, y1, x2, y2 = xywh[0]-xywh[2]//2, xywh[1]-xywh[3]//2, xywh[0]+xywh[2]//2, xywh[1]+xywh[3]//2

            if len(ann) > 5:
                confidence = float(ann[5])
                label += f' {confidence:.2f}'

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
    visualize_dataset(args)