import os
import yaml
from glob import glob
import logging

import cv2
import numpy as np
import torch
from math import ceil

from model.data.utils import pad_to

from typing import Tuple

log = logging.getLogger("dataset")
logging.basicConfig(level=logging.DEBUG)


class Dataset(torch.utils.data.Dataset):
    """
    Dataset class for loading images and annotations.

    Args:
        config (str): path to dataset config file
        batch_size (optional, int): batch size for dataloader
        mode (optional, str): dataset mode (train, val, test)
        img_size (optional, Tuple[int,int]): image size to pad images to
    """
    def __init__(self, config:str, batch_size:int=8, mode:str='train', img_size:Tuple[int,int]=(640, 640)):
        super().__init__()
        self.config = yaml.safe_load(open(config, 'r'))
        self.dataset_path = os.path.join(os.path.dirname(config), self.config['path'])
        self.batch_size = batch_size
        self.img_size = img_size

        assert mode in ('train', 'val', 'test'), f'Invalid mode: {mode}'
        self.mode = mode

        self.im_files = self.get_image_paths()
        log.debug(f'Found {len(self.im_files)} images in {os.path.join(self.dataset_path, self.config[self.mode])}')

        self.label_files = self.get_label_paths()
        if self.label_files is not None:
            log.debug(f'Found {len(self.label_files)} labels in {os.path.join(self.dataset_path, self.config[self.mode+"_labels"])}')
        else:
            log.debug(f'No labels found in {os.path.join(self.dataset_path, self.config[self.mode+"_labels"])}')

        self.labels = self.get_labels()

    def get_image_paths(self):
        """
        Get image paths from dataset directory

        Searches recursively for .jpg, .png, and .jpeg files.
        """
        im_dir = os.path.join(self.dataset_path, self.config[self.mode])

        image_paths = glob(os.path.join(im_dir, '*.jpg')) + \
                      glob(os.path.join(im_dir, '*.png')) + \
                      glob(os.path.join(im_dir, '*.jpeg'))
 
        return image_paths
    
    def get_label_paths(self):
        """
        Get label paths from dataset directory

        Uses ids from image paths to find corresponding label files.

        If no label directory is found, returns None.
        """
        label_dir = os.path.join(self.dataset_path, self.config[self.mode+'_labels'])
        if os.path.isdir(label_dir):
            return [os.path.join(label_dir, os.path.splitext(os.path.basename(p))[0]+".txt") for p in self.im_files]
        return None
    
    def get_labels(self):
        """
        Gets labels from label files (assumes COCO formatting)

        Returns a list of dictionaries for each file
            {
                'cls': torch.Tensor of shape (num_boxes,)
                'bboxes': torch.Tensor of shape (num_boxes, 4) in (xywh) format
            }

        If no label files were found, returns a list of empty dictionaries.
        """
        if self.label_files is None:
            return [{} for _ in range(len(self.im_files))]
        labels = []
        for label_file in self.label_files:
            annotations = open(label_file, 'r').readlines()
            cls, boxes = [], []
            for ann in annotations:
                ann = ann.strip('\n').split(' ')
                cls.append(int(ann[0]))

                # box provided in xywh format
                boxes.append(torch.from_numpy(np.array(ann[1:5], dtype=float)))

            labels.append({
                'cls': torch.tensor(cls),
                'bboxes': torch.vstack(boxes)
            })
        return labels
    
    def load_image(self, idx):
        """
        Loads image at specified index and prepares for model input.

        Changes image shape to be specified img_size, but preserves aspect ratio.
        """
        im_file = self.im_files[idx]
        image = cv2.cvtColor(cv2.imread(im_file), cv2.COLOR_BGR2RGB)

        h0, w0 = image.shape[:2]

        if h0 > self.img_size[0] or w0 > self.img_size[1]:
            # Resize to have max dimension of img_size, but preserve aspect ratio
            ratio = min(self.img_size[0]/h0, self.img_size[1]/w0)
            h, w = min(ceil(h0*ratio), self.img_size[0]), min(ceil(w0*ratio), self.img_size[1])
            image = cv2.resize(image, (h, w), interpolation=cv2.INTER_LINEAR)

        image = image.transpose((2, 0, 1))  # (h, w, 3) -> (3, h, w)
        image = torch.from_numpy(image).float() / 255.0
        
        # Pad image with black bars to desired img_size
        image, pads = pad_to(image, shape=self.img_size)

        h, w = image.shape[-2:]

        return image, (h0, w0), (h, w)

    def get_image_and_label(self, idx):
        """
        Gets image and annotations at specified index
        """
        label = self.labels[idx]
        # label['images'], label['orig_shapes'], label['shapes'] = self.load_image(idx)
        label['images'], _, _ = self.load_image(idx)

        return label

    def __len__(self) -> int:
        return len(self.im_files)

    def __getitem__(self, index):
        return self.get_image_and_label(index)
    
    @staticmethod
    def collate_fn(batch):
        """
        Collate function to specify how to combine a list of samples into a batch
        """
        collated_batch = {}
        for k in batch[0].keys():
            if k == "images":
                collated_batch[k] = torch.stack([b[k] for b in batch], dim=0)
            elif k in ('cls', 'bboxes'):
                collated_batch[k] = torch.cat([b[k] for b in batch], dim=0)
        
        collated_batch['batch_idx'] = [torch.full((batch[i]['cls'].shape[0],), i) for i in range(len(batch))]
        collated_batch['batch_idx'] = torch.cat(collated_batch['batch_idx'], dim=0)
                
        return collated_batch
