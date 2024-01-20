# <div align="center">YOLOv8</div>
A full from scratch re-implementation of Ultralytics YOLOv8

YOLOv8 by [Ultralytics](https://www.ultralytics.com/) is a SOTA model that is designed to be highly accurate and fast.
This re-implementation only implements object detection and tracking, but could easily be extrapolated to the other tasks of pose estimation, instance segmentation, and image classification.

## <div align="center">Setup</div>

The python setup uses `poetry` which can be installed using
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

To install the dependencies with poetry,
```bash
poetry shell
poetry install
```

## <div align="center">Usage</div>

### Training

To train or fine-tune a model, use the `train.py` script. For example, to fine-tune the pretained YOLOv8n model on the coco128 dataset and save the weights:
```bash
python3 train.py --weights model/weights/yolov8n.pt \
                 --train-config model/config/training/fine_tune.yaml \
                 --dataset model/config/datasets/coco128.yaml \
                 --save
```

### Inference

To perform inference with a model, use the `inference.py` script. For example, to evaluate a model on a particular dataset:
```bash
python3 inference.py --weights model/weights/yolov8n.pt \
                     --dataset model/config/datasets/coco8.yaml \
                     -v
```

## <div align="center">Using a Pre-trained YOLOv8 Model</div>

In order to use a pre-trained YOLOv8 model, such as `yolov8m`, use the `convert_yolo_weights.py` script. This allows you to convert an Ultralytics YOLO model to one that can be used with this code.

Note that you will need to have `ultralytics` installed in whatever python environment you choose to run this script in as it is necessary to read the `.pt` file.

## <div align="center">Resources</div>

Some resources that were referenced in order to write this code (in addition to the [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) library itself) are listed below.

- https://arxiv.org/abs/2304.00501
- https://openmmlab.medium.com/dive-into-yolov8-how-does-this-state-of-the-art-model-work-10f18f74bab1