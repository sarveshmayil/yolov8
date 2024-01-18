import os
import argparse
import yaml
import torch
from tqdm import trange

from model.models.detection_model import DetectionModel
from model.data.dataset import Dataset

from torch.utils.data import DataLoader


def get_args():
    parser = argparse.ArgumentParser(description='YOLOv8 model training')
    parser.add_argument(
        '--model_config',
        type=str,
        default='model/config/models/yolov8n.yaml',
        help='path to model config file'
    )
    parser.add_argument(
        '--weights',
        type=str,
        help='path to weights file'
    )

    parser.add_argument(
        '--train_config',
        type=str,
        default='model/config/training/fine_tune.yaml',
        help='path to training config file'
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
        default='train',
        help='dataset mode'
    )

    parser.add_argument(
        '--device',
        '-d',
        type=str,
        default='cuda',
        help='device to model on'
    )

    parser.add_argument(
        '--save',
        '-s',
        action='store_true',
        help='save trained model weights'
    )

    return parser.parse_args()


def main(args):
    train_config = yaml.safe_load(open(args.train_config, 'r'))

    device = torch.device(args.device)
    model = DetectionModel(args.model_config, device=device)
    if args.weights is not None:
        model.load(torch.load(args.weights))

    optimizer = torch.optim.Adam(model.parameters(), lr=train_config['lr'])

    dataset = Dataset(args.dataset, mode=args.dataset_mode)
    dataloader = DataLoader(dataset, batch_size=dataset.batch_size, shuffle=True, collate_fn=Dataset.collate_fn)

    if args.save:
        save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 train_config['save_dir'],
                                 os.path.splitext(os.path.basename(args.model_config))[0])
        os.makedirs(save_path, exist_ok=True)

    for epoch in trange(train_config['epochs']):
        for batch in dataloader:
            loss = model.loss(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch+1) % train_config['save_freq'] == 0 and args.save:
            model.save(os.path.join(save_path, f'{epoch+1}.pt'))


if __name__ == '__main__':
    args = get_args()
    main(args)
    