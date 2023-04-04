import argparse
import os.path as osp
import sys

import mmengine
import torch

sys.path.insert(0, sys.path[0]+'/../..')

import cv2
import mmcv
import numpy as np
from mmengine.config import Config, DictAction
from mmengine.dataset import Compose
from mmengine.utils import ProgressBar
from mmengine.visualization import Visualizer

from mmpl.datasets.builder import build_dataset
from mmpl.utils import register_all_modules
from mmpl.datasets.data_utils import lafan1_utils_torch


def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument('--config', default='../../configs/motiongpt_config.py', help='train config file path')
    parser.add_argument(
        '--output-dir',
        '-o',
        default=None,
        type=str,
        help='If there is no display interface, you can save it.')
    parser.add_argument(
        '--phase',
        '-p',
        default='train',
        type=str,
        choices=['train', 'test', 'val'],
        help='phase of dataset to visualize, accept "train" "test" and "val".'
        ' Defaults to "train".')
    parser.add_argument(
        '--show-number',
        '-n',
        type=int,
        default=sys.maxsize,
        help='number of images selected to visualize, must bigger than 0. if '
        'the number is bigger than length of dataset, show all the images in '
        'dataset; default "sys.maxsize", show all images in dataset')
    parser.add_argument(
        '--show-interval',
        '-i',
        type=float,
        default=2,
        help='the interval of show (s)')
    parser.add_argument(
        '--mode',
        '-m',
        default='transformed',
        type=str,
        choices=['original', 'transformed', 'concat', 'pipeline'],
        help='display mode; display original pictures or transformed pictures'
        ' or comparison pictures. "original" means show images load from disk'
        '; "transformed" means to show images after transformed; "concat" '
        'means show images stitched by "original" and "output" images. '
        '"pipeline" means show all the intermediate images. '
        'Defaults to "transformed".')
    parser.add_argument(
        '--rescale-factor',
        '-r',
        type=float,
        help='image rescale factor, which is useful if the output is too '
        'large or too small.')
    parser.add_argument(
        '--channel-order',
        '-c',
        default='BGR',
        choices=['BGR', 'RGB'],
        help='The channel order of the showing images, could be "BGR" '
        'or "RGB", Defaults to "BGR".')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # register all modules in mmcls into the registries
    register_all_modules()

    dataset_cfg = cfg.get('datamodule_cfg').get(args.phase + '_loader').get('dataset')
    dataset_cfg['data_root'] = '../'+dataset_cfg['data_root']
    dataset = build_dataset(dataset_cfg)

    progress_bar = ProgressBar(len(dataset))
    input_data = {}
    for i, item in zip(range(len(dataset)), dataset):
        progress_bar.update()
        # if i % 100 == 0:
        if True:
            positions, rotations, global_positions, global_rotations, foot_contact, parents, _, _, _ = item.values()
            x_positions = positions[:dataset.block_size]
            x_rotations = rotations[:dataset.block_size]
            positions_shift, rotations_shift = lafan1_utils_torch.reduce_frame_root_shift_and_rotation(
                x_positions, x_rotations, base_frame_id=0)
            rot_6d_with_position, diff_root_xz = lafan1_utils_torch.get_shift_model_input(positions_shift, rotations_shift)
            input_data['rot_6d_with_position'] = input_data.get('rot_6d_with_position', []) + [rot_6d_with_position]
            input_data['diff_root_xz'] = input_data.get('diff_root_xz', []) + [diff_root_xz]

    input_data['rot_6d_with_position'] = torch.cat(input_data['rot_6d_with_position'], dim=0)
    input_data['diff_root_xz'] = torch.cat(input_data['diff_root_xz'], dim=0)
    mean_rot_6d_with_position = torch.mean(input_data['rot_6d_with_position'], dim=0)
    std_rot_6d_with_position = torch.std(input_data['rot_6d_with_position'], dim=0)
    mean_diff_root_xz = torch.mean(input_data['diff_root_xz'], dim=0)
    std_diff_root_xz = torch.std(input_data['diff_root_xz'], dim=0)

    max_rot_6d_with_position = torch.max(input_data['rot_6d_with_position'], dim=0)[0]
    min_rot_6d_with_position = torch.min(input_data['rot_6d_with_position'], dim=0)[0]
    max_diff_root_xz = torch.max(input_data['diff_root_xz'], dim=0)[0]
    min_diff_root_xz = torch.min(input_data['diff_root_xz'], dim=0)[0]

    train_stats = {
        'mean_rot_6d_with_position': mean_rot_6d_with_position,
        'std_rot_6d_with_position': std_rot_6d_with_position,
        'mean_diff_root_xz': mean_diff_root_xz,
        'std_diff_root_xz': std_diff_root_xz,
        'max_rot_6d_with_position': max_rot_6d_with_position,
        'min_rot_6d_with_position': min_rot_6d_with_position,
        'max_diff_root_xz': max_diff_root_xz,
        'min_diff_root_xz': min_diff_root_xz,
    }

    mmengine.dump(train_stats, f"../../data/lafan1_train_mean_std_info_{dataset.block_size}.pkl")
    for key, value in train_stats.items():
        train_stats[key] = value.tolist()
    mmengine.dump(train_stats, f"../../data/lafan1_train_mean_std_info_{dataset.block_size}.json", indent=4)


if __name__ == '__main__':
    main()
