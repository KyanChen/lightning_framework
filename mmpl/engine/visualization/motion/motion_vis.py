import glob
import os
import time

import cv2
import mmcv
import mmengine
# os.environ['IMAGEIO_FFMPEG_EXE'] = '/Users/kyanchen/Documents/ffmpeg/ffmpeg'
# https://ffmpeg.org/download.html
import imageio
import numpy as np
import torch
from matplotlib import pyplot as plt
from mmpl.datasets.data_utils import lafan1_utils_torch
from mmpl.registry import HOOKS
from lightning.pytorch.callbacks import Callback


@HOOKS.register_module()
class MotionVisualizer(Callback):
    def __init__(self, save_dir):
        self.save_dir = save_dir
        mmengine.mkdir_or_exist(self.save_dir)
        self.motions = []
        # self.bbox_color = bbox_color
        # self.text_color = text_color
        # self.mask_color = mask_color
        # self.line_width = line_width
        # self.alpha = alpha
        # # Set default value. When calling
        # # `DetLocalVisualizer().dataset_meta=xxx`,
        # # it will override the default value.
        # self.dataset_meta = {}

    def parse_results(self, results):
        for idx, item in enumerate(results):
            positions, rotations, batch = item
            for i_item in range(len(positions)):
                item_dict = dict(
                    pred_pos=positions[i_item],
                    pred_rot=rotations[i_item],
                )
                input_dict = dict(
                    input_pos=batch['positions'][i_item],
                    input_rot=batch['rotations'][i_item],
                    foot_contact=batch['foot_contact'][i_item],
                    parents=batch['parents'][i_item],
                    bvh_file=batch['bvh_file'][i_item],
                    frame_idx=batch['frame_idx'][i_item],
                    seq_idx=batch['seq_idx'][i_item],
                )
                item_dict.update(input_dict)
                self.motions.append(item_dict)
        return self.motions

    def fig2img(self, fig) -> np.ndarray:
        # convert matplotlib.figure.Figure to np.ndarray(cv2 format)
        fig.canvas.draw()
        graph_image = np.array(fig.canvas.get_renderer()._renderer)
        graph_image = cv2.cvtColor(graph_image, cv2.COLOR_RGB2BGR)
        return graph_image

    def plot_pose(self, poses, prefix, parents):
        # z, y, x -> x, z, y
        # poses = np.transpose(poses, (2, 0, 1))
        np_imgs = []
        for i_frame, pose in enumerate(poses):
            pose = np.concatenate((poses[0], pose, poses[-1]), axis=0)

            fig = plt.figure(figsize=(16, 16))
            ax = fig.add_subplot(111, projection='3d')

            if parents is None:
                parents = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 11, 14, 15, 16, 11, 18, 19, 20]
            ax.cla()
            num_joint = pose.shape[0] // 3
            for i, p in enumerate(parents):
                if i > 0:
                    ax.plot([pose[i, 0], pose[p, 0]], \
                            [pose[i, 2], pose[p, 2]], \
                            [pose[i, 1], pose[p, 1]], c='r')
                    ax.plot([pose[i + num_joint, 0], pose[p + num_joint, 0]], \
                            [pose[i + num_joint, 2], pose[p + num_joint, 2]], \
                            [pose[i + num_joint, 1], pose[p + num_joint, 1]], c='b')
                    ax.plot([pose[i + num_joint * 2, 0], pose[p + num_joint * 2, 0]], \
                            [pose[i + num_joint * 2, 2], pose[p + num_joint * 2, 2]], \
                            [pose[i + num_joint * 2, 1], pose[p + num_joint * 2, 1]], c='g')
            ax.scatter(pose[:num_joint, 0], pose[:num_joint, 2], pose[:num_joint, 1], c='b')
            ax.scatter(pose[num_joint:num_joint * 2, 0], pose[num_joint:num_joint * 2, 2],
                       pose[num_joint:num_joint * 2, 1], c='b')
            ax.scatter(pose[num_joint * 2:num_joint * 3, 0], pose[num_joint * 2:num_joint * 3, 2],
                       pose[num_joint * 2:num_joint * 3, 1], c='g')
            xmin = np.min(pose[:, 0])
            ymin = np.min(pose[:, 2])
            zmin = np.min(pose[:, 1])
            xmax = np.max(pose[:, 0])
            ymax = np.max(pose[:, 2])
            zmax = np.max(pose[:, 1])
            scale = np.max([xmax - xmin, ymax - ymin, zmax - zmin])
            xmid = (xmax + xmin) // 2
            ymid = (ymax + ymin) // 2
            zmid = (zmax + zmin) // 2
            ax.set_xlim(xmid - scale // 2, xmid + scale // 2)
            ax.set_ylim(ymid - scale // 2, ymid + scale // 2)
            ax.set_zlim(zmid - scale // 2, zmid + scale // 2)

            plt.draw()
            np_imgs.append(self.fig2img(fig))
            # plt.savefig(f"{prefix}_{i_frame:04}.png", dpi=200, bbox_inches='tight')
            plt.close()
        return np_imgs

    def save_gif(self, image_list, filepath, fps=30):
        imageio.mimsave(filepath, image_list, fps=fps)
        # # time.sleep(5)
        # frames = glob.glob(frames_prefix+'*.png')
        # frames.sort()
        # clip = ImageSequenceClip.ImageSequenceClip(frames, fps=fps)
        # clip.write_videofile(filepath)

    def on_test_end(self, results, show=True, num_save=0, **kwargs):
        motions = self.parse_results(results)

        if num_save > 0:
            motions = motions[:num_save]
        for idx, motion in enumerate(motions):
            bvh_file = os.path.splitext(os.path.basename(motion['bvh_file']))[0]
            frame_idx = motion['frame_idx'].item()
            seq_idx = motion['seq_idx'].item()

            pred_pos = motion['pred_pos']
            pred_rot = motion['pred_rot']
            parents = motion['parents']

            g_rot, g_pos = lafan1_utils_torch.fk_torch(pred_rot, pred_pos, parents)

            prefix = f"{self.save_dir}/{bvh_file}_{frame_idx}_{seq_idx}"
            imgs = self.plot_pose(g_pos.cpu().numpy(), prefix, parents.tolist())
            self.save_gif(imgs, f"{prefix}.gif", fps=10)

