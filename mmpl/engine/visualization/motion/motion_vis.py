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



    #
    # def _draw_instances(self, image: np.ndarray, instances: ['InstanceData'],
    #                     classes: Optional[List[str]],
    #                     palette: Optional[List[tuple]]) -> np.ndarray:
    #     """Draw instances of GT or prediction.
    #
    #     Args:
    #         image (np.ndarray): The image to draw.
    #         instances (:obj:`InstanceData`): Data structure for
    #             instance-level annotations or predictions.
    #         classes (List[str], optional): Category information.
    #         palette (List[tuple], optional): Palette information
    #             corresponding to the category.
    #
    #     Returns:
    #         np.ndarray: the drawn image which channel is RGB.
    #     """
    #     self.set_image(image)
    #
    #     if 'bboxes' in instances:
    #         bboxes = instances.bboxes
    #         labels = instances.labels
    #
    #         max_label = int(max(labels) if len(labels) > 0 else 0)
    #         text_palette = get_palette(self.text_color, max_label + 1)
    #         text_colors = [text_palette[label] for label in labels]
    #
    #         bbox_color = palette if self.bbox_color is None \
    #             else self.bbox_color
    #         bbox_palette = get_palette(bbox_color, max_label + 1)
    #         colors = [bbox_palette[label] for label in labels]
    #         self.draw_bboxes(
    #             bboxes,
    #             edge_colors=colors,
    #             alpha=self.alpha,
    #             line_widths=self.line_width)
    #
    #         positions = bboxes[:, :2] + self.line_width
    #         areas = (bboxes[:, 3] - bboxes[:, 1]) * (
    #             bboxes[:, 2] - bboxes[:, 0])
    #         scales = _get_adaptive_scales(areas)
    #
    #         for i, (pos, label) in enumerate(zip(positions, labels)):
    #             label_text = classes[
    #                 label] if classes is not None else f'class {label}'
    #             if 'scores' in instances:
    #                 score = round(float(instances.scores[i]) * 100, 1)
    #                 label_text += f': {score}'
    #
    #             self.draw_texts(
    #                 label_text,
    #                 pos,
    #                 colors=text_colors[i],
    #                 font_sizes=int(13 * scales[i]),
    #                 bboxes=[{
    #                     'facecolor': 'black',
    #                     'alpha': 0.8,
    #                     'pad': 0.7,
    #                     'edgecolor': 'none'
    #                 }])
    #
    #     if 'masks' in instances:
    #         labels = instances.labels
    #         masks = instances.masks
    #         if isinstance(masks, torch.Tensor):
    #             masks = masks.numpy()
    #         elif isinstance(masks, (PolygonMasks, BitmapMasks)):
    #             masks = masks.to_ndarray()
    #
    #         masks = masks.astype(bool)
    #
    #         max_label = int(max(labels) if len(labels) > 0 else 0)
    #         mask_color = palette if self.mask_color is None \
    #             else self.mask_color
    #         mask_palette = get_palette(mask_color, max_label + 1)
    #         colors = [jitter_color(mask_palette[label]) for label in labels]
    #         text_palette = get_palette(self.text_color, max_label + 1)
    #         text_colors = [text_palette[label] for label in labels]
    #
    #         polygons = []
    #         for i, mask in enumerate(masks):
    #             contours, _ = bitmap_to_polygon(mask)
    #             polygons.extend(contours)
    #         self.draw_polygons(polygons, edge_colors='w', alpha=self.alpha)
    #         self.draw_binary_masks(masks, colors=colors, alphas=self.alpha)
    #
    #         if len(labels) > 0 and \
    #                 ('bboxes' not in instances or
    #                  instances.bboxes.sum() == 0):
    #             # instances.bboxes.sum()==0 represent dummy bboxes.
    #             # A typical example of SOLO does not exist bbox branch.
    #             areas = []
    #             positions = []
    #             for mask in masks:
    #                 _, _, stats, centroids = cv2.connectedComponentsWithStats(
    #                     mask.astype(np.uint8), connectivity=8)
    #                 if stats.shape[0] > 1:
    #                     largest_id = np.argmax(stats[1:, -1]) + 1
    #                     positions.append(centroids[largest_id])
    #                     areas.append(stats[largest_id, -1])
    #             areas = np.stack(areas, axis=0)
    #             scales = _get_adaptive_scales(areas)
    #
    #             for i, (pos, label) in enumerate(zip(positions, labels)):
    #                 label_text = classes[
    #                     label] if classes is not None else f'class {label}'
    #                 if 'scores' in instances:
    #                     score = round(float(instances.scores[i]) * 100, 1)
    #                     label_text += f': {score}'
    #
    #                 self.draw_texts(
    #                     label_text,
    #                     pos,
    #                     colors=text_colors[i],
    #                     font_sizes=int(13 * scales[i]),
    #                     horizontal_alignments='center',
    #                     bboxes=[{
    #                         'facecolor': 'black',
    #                         'alpha': 0.8,
    #                         'pad': 0.7,
    #                         'edgecolor': 'none'
    #                     }])
    #     return self.get_image()
    #
    # def _draw_panoptic_seg(self, image: np.ndarray,
    #                        panoptic_seg: ['PixelData'],
    #                        classes: Optional[List[str]]) -> np.ndarray:
    #     """Draw panoptic seg of GT or prediction.
    #
    #     Args:
    #         image (np.ndarray): The image to draw.
    #         panoptic_seg (:obj:`PixelData`): Data structure for
    #             pixel-level annotations or predictions.
    #         classes (List[str], optional): Category information.
    #
    #     Returns:
    #         np.ndarray: the drawn image which channel is RGB.
    #     """
    #     # TODO: Is there a way to bypass？
    #     num_classes = len(classes)
    #
    #     panoptic_seg = panoptic_seg.sem_seg[0]
    #     ids = np.unique(panoptic_seg)[::-1]
    #     legal_indices = ids != num_classes  # for VOID label
    #     ids = ids[legal_indices]
    #
    #     labels = np.array([id % INSTANCE_OFFSET for id in ids], dtype=np.int64)
    #     segms = (panoptic_seg[None] == ids[:, None, None])
    #
    #     max_label = int(max(labels) if len(labels) > 0 else 0)
    #     mask_palette = get_palette(self.mask_color, max_label + 1)
    #     colors = [mask_palette[label] for label in labels]
    #
    #     self.set_image(image)
    #
    #     # draw segm
    #     polygons = []
    #     for i, mask in enumerate(segms):
    #         contours, _ = bitmap_to_polygon(mask)
    #         polygons.extend(contours)
    #     self.draw_polygons(polygons, edge_colors='w', alpha=self.alpha)
    #     self.draw_binary_masks(segms, colors=colors, alphas=self.alpha)
    #
    #     # draw label
    #     areas = []
    #     positions = []
    #     for mask in segms:
    #         _, _, stats, centroids = cv2.connectedComponentsWithStats(
    #             mask.astype(np.uint8), connectivity=8)
    #         max_id = np.argmax(stats[1:, -1]) + 1
    #         positions.append(centroids[max_id])
    #         areas.append(stats[max_id, -1])
    #     areas = np.stack(areas, axis=0)
    #     scales = _get_adaptive_scales(areas)
    #
    #     text_palette = get_palette(self.text_color, max_label + 1)
    #     text_colors = [text_palette[label] for label in labels]
    #
    #     for i, (pos, label) in enumerate(zip(positions, labels)):
    #         label_text = classes[label]
    #
    #         self.draw_texts(
    #             label_text,
    #             pos,
    #             colors=text_colors[i],
    #             font_sizes=int(13 * scales[i]),
    #             bboxes=[{
    #                 'facecolor': 'black',
    #                 'alpha': 0.8,
    #                 'pad': 0.7,
    #                 'edgecolor': 'none'
    #             }],
    #             horizontal_alignments='center')
    #     return self.get_image()
    #
    # @master_only
    # def add_datasample(
    #         self,
    #         name: str,
    #         image: np.ndarray,
    #         data_sample: Optional['DetDataSample'] = None,
    #         draw_gt: bool = True,
    #         draw_pred: bool = True,
    #         show: bool = False,
    #         wait_time: float = 0,
    #         # TODO: Supported in mmengine's Viusalizer.
    #         out_file: Optional[str] = None,
    #         pred_score_thr: float = 0.3,
    #         step: int = 0) -> None:
    #     """Draw datasample and save to all backends.
    #
    #     - If GT and prediction are plotted at the same time, they are
    #     displayed in a stitched image where the left image is the
    #     ground truth and the right image is the prediction.
    #     - If ``show`` is True, all storage backends are ignored, and
    #     the images will be displayed in a local window.
    #     - If ``out_file`` is specified, the drawn image will be
    #     saved to ``out_file``. t is usually used when the display
    #     is not available.
    #
    #     Args:
    #         name (str): The image identifier.
    #         image (np.ndarray): The image to draw.
    #         data_sample (:obj:`DetDataSample`, optional): A data
    #             sample that contain annotations and predictions.
    #             Defaults to None.
    #         draw_gt (bool): Whether to draw GT DetDataSample. Default to True.
    #         draw_pred (bool): Whether to draw Prediction DetDataSample.
    #             Defaults to True.
    #         show (bool): Whether to display the drawn image. Default to False.
    #         wait_time (float): The interval of show (s). Defaults to 0.
    #         out_file (str): Path to output file. Defaults to None.
    #         pred_score_thr (float): The threshold to visualize the bboxes
    #             and masks. Defaults to 0.3.
    #         step (int): Global step value to record. Defaults to 0.
    #     """
    #     image = image.clip(0, 255).astype(np.uint8)
    #     classes = self.dataset_meta.get('classes', None)
    #     palette = self.dataset_meta.get('palette', None)
    #
    #     gt_img_data = None
    #     pred_img_data = None
    #
    #     if data_sample is not None:
    #         data_sample = data_sample.cpu()
    #
    #     if draw_gt and data_sample is not None:
    #         gt_img_data = image
    #         if 'gt_instances' in data_sample:
    #             gt_img_data = self._draw_instances(image,
    #                                                data_sample.gt_instances,
    #                                                classes, palette)
    #
    #         if 'gt_panoptic_seg' in data_sample:
    #             assert classes is not None, 'class information is ' \
    #                                         'not provided when ' \
    #                                         'visualizing panoptic ' \
    #                                         'segmentation results.'
    #             gt_img_data = self._draw_panoptic_seg(
    #                 gt_img_data, data_sample.gt_panoptic_seg, classes)
    #
    #     if draw_pred and data_sample is not None:
    #         pred_img_data = image
    #         if 'pred_instances' in data_sample:
    #             pred_instances = data_sample.pred_instances
    #             pred_instances = pred_instances[
    #                 pred_instances.scores > pred_score_thr]
    #             pred_img_data = self._draw_instances(image, pred_instances,
    #                                                  classes, palette)
    #         if 'pred_panoptic_seg' in data_sample:
    #             assert classes is not None, 'class information is ' \
    #                                         'not provided when ' \
    #                                         'visualizing panoptic ' \
    #                                         'segmentation results.'
    #             pred_img_data = self._draw_panoptic_seg(
    #                 pred_img_data, data_sample.pred_panoptic_seg.numpy(),
    #                 classes)
    #
    #     if gt_img_data is not None and pred_img_data is not None:
    #         drawn_img = np.concatenate((gt_img_data, pred_img_data), axis=1)
    #     elif gt_img_data is not None:
    #         drawn_img = gt_img_data
    #     elif pred_img_data is not None:
    #         drawn_img = pred_img_data
    #     else:
    #         # Display the original image directly if nothing is drawn.
    #         drawn_img = image
    #
    #     # It is convenient for users to obtain the drawn image.
    #     # For example, the user wants to obtain the drawn image and
    #     # save it as a video during video inference.
    #     self.set_image(drawn_img)
    #
    #     if show:
    #         self.show(drawn_img, win_name=name, wait_time=wait_time)
    #
    #     if out_file is not None:
    #         mmcv.imwrite(drawn_img[..., ::-1], out_file)
    #     else:
    #         self.add_image(name, drawn_img, step)
