# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.two_stage import TwoStageDetector

import cv2
import numpy as np
import os
import torch

# def featuremap_2_heatmap(feature_map):
#     assert isinstance(feature_map, torch.Tensor)
#     feature_map = feature_map.detach()
#     heatmap = feature_map[:, 0, :, :] * 0
#     heatmaps = []
#     for c in range(feature_map.shape[1]):
#         heatmap += feature_map[:, c, :, :]
#     heatmap = heatmap.cpu().numpy()
#     heatmap = np.mean(heatmap, axis=0)
#
#     heatmap = np.maximum(heatmap, 0)
#     heatmap /= np.max(heatmap)
#     heatmaps.append(heatmap)
#
#     return heatmaps
#
# def draw_feature_map(features, name,):
#     save_dir='/ai/zyr/data_test/featuremap/'
#     img_path = '/ai/zyr/data_test/dior_test/16122.jpg'
#     # 读取原始图像
#     img = cv2.imread(img_path)
#
#     i = 0  # 初始化计数器
#
#     for featuremap in features:
#         heatmaps = featuremap_2_heatmap(featuremap)
#         for heatmap in heatmaps:
#             heatmap = np.uint8(255 * heatmap)
#             heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
#             heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
#             superimposed_img = heatmap * 0.5 + img*0.8
#             cv2.imwrite(os.path.join(save_dir, name + str(i) + '.png'), superimposed_img)
#             i += 1


@DETECTORS.register_module()
class TFA(TwoStageDetector):
    """Implementation of `TFA <https://arxiv.org/abs/2003.06957>`_"""
    #
    # def extract_feat(self, img):
    #     """Directly extract features from the backbone+neck."""
    #     x = self.backbone(img)
    #     # 可视化resnet产生的特征
    #     # a1 = draw_feature_map(x, "before")
    #     if self.with_neck:
    #         x = self.neck(x)
    #         # 可视化FPN产生的特征
    #         # a2 = draw_feature_map(x, "fpn")
    #
    #     return x