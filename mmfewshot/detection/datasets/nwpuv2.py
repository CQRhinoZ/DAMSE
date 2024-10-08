# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import xml.etree.ElementTree as ET
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Union

import mmcv
import numpy as np
from mmcv.utils import print_log
from mmdet.core import eval_recalls
from mmdet.datasets.builder import DATASETS

from mmfewshot.detection.core import eval_map
from .base import BaseFewShotDataset

# pre-defined classes split for few shot setting
NWPUV2_SPLIT = dict(
    ALL_CLASSES_SPLIT1=('airplane', 'baseball', 'basketball','bridge', 'groundtrackfield',
            'harbor', 'ship', 'storagetank', 'tenniscourt', 'vehicle'),
    NOVEL_CLASSES_SPLIT1=('airplane', 'baseball', 'tenniscourt'),
    BASE_CLASSES_SPLIT1=('basketball','bridge', 'groundtrackfield',
            'harbor', 'ship', 'storagetank',  'vehicle'),

    ALL_CLASSES_SPLIT2=('airplane', 'baseball', 'basketball','bridge', 'groundtrackfield',
            'harbor', 'ship', 'storagetank', 'tenniscourt', 'vehicle'),
    NOVEL_CLASSES_SPLIT2=('basketball', 'groundtrackfield', 'vehicle'),
    BASE_CLASSES_SPLIT2=('airplane', 'baseball','bridge',
            'harbor', 'ship', 'storagetank', 'tenniscourt', ),
                )


@DATASETS.register_module()
class FewShotNWPUV2Dataset(BaseFewShotDataset):
    def __init__(self,
                 classes: Optional[Union[str, Sequence[str]]] = None,
                 num_novel_shots: Optional[int] = None,
                 num_base_shots: Optional[int] = None,
                 ann_shot_filter: Optional[Dict] = None,
                 use_difficult: bool = False,
                 min_bbox_area: Optional[Union[int, float]] = None,
                 dataset_name: Optional[str] = None,
                 test_mode: bool = False,
                 coordinate_offset: List[int] = [-1, -1, 0, 0],
                 **kwargs):
        if dataset_name is None:
            self.dataset_name = 'Test dataset' \
                if test_mode else 'Train dataset'
        else:
            self.dataset_name = dataset_name
        self.SPLIT = NWPUV2_SPLIT

        # the split_id would be set value in `self.get_classes`
        self.split_id = None

        assert classes is not None, f'{self.dataset_name}: classes in ' \
                                    f'`FewShotNWPUV2Dataset` can not be None.'

        self.num_novel_shots = num_novel_shots
        self.num_base_shots = num_base_shots
        self.min_bbox_area = min_bbox_area
        self.CLASSES = self.get_classes(classes)
        # `ann_shot_filter` will be used to filter out excess annotations
        # for few shot setting. It can be configured manually or generated
        # by the `num_novel_shots` and `num_base_shots`
        if ann_shot_filter is None:
            # configure ann_shot_filter by num_novel_shots and num_base_shots
            if num_novel_shots is not None or num_base_shots is not None:
                ann_shot_filter = self._create_ann_shot_filter()
        else:
            assert num_novel_shots is None and num_base_shots is None, \
                f'{self.dataset_name}: can not config ann_shot_filter and ' \
                f'num_novel_shots/num_base_shots at the same time.'
        self.coordinate_offset = coordinate_offset
        self.use_difficult = use_difficult
        super().__init__(
            classes=None,
            ann_shot_filter=ann_shot_filter,
            dataset_name=dataset_name,
            test_mode=test_mode,
            **kwargs)

    def get_classes(self, classes: Union[str, Sequence[str]]):

        # configure few shot classes setting
        if isinstance(classes, str):
            assert classes in self.SPLIT.keys(
            ), f'{self.dataset_name}: not a pre-defined classes or ' \
               f'split in NWPUV2_SPLIT'
            class_names = self.SPLIT[classes]
            if 'BASE_CLASSES' in classes:
                assert self.num_novel_shots is None, \
                    f'{self.dataset_name}: BASE_CLASSES do not have ' \
                    f'novel instances.'
            elif 'NOVEL_CLASSES' in classes:
                assert self.num_base_shots is None, \
                    f'{self.dataset_name}: NOVEL_CLASSES do not have ' \
                    f'base instances.'
            self.split_id = int(classes[-1])
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')
        return class_names

    def _create_ann_shot_filter(self):
        """Generate `ann_shot_filter` for novel and base classes.

        Returns:
            dict[str, int]: The number of shots to keep for each class.
        """
        ann_shot_filter = {}
        if self.num_novel_shots is not None:
            for class_name in self.SPLIT[
                    f'NOVEL_CLASSES_SPLIT{self.split_id}']:
                ann_shot_filter[class_name] = self.num_novel_shots
        if self.num_base_shots is not None:
            for class_name in self.SPLIT[f'BASE_CLASSES_SPLIT{self.split_id}']:
                ann_shot_filter[class_name] = self.num_base_shots
        return ann_shot_filter

    def load_annotations(self, ann_cfg: List[Dict]):
        """Support to load annotation from two type of ann_cfg.

        Args:
            ann_cfg (list[dict]): Support two type of config.

            - loading annotation from common ann_file of dataset
              with or without specific classes.
              example:dict(type='ann_file', ann_file='path/to/ann_file',
              ann_classes=['dog', 'cat'])
            - loading annotation from a json file saved by dataset.
              example:dict(type='saved_dataset', ann_file='path/to/ann_file')

        Returns:
            list[dict]: Annotation information.
        """
        self.cat2label = {cat: i for i, cat in enumerate(self.CLASSES)}
        data_infos = []
        for ann_cfg_ in ann_cfg:
            if ann_cfg_['type'] == 'saved_dataset':
                data_infos += self.load_annotations_saved(ann_cfg_['ann_file'])
            elif ann_cfg_['type'] == 'ann_file':
                # load annotation from specific classes
                ann_classes = ann_cfg_.get('ann_classes', None)
                if ann_classes is not None:
                    for c in ann_classes:
                        assert c in self.CLASSES, \
                            f'{self.dataset_name}: ann_classes must in ' \
                            f'dataset classes.'
                else:
                    ann_classes = self.CLASSES
                data_infos += self.load_annotations_xml(
                    ann_cfg_['ann_file'], ann_classes)
            else:
                raise ValueError(
                    f'{self.dataset_name}: not support '
                    f'annotation type {ann_cfg_["type"]} in ann_cfg.')
        return data_infos

    def load_annotations_xml(
            self,
            ann_file: str,
            classes: Optional[List[str]] = None):
        """Load annotation from XML style ann_file.

        It supports using image id or image path as image names
        to load the annotation file.

        Args:
            ann_file (str): Path of annotation file.
            classes (list[str] | None): Specific classes to load form xml file.
                If set to None, it will use classes of whole dataset.
                Default: None.

        Returns:
            list[dict]: Annotation info from XML file.
        """
        data_infos = []
        img_names = mmcv.list_from_file(ann_file)
        for idx_, img_name in enumerate(img_names):
            # dataset_year = 'NWPU2017'
            dataset_year = ''
            # ann file in image path format
            img_id = img_name
            # filename = f'NWPU2017/JPEGImages/{img_name}.jpg'
            filename = f'JPEGImages/{img_name}.jpg'
            filename = filename.replace(" ", "")
            Annotation_dir = 'Annotations'
            JPEGImages_dir = 'JPEGImages'


            xml_path = osp.join(self.img_prefix, dataset_year, Annotation_dir,
                                f'{img_id}.xml')
            xml_path = xml_path.replace(" ", "")
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find('size')
            if size is not None:
                width = int(size.find('width').text)
                height = int(size.find('height').text)
            else:
                img_path = osp.join(self.img_prefix, dataset_year,
                                    JPEGImages_dir, '{}.jpg'.format(img_id))
                img_path = img_path.replace(" ", "")
                img = mmcv.imread(img_path)
                width, height = img.size
            ann_info = self._get_xml_ann_info(dataset_year, img_id, classes, Annotation_dir)
            data_infos.append(
                dict(
                    id=img_id,
                    filename=filename,
                    width=width,
                    height=height,
                    ann=ann_info))

        return data_infos

    def _get_xml_ann_info(self,
                          dataset_year: str,
                          img_id: str,
                          classes: Optional[List[str]] = None,
                          Annotation_dir: str = 'Annotations'):
        """Get annotation from XML file by img_id.

        Args:
            dataset_year (str): Year of NWPUV2 dataset. Options are
                'NWPUV22007'
            img_id (str): Id of image.
            classes (list[str] | None): Specific classes to load form
                xml file. If set to None, it will use classes of whole
                dataset. Default: None.

        Returns:
            dict: Annotation info of specified id with specified class.
        """

        if classes is None or classes[0] == 'mosica':
            # print('classes', classes)
            # assert 2==1
            classes = self.CLASSES
        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []

        xml_path = osp.join(self.img_prefix, dataset_year, Annotation_dir,
                            f'{img_id}.xml')
        xml_path = xml_path.replace(" ", "")
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in classes:
                continue
            label = self.cat2label[name]
            if self.use_difficult:
                difficult = 0
            else:
                difficult = obj.find('difficult')
                difficult = 0 if difficult is None else int(difficult.text)
            bnd_box = obj.find('bndbox')

            # It should be noted that in the original mmdet implementation,
            # the four coordinates are reduced by 1 when the annotation
            # is parsed. Here we following detectron2, only xmin and ymin
            # will be reduced by 1 during training. The groundtruth used for
            # evaluation or testing keep consistent with original xml
            # annotation file and the xmin and ymin of prediction results
            # will add 1 for inverse of data loading logic.
            bbox = [
                int(float(bnd_box.find('xmin').text)),
                int(float(bnd_box.find('ymin').text)),
                int(float(bnd_box.find('xmax').text)),
                int(float(bnd_box.find('ymax').text))
            ]
            if not self.test_mode:
                bbox = [
                    i + offset
                    for i, offset in zip(bbox, self.coordinate_offset)
                ]
            ignore = False
            if difficult or ignore:
                bboxes_ignore.append(bbox)
                labels_ignore.append(label)
            else:
                bboxes.append(bbox)
                labels.append(label)
        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0, ))
        else:
            bboxes = np.array(bboxes, ndmin=2)
            labels = np.array(labels)
        if not bboxes_ignore:
            bboxes_ignore = np.zeros((0, 4))
            labels_ignore = np.zeros((0, ))
        else:
            bboxes_ignore = np.array(bboxes_ignore, ndmin=2)
            labels_ignore = np.array(labels_ignore)
        ann_info = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64))
        return ann_info

    def _filter_imgs(self,
                     min_size: int = 32,
                     min_bbox_area: Optional[int] = None):
        """Filter images not meet the demand.

        Args:
            min_size (int): Filter images with length or width
                smaller than `min_size`. Default: 32.
            min_bbox_area (int | None): Filter images with bbox whose
                area smaller `min_bbox_area`. If set to None, skip
                this filter. Default: None.

        Returns:
            list[int]: valid indices of `data_infos`.
        """
        valid_inds = []
        if min_bbox_area is None:
            min_bbox_area = self.min_bbox_area
        for i, img_info in enumerate(self.data_infos):
            # filter empty image
            if self.filter_empty_gt:
                cat_ids = img_info['ann']['labels'].astype(np.int64).tolist()
                if len(cat_ids) == 0:
                    continue
            # filter images smaller than `min_size`
            if min(img_info['width'], img_info['height']) < min_size:
                continue
            # filter image with bbox smaller than min_bbox_area
            # it is usually used in Attention RPN
            if min_bbox_area is not None:
                skip_flag = False
                for bbox in img_info['ann']['bboxes']:
                    bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    if bbox_area < min_bbox_area:
                        skip_flag = True
                if skip_flag:
                    continue
            valid_inds.append(i)
        return valid_inds

    def evaluate(self,
                 results: List[Sequence],
                 metric: Union[str, List[str]] = 'mAP',
                 logger: Optional[object] = None,
                 proposal_nums: Sequence[int] = (100, 300, 1000),
                 iou_thr: Optional[Union[float, Sequence[float]]] = 0.5,
                 class_splits: Optional[List[str]] = None) :
        """Evaluation in VOC protocol and summary results of different splits
        of classes.

        Args:
            results (list[list | tuple]): Predicftions of the model.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'mAP', 'recall'. Default: mAP.
            logger (logging.Logger | None): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. Default: 0.5.
            class_splits: (list[str] | None): Calculate metric of classes
                split  defined in NWPUV2_SPLIT. For example:
                ['BASE_CLASSES_SPLIT1', 'NOVEL_CLASSES_SPLIT1'].
                Default: None.

        Returns:
            dict[str, float]: AP/recall metrics.
        """
        # It should be noted that in the original mmdet implementation,
        # the four coordinates are reduced by 1 when the annotation
        # is parsed. Here we following detectron2, only xmin and ymin
        # will be reduced by 1 during training. The groundtruth used for
        # evaluation or testing keep consistent with original xml
        # annotation file and the xmin and ymin of prediction results
        # will add 1 for inverse of data loading logic.
        for i in range(len(results)):
            for j in range(len(results[i])):
                for k in range(4):
                    results[i][j][:, k] -= self.coordinate_offset[k]

        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'recall']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        if class_splits is not None:
            for k in class_splits:
                assert k in self.SPLIT.keys(), 'undefiend classes split.'
            class_splits = {k: self.SPLIT[k] for k in class_splits}
            class_splits_mean_aps = {k: [] for k in class_splits.keys()}

        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = OrderedDict()
        iou_thrs = [iou_thr] if isinstance(iou_thr, float) else iou_thr
        if metric == 'mAP':
            assert isinstance(iou_thrs, list)
            mean_aps = []
            for iou_thr in iou_thrs:
                print_log(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
                mean_ap, ap_results = eval_map(
                    results,
                    annotations,
                    classes=self.CLASSES,
                    scale_ranges=None,
                    iou_thr=iou_thr,
                    dataset='voc07',
                    logger=logger,
                    use_legacy_coordinate=True)
                mean_aps.append(mean_ap)
                eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)

                # calculate evaluate results of different class splits
                if class_splits is not None:
                    for k in class_splits.keys():
                        aps = [
                            cls_results['ap']
                            for i, cls_results in enumerate(ap_results)
                            if self.CLASSES[i] in class_splits[k]
                        ]
                        class_splits_mean_ap = np.array(aps).mean().item()
                        class_splits_mean_aps[k].append(class_splits_mean_ap)
                        eval_results[
                            f'{k}: AP{int(iou_thr * 100):02d}'] = round(
                                class_splits_mean_ap, 3)

            eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
            if class_splits is not None:
                for k in class_splits.keys():
                    mAP = sum(class_splits_mean_aps[k]) / len(
                        class_splits_mean_aps[k])
                    print_log(f'{k} mAP: {mAP}', logger=logger)
        elif metric == 'recall':
            gt_bboxes = [ann['bboxes'] for ann in annotations]
            recalls = eval_recalls(
                annotations, results, proposal_nums, iou_thr, logger=logger)
            for i, num in enumerate(proposal_nums):
                for j, iou in enumerate(iou_thr):
                    eval_results[f'recall@{num}@{iou}'] = recalls[i, j]
            if recalls.shape[1] > 1:
                ar = recalls.mean(axis=1)
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
        return eval_results


@DATASETS.register_module()
class FewShotNWPUV2CopyDataset(FewShotNWPUV2Dataset):
    """Copy other NWPUV2 few shot datasets' `data_infos` directly.

    This dataset is mainly used for model initialization in some meta-learning
    detectors. In their cases, the support data are randomly sampled
    during training phase and they also need to be used in model
    initialization before evaluation. To copy the random sampling results,
    this dataset supports to load `data_infos` of other datasets via `ann_cfg`

    Args:
        ann_cfg (list[dict] | dict): contain `data_infos` from other
            dataset. Example: [dict(data_infos=FewShotNWPUV2Dataset.data_infos)]
    """

    def __init__(self, ann_cfg: Union[List[Dict], Dict], **kwargs):
        super().__init__(ann_cfg=ann_cfg, **kwargs)

    def ann_cfg_parser(self, ann_cfg: Union[List[Dict], Dict]):
        """Parse annotation config from a copy of other dataset's `data_infos`.

        Args:
            ann_cfg (list[dict] | dict): contain `data_infos` from other
                dataset. Example:
                [dict(data_infos=FewShotNWPUV2Dataset.data_infos)]

        Returns:
            list[dict]: Annotation information.
        """
        data_infos = []
        if isinstance(ann_cfg, dict):
            assert ann_cfg.get('data_infos', None) is not None, \
                f'{self.dataset_name}: ann_cfg of ' \
                f'FewShotNWPUV2CopyDataset require data_infos.'
            # directly copy data_info
            data_infos = ann_cfg['data_infos']
        elif isinstance(ann_cfg, list):
            for ann_cfg_ in ann_cfg:
                assert ann_cfg_.get('data_infos', None) is not None, \
                    f'{self.dataset_name}: ann_cfg of ' \
                    f'FewShotNWPUV2CopyDataset require data_infos.'
                # directly copy data_info
                data_infos += ann_cfg_['data_infos']
        return data_infos




@DATASETS.register_module()
class FewShotNWPUV2DefaultDataset(FewShotNWPUV2Dataset):
    """Dataset with some pre-defined NWPUV2 annotation paths.

    :obj:`FewShotNWPUV2DefaultDataset` provides pre-defined annotation files
    to ensure the reproducibility. The pre-defined annotation files provide
    fixed training data to avoid random sampling. The usage of `ann_cfg' is
    different from :obj:`FewShotNWPUV2Dataset`. The `ann_cfg' should contain
    two filed: `method` and `setting`.

    Args:
        ann_cfg (list[dict]): Each dict should contain
            `method` and `setting` to get corresponding
            annotation from `DEFAULT_ANN_CONFIG`.
            For example: [dict(method='TFA', setting='SPILT1_1shot')].
    """

    nwpuv2_benchmark = {
        f'SPLIT{split}_{shot}SHOT': [
            dict(
                type='ann_file',
                ann_file=f'/ai/zyr/NWPU/few_shot_ann/benchmark_{shot}shot/'
                f'box_{shot}shot_{class_name}_train.txt',
                ann_classes=[class_name])
            for class_name in NWPUV2_SPLIT[f'ALL_CLASSES_SPLIT{split}']
        ]
        for shot in [3, 5, 10, 20] for split in [1,2]
    }

    # pre-defined annotation config for model reproducibility
    DEFAULT_ANN_CONFIG = dict(
        TFA=nwpuv2_benchmark,
        FSCE=nwpuv2_benchmark,
        Attention_RPN=nwpuv2_benchmark,
        MPSR=nwpuv2_benchmark,
        MetaRCNN=nwpuv2_benchmark,
        FSDetView=nwpuv2_benchmark)

    def __init__(self, ann_cfg: List[Dict], **kwargs):
        super().__init__(ann_cfg=ann_cfg, **kwargs)

    def ann_cfg_parser(self, ann_cfg: List[Dict]):
        """Parse pre-defined annotation config to annotation information.

        Args:
            ann_cfg (list[dict]): contain method and setting
                of pre-defined annotation config. Example:
                [dict(method='TFA', setting='SPILT1_1shot')]

        Returns:
            list[dict]: Annotation information.
        """
        new_ann_cfg = []
        for ann_cfg_ in ann_cfg:
            assert isinstance(ann_cfg_, dict), \
                f'{self.dataset_name}: ann_cfg should be list of dict.'
            method = ann_cfg_['method']
            setting = ann_cfg_['setting']
            default_ann_cfg = self.DEFAULT_ANN_CONFIG[method][setting]
            ann_root = ann_cfg_.get('ann_root', None)
            if ann_root is not None:
                for i in range(len(default_ann_cfg)):
                    default_ann_cfg[i]['ann_file'] = osp.join(
                        ann_root, default_ann_cfg[i]['ann_file'])
            new_ann_cfg += default_ann_cfg
        return super(FewShotNWPUV2Dataset, self).ann_cfg_parser(new_ann_cfg)