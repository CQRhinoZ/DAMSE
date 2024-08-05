## Introduction
Few-shot object detection for remote sensing images with direction-aware and multi-scale enhancement fusion

This is a pytorch implementation of 《GUGEN: Global User Graph Enhanced Network For Next POI Recommendation》(IEEE TMC 2024, under review).

Detail information will be released after publication.

## Installation
This code is based on [MMFewshot](https://github.com/open-mmlab/mmfewshot), you can see the mmfew for more detail about the instructions.


## Two-stage training framework


Following the original implementation, it consists of 3 steps:
- **Step1: Base training**
   - use all the images and annotations of base classes to train a base model.

- **Step2: Reshape the bbox head of base model**:
   - create a new bbox head for all classes fine-tuning (base classes + novel classes) using provided script.
   - the weights of base class in new bbox head directly use the original one as initialization.
   - the weights of novel class in new bbox head use random initialization.

- **Step3: Few shot fine-tuning**:
   - use the base model from step2 as model initialization and further fine tune the bbox head with few shot datasets.


### An example of NWPU VHRv2 split1 3-shot setting with 2 gpus

```bash
# step1: base training for voc split1
bash ./tools/detection/dist_train.sh \
    configs/detection/dms-fsod/nwou/split1/my_r101_fpn_nwpu-split1_base-training.py 2

# step2: reshape the bbox head of base model for few shot fine-tuning
python -m tools.detection.misc.initialize_bbox_head \
    --src1 work_dirs/my_r101_fpn_nwpu-split1_base-training/latest.pth \
    --method randinit \
    --save-dir work_dirs/my_r101_fpn_nwpu-split1_base-training

# step3(Model DMS): few shot fine-tuning
bash ./tools/detection/dist_train.sh \
    configs/detection/dms-fsod/dior/split1/dms_r101_fpn_nwpu-split1_3shot-fine-tuning.py 2
```
### Visualization
<p align="center">
  <img src="https://github.com/CQRhinoZ/DAMSE/blob/main/20873 (2).jpg">
</p>
<p align="center">
  <img src="https://github.com/CQRhinoZ/DAMSE/blob/main/11739 (2).jpg">
</p>
<p align="center">
  <img src="https://github.com/CQRhinoZ/DAMSE/blob/main/20432 (2).jpg">
</p>


### Citation
```bash
Feel free to contact us:

Xu ZHANG, Ph.D, Professor

Chongqing University of Posts and Telecommunications

Email: zhangx@cqupt.edu.cn

Website: https://faculty.cqupt.edu.cn/zhangx/zh_CN/index.htm
