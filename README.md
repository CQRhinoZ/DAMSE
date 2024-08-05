## Introduction

This is a pytorch implementation of 《Few-shot object detection for remote sensing images with direction-aware and multi-scale enhancement fusion》(SIGSpatial 2024, under review).

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

## Performance
TABLE 3: FSOD performance on the novel classes of the NWPU VHR-10.v2 test set under 3, 5, and 10-shot settings. Red represents
the highest accuracy, while blue represents the second highest accuracy.
<table style="width:100%;">
  <tr>
    <th rowspan="2">Method</th>
    <th rowspan="2">Source</th>
    <th colspan="4">3-shot</th>
    <th colspan="4">5-shot</th>
    <th colspan="4">10-shot</th>
  </tr>
  <tr>
    <td>AP</td>
    <td>BD</td>
    <td>TC</td>
    <td>mean</td>
    <td>AP</td>
    <td>BD</td>
    <td>TC</td>
    <td>mean</td>
    <td>AP</td>
    <td>BD</td>
    <td>TC</td>
    <td>mean</td>
  </tr>
  <tr>
    <td>FAM&SAM</td>
    <td>IEEE J-STARS2024</td>
    <td>19</td>
    <td>66</td>
    <td>22</td>
    <td>36</td>
    <td>43</td>
    <td>77</td>
    <td>37</td>
    <td>52</td>
    <td>56</td>
    <td><font color="blue">83</font></td>
    <td><font color="red">57</font></td>
    <td>65</td>
  </tr>
</table>

### Visualization
<p align="center">
  <img src="https://github.com/CQRhinoZ/DAMSE/blob/main/Vis_20873.jpg">
</p>
<p align="center">
  <img src="https://github.com/CQRhinoZ/DAMSE/blob/main/Vis_11739.jpg">
</p>
<p align="center">
  <img src="https://github.com/CQRhinoZ/DAMSE/blob/main/Vis_20432.jpg">
</p>


### Citation
```bash
Feel free to contact us:

Xu ZHANG, Ph.D, Professor

Chongqing University of Posts and Telecommunications

Email: zhangx@cqupt.edu.cn

Website: https://faculty.cqupt.edu.cn/zhangx/zh_CN/index.htm
