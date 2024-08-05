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

TABLE 2: FSOD performance on the novel classes of the DIOR test set under 3, 5, 10-shot settings.
<table style="width:100%;">
  <tr>
    <th rowspan="1">Split</th>
    <th rowspan="2">Source</th>
    <th colspan="4">Split 1</th>
    <th colspan="4">Split 2</th>
    <th colspan="4">Split 3</th>
    <th colspan="4">Split 4</th>
  </tr>
  <tr>
    <td>Method</td>
    <td>3-shot</td>
    <td>5-shot</td>
    <td>10-shot</td>
    <td>mean</td>
    <td>3-shot</td>
    <td>5-shot</td>
    <td>10-shot</td>
    <td>mean</td>
    <td>3-shot</td>
    <td>5-shot</td>
    <td>10-shot</td>
    <td>mean</td>
   <td>3-shot</td>
    <td>5-shot</td>
    <td>10-shot</td>
    <td>mean</td>
  </tr>
  
</table>

TABLE 3: FSOD performance on the novel classes of the NWPU VHR-10.v2 test set under 3, 5, and 10-shot settings.
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
    <td>83</td>
    <td>57</td>
    <td>65</td>
  </tr>
  <tr>
    <td>RCCA-FSD</td>
    <td>IEEE J-STARS2023</td>
    <td>40.85</td>
    <td>66.77</td>
    <td>29.66</td>
    <td>45.76</td>
    <td>46.94</td>
    <td>72.94</td>
    <td>37.5</td>
    <td>52.46</td>
    <td>49.24</td>
    <td>71.8</td>
    <td>42.22</td>
    <td>54.42</td>
  </tr>
  <tr>
    <td>G-FSDet</td>
    <td>ISPRS2023</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>49.05</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>56.1</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>71.82</td>
  </tr>
   <tr>
    <td>ICPE</td>
    <td>AAAI2023</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>6.1</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>9.1</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>12.0</td>
  </tr>
   <tr>
    <td>VFA</td>
    <td>AAAI2023</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>13.14</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>15.08</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>13.89</td>
  </tr>
   <tr>
    <td>MM-RCNN</td>
    <td>IEEE TGRS2022</td>
    <td>23</td>
    <td>81</td>
    <td>24</td>
    <td>43</td>
    <td>57</td>
    <td>89</td>
    <td>21</td>
    <td>56</td>
    <td>63</td>
    <td>90</td>
    <td>51</td>
    <td>68</td>
  </tr>
   <tr>
    <td>FRW</td>
    <td>IEEE TGRS2022</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>15.35</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>16.24</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>24</td>
  </tr>
   <tr>
    <td>P-CNN</td>
    <td>IEEE TGRS2021</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>41.8</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>49.17</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>63.29</td>
  </tr>
   <tr>
    <td>TFA</td>
    <td>ICML2020</td>
    <td>17.4</td>
    <td>6.1</td>
    <td>20.6</td>
    <td>14.71</td>
    <td>28.6</td>
    <td>10.9</td>
    <td>20.1</td>
    <td>19.88</td>
    <td>34.2</td>
    <td>12.6</td>
    <td>21.0</td>
    <td>22.61</td>
  </tr>
   <tr>
    <td>Ours</td>
    <td></td>
    <td>55.3</td>
    <td>80.4</td>
    <td>34.7</td>
    <td>56.81</td>
    <td>72.9</td>
    <td>88.5</td>
    <td>40.1</td>
    <td>67.18</td>
    <td>84.0</td>
    <td>90.0</td>
    <td>46.8</td>
    <td>73.61</td>
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
