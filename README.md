# SSD Keras - Object detection
Thanks [rykov8] for basic structure and [wikke] for updating model.
I split from `generate_data` to `model` into several part which notebook name starts with TX_ .
My goal is sharing my knowlege in simple way to everyone. Have fun !
![Imgur](https://i.imgur.com/2BufYZO.png)

## Requirement
- tensorflow >= 1.4.0
- opencv-python
- keras >= 2.0
- scipy
- matplotlib
- google.protobuf

## [*Notice for Dataset*]
you must to create dict classes, check [sample](sample/pascal.pbtxt).

## Dataset
* VOC2007 : ```wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar```
* VOC2012 : ```wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar```

## Pretrained Model Weights
download link is [here](https://drive.google.com/open?id=1wNTwvdSCmVbt_vE-w2Q0xieTs6yUsmrR)

## Description
1. In [T1_GenerateData.ipynb](T1_GenerateData.ipynb) shows you each of steps image process effect and flow of generating dataset.
2. In [T2_Modulized_GenerateData.ipynb](T2_Modulized_GenerateData.ipynb) shows you I modulized generate_data then placed in [utils/generate_data.py](utils/generate_data.py) and [demo part](utils/demo.py).
3. In [T3_AssignBBoxes.ipynb](T3_AssignBBoxes.ipynb) describes the way to **assign bboxes** which is core procedure in **generate_data.py**.
4. In [T4_PriorBox.ipynb](T4_PriorBox.ipynb) describe the layer in SSD model which is core layer for Object-detection.
5. In [T5_SSDModel.ipynb](T5_SSDModel.ipynb) I restructure original model which I think more easily to understand.
6. In [T6_CreatePriorBoxes](T6_CreatePriorBoxes.ipynb) I find this on website and make a explanation.
7. In [T7_LossFunction](T7_LossFunction.ipynb) Note here, I still figure it out.
8. In [T8_DecodePredictValue](T8_DecodePredictValue.ipynb) As the filename, script is quite intuition, very easy to understand.

[rykov8]: <https://github.com/rykov8/ssd_keras>
[wikke]: <https://github.com/wikke/SSD_Keras>
