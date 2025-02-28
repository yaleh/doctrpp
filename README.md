# DocTrPP: DocTr++ in PaddlePaddle

## Introduction

This is a PaddlePaddle implementation of DocTr++. The original paper is [DocTr++: Deep Unrestricted Document Image Rectification](https://arxiv.org/abs/2304.08796). The original code is [here](https://github.com/fh2019ustc/DocTr-Plus).

![demo](https://github.com/GreatV/DocTrPP/assets/17264618/4e491512-bfc4-4e69-a833-fd1c6e17158c)

## Requirements

You need to install the latest version of PaddlePaddle, which is done through this [link](https://www.paddlepaddle.org.cn/).

## Training

1. Data Preparation

To prepare datasets, refer to [doc3D](https://github.com/cvlab-stonybrook/doc3D-dataset).

2. Training

```shell
sh train.sh
```

or

```shell
export OPENCV_IO_ENABLE_OPENEXR=1
export CUDA_VISIBLE_DEVICES=0

python train.py --img-size 288 \
    --name "DocTr++" \
    --batch-size 12 \
    --lr 2.5e-5 \
    --exist-ok \
    --use-vdl
```

3. Load Trained Model and Continue Training

```shell
export OPENCV_IO_ENABLE_OPENEXR=1
export CUDA_VISIBLE_DEVICES=0

python train.py --img-size 288 \
    --name "DocTr++" \
    --batch-size 12 \
    --lr 2.5e-5 \
    --resume "runs/train/DocTr++/weights/last.ckpt" \
    --exist-ok \
    --use-vdl
```

## Test and Inference

Test the dewarp result on a single image:

```shell
python predict.py -i "crop/12_2 copy.png" -m runs/train/DocTr++/weights/best.ckpt -o 12.2.png
```
![document image rectification](https://raw.githubusercontent.com/greatv/DocTrPP/main/doc/imgs/document_image_rectification.jpg)

## Export to onnx

```
pip install paddle2onnx

python export.py -m ./best.ckpt --format onnx
```

## Model Download

The trained model can be downloaded from [here](https://github.com/GreatV/DocTrPP/releases/download/v0.0.2/best.ckpt).

## 打包说明

本项目已通过Poetry打包为pip包。您可以通过以下方式安装和使用：

### 安装

```bash
# 从本地安装
pip install dist/doctrpp-0.1.0-py3-none-any.whl

# 也可以直接从源码安装
pip install .
```

### 使用方法

#### 命令行工具

安装后，可以使用命令行工具`doctrpp`：

```bash
doctrpp -i <输入图像路径> -m <模型路径> -o <输出图像路径>
```

#### Python API

在Python代码中导入和使用：

```python
# 导入预测函数
from doctrpp import predict

# 处理图像
predict(
    image_path="input.jpg",
    model_path="model.ckpt",
    output_path="output.jpg"
)
```

或者自定义处理过程：

```python
import cv2
import paddle
from doctrpp import GeoTr, to_tensor, to_image

# 加载模型
checkpoint = paddle.load("model.ckpt")
state_dict = checkpoint["model"]
model = GeoTr()
model.set_state_dict(state_dict)
model.eval()

# 加载图像
img_org = cv2.imread("input.jpg")
img = cv2.resize(img_org, (288, 288))
x = to_tensor(img)
y = to_tensor(img_org)

# 处理图像
bm = model(x)
bm = paddle.nn.functional.interpolate(
    bm, y.shape[2:], mode="bilinear", align_corners=False
)
bm_nhwc = bm.transpose([0, 2, 3, 1])
out = paddle.nn.functional.grid_sample(y, (bm_nhwc / 288 - 0.5) * 2)
out_image = to_image(out)

# 保存结果
cv2.imwrite("output.jpg", out_image)
```
