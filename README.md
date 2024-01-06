#                feature detection and matching algorithm models

## Additional Information
This fork adds a tensorrt conversion script, slight modification to superpoint/superglue models to work with the converted engine models. See `How to Run` section on how to use the script.

## Introduction		

🚀🚀This warehouse mainly uses C++ to compare traditional image feature detection and matching, and deep learning feature detection and matching algorithm models. Deep learning includes superpoint-superglue, and traditional algorithms include AKAZE, SURF, ORB, etc.

1. akaze feature point detection and matching display.

![akaze-image](./image/akaze_example.gif)

2. superpoint-superpoint feature point detection and matching display.

![akaze-video](./image/deep-learning_example.gif)

## Dependencies

All operating environments, please strictly follow the given configuration，the configuration is as follows：

OpenCV >= 3.4

CUDA >=10.2

CUDNN>=8.02

TensorRT>=7.2.3

### Python 3 Dependencies

Install the following packages to use the conversion script:

polygraphy

## How to Run

### Tensorrt Conversion
1. Execute both `convert/superpoint_convert.py` and `convert/superglue_convert.py`.
2. Once converted, the resulting models will be inside the `convert/weights` directory named `superpoint_v1.engine` and `superglue_indoor.engine`. Move these files inside the `engines` folder.

### Compilation and Execution
1. build.

```
cd feature-detection-matching-algorithm/
mkdir build
cd build
cmake ..
make
```

2. run camera.

deep learning algorithms.

```
./IR --deeplearning --camera 0
```

traditional algorithms.

```
./IR --traditional  --camera 0
```

3. run image-pair.

deep learning algorithms.

```
./IR --deeplearning --image-pair xx01.jpg xx02.jpg
```

traditional algorithms.

```
./IR --traditional  --image-pair xx01.jpg xx02.jpg
```

## SuperPoint

Superpoint pretrained models are from [magicleap/SuperGluePretrainedNetwork.](https://github.com/magicleap/SuperGluePretrainedNetwork)

## SuperGlue

SuperGlue pretrained models are from [magicleap/SuperGluePretrainedNetwork.](https://github.com/magicleap/SuperGluePretrainedNetwork)

## Reference

```
@inproceedings{sarlin20superglue,
  author    = {Paul-Edouard Sarlin and
               Daniel DeTone and
               Tomasz Malisiewicz and
               Andrew Rabinovich},
  title     = {{SuperGlue}: Learning Feature Matching with Graph Neural Networks},
  booktitle = {CVPR},
  year      = {2020},
  url       = {https://arxiv.org/abs/1911.11763}
}
```

