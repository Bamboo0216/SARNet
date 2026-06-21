# SARNet

Implementation of SARNet for remote sensing scene graph generation based on SGG-ToolKit.

## Introduction

Spatially Anisotropic Reasoning Network (SARNet) is designed for remote sensing scene graph generation, where large-area images contain multiple functional subscenes and predicate semantics are strongly shaped by relative spatial configurations. It introduces Anisotropic Ellipse Influence Propagation (AEIP) to model object-centered spatial influence regions from oriented bounding boxes and suppress noisy long-range interactions, together with Relative Spatial Configuration Attention (RSCA) to enhance geometry-discriminative relation features for predicate prediction.

![motivation.png](demo/motivation.png)

## Installation

Check [INSTALL.md](INSTALL.md) for environment setup instructions.

## Dataset

Check [DATASET.md](DATASET.md) for STAR dataset preprocessing instructions.

## Pretrained Models

Pretrained SARNet checkpoints for evaluation on the STAR dataset are available at [Bamboo0216/SARNet-SGG-STAR](https://huggingface.co/Bamboo0216/SARNet-SGG-STAR). Download the desired checkpoint, place it under `data/weights/`.

## Required Files / Paths

Before running SARNet, prepare the following files and update the paths in the scripts or configs for your environment:

- STAR images and annotation files, following the structure described in [DATASET.md](DATASET.md).
- OBB or HBB detector checkpoint for training, such as `OBB_swin_L_OBD.pth` or `HBB_swin_L_OBD.pth`.
- For evaluation only, `--mm_weight` can point to the trained checkpoint that you want to test or validate.

## Training script

Before running it, check and update these fields in the script:

```bash
CUDA_VISIBLE_DEVICES=1
path="/data/${MODEL_NAME}/"
--config-file "SGG-ToolKit/configs/e2e_relation_X_101_32_8_FPN_1x_trans_base.yaml"
--mm_config "SGG-ToolKit/configs/RSOBB/STAR_obb_predcls_sgcls.py"
--mm_weight "data/OBB_swin_L_OBD.pth"
GLOVE_DIR data/STAR_SGG/glove
OUTPUT_DIR "$path"
MODEL.ROI_RELATION_HEAD.PREDICTOR SARNet
Type "Large_RS_OBB"
filter_method "PPG"
```

PredCls, SGCls and SGDet scripts can be found in `Scripts/`. Before using them, verify the local paths, `--mm_config`, `--mm_weight`, `Type`, and `MODEL.ROI_RELATION_HEAD.PREDICTOR SARNet`.

## Configuration Notes

- `Type` selects the remote sensing detection setting, such as `Large_RS_OBB` or `Large_RS_HBB`.
- `filter_method "PPG"` uses PPG candidate filtering; `best_model_OBB.pth` and `best_model_HBB.pth` are the corresponding PPG weights.
- `Sema_F` applies a semantic filter during testing.

## Object Detection

If you only need OBB/HBB object detection, refer to the corresponding STAR object detection projects:

- [STAR-MMRotate](https://github.com/yangxue0827/STAR-MMRotate)
- [STAR-MMDetection](https://github.com/Zhuzi24/STAR-MMDetection)
