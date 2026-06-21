#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export NUM_GUP=1

MODEL_NAME='LOBB_SARNet_predcls'

path="Checkpoints/GFLOPs/${MODEL_NAME}/"
mkdir -p "$path"

# FLOPs are counted by tools/count_relation_gflops.py with 1 MAC = 1 FLOP.

python3 \
  tools/count_relation_gflops.py \
  --config-file "configs/e2e_relation_X_101_32_8_FPN_1x_trans_base.yaml" \
  --mm-config "configs/RSOBB/STAR_obb_predcls_sgcls.py" \
  --mm-weight "data/weights/OBB_swin_L_OBD.pth" \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
  MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
  MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS False \
  MODEL.ROI_RELATION_HEAD.PREDICTOR SARNet \
  DTYPE "float32" \
  GLOVE_DIR glove\
  TEST.IMS_PER_BATCH "$NUM_GUP" \
  DATALOADER.NUM_WORKERS 0 \
  OUTPUT_DIR "$path" \
  Type "Large_RS_OBB" \
  Sema_F False \
  filter_method "PPG"