#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export NUM_GUP=1

MODEL_NAME='LOBB_SARNet_predcls'

path="Checkpoints/params/${MODEL_NAME}/"
mkdir -p "$path"

python3 \
  tools/count_relation_model_params.py \
  --config-file "configs/e2e_relation_X_101_32_8_FPN_1x_trans_base.yaml" \
  --output "$path" \
  --used-mode backward \
  --num-used-batches 1 \
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
  filter_method "PPG"
