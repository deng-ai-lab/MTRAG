#!/bin/sh

## USAGE

## bash eval/gcg/run_gcg.sh

# export CUDA_VISIBLE_DEVICES=4,5,6,7
MASTER_PORT=25001
NUM_GPUS=8  # Adjust it as per the available #GPU

# Positional arguments for the bash scripts

CKPT_PATH=MTRAG-Model-Path
# Path to the GranD-f evaluation dataset images directory
IMAGE_DIR=./data/GranDf_HA_images/val_test
RESULT_PATH=./eval_results/gcg/MTRAG-Model-Name
# Run Inference
torchrun --nnodes=1 --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" eval/gcg/infer_gcg.py --hf_model_path "$CKPT_PATH" --img_dir "$IMAGE_DIR" --output_dir "$RESULT_PATH"

# # Path to the GranD-f evaluation dataset ground-truths directory
GT_DIR=./data/GranDf/annotations/val_test
# Evaluate
python eval/gcg/evaluate.py --prediction_dir_path "$RESULT_PATH" --gt_dir_path "$GT_DIR" --split "val"
python eval/gcg/evaluate.py --prediction_dir_path "$RESULT_PATH" --gt_dir_path "$GT_DIR" --split "test"
