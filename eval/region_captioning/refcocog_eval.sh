#!/bin/sh

## USAGE

## bash eval/region_captioning/refcocog_eval.sh <path to the HF checkpoints path> <path to the directory to save the evaluation results>

## USAGE

export CUDA_VISIBLE_DEVICES=4,5,6,7
# export PYTHONPATH="./:$PYTHONPATH"
MASTER_PORT=24998
NUM_GPUS=4  # Adjust it as per the available #GPU
# Positional arguments for the bash scripts
CKPT_PATH=MTRAG-Model-Path
RESULT_PATH=./eval_results/region_captioning/MTRAG-Model-Name

# Adjust if needed
ANNOTATION_FILE=./data/region_cap/finetune_refcocog_val_with_mask.json
IMAGE_DIR=./data/coco_2014/train2014
DATASET=refcocog

# Run Inference
torchrun --nnodes=1 --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" eval/region_captioning/infer_refcocog.py --hf_model_path "$CKPT_PATH" --lora_model_path "$LoRA_PATH" --annotation_file "$ANNOTATION_FILE" --image_dir "$IMAGE_DIR" --dataset "$DATASET" --results_dir "$RESULT_PATH"


# Evaluate
python eval/region_captioning/evaluate.py --annotation_file "$ANNOTATION_FILE" --results_dir "$RESULT_PATH"
