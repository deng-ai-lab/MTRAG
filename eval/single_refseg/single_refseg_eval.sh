#!/bin/sh

## USAGE

## bash eval/single_refseg/single_refseg_eval.sh <path to the HF checkpoints path> <path to the directory to save the evaluation results>

MASTER_PORT=24997

CKPT_PATH=MTRAG-Model-Path
CKPT_NAME=MTRAG-Model-Name
RESULT_PATH=./eval_results/single_refseg/$CKPT_NAME
Adjust the environment variable if you have multiple gpus available, e.g. CUDA_VISIBLE_DEVICES=0,1,2,3 if you have 4 GPUs available
CUDA_DEVICES="localhost:0,1,2,3"
# RefCOCO
deepspeed --include "$CUDA_DEVICES" --master_port="$MASTER_PORT" ./eval/single_refseg/infer_and_evaluate.py --version "$CKPT_PATH" --refer_seg_data "refcoco|val" --results_path "$RESULT_PATH"
deepspeed --include "$CUDA_DEVICES" --master_port="$MASTER_PORT" ./eval/single_refseg/infer_and_evaluate.py --version "$CKPT_PATH" --refer_seg_data "refcoco|testA" --results_path "$RESULT_PATH"
deepspeed --include "$CUDA_DEVICES" --master_port="$MASTER_PORT" ./eval/single_refseg/infer_and_evaluate.py --version "$CKPT_PATH" --refer_seg_data "refcoco|testB" --results_path "$RESULT_PATH" 

# RefCOCO+
deepspeed --include "$CUDA_DEVICES" --master_port="$MASTER_PORT" ./eval/single_refseg/infer_and_evaluate.py --version "$CKPT_PATH" --refer_seg_data "refcoco+|val" --results_path "$RESULT_PATH"
deepspeed --include "$CUDA_DEVICES" --master_port="$MASTER_PORT" ./eval/single_refseg/infer_and_evaluate.py --version "$CKPT_PATH" --refer_seg_data "refcoco+|testA" --results_path "$RESULT_PATH"
deepspeed --include "$CUDA_DEVICES" --master_port="$MASTER_PORT" ./eval/single_refseg/infer_and_evaluate.py --version "$CKPT_PATH" --refer_seg_data "refcoco+|testB" --results_path "$RESULT_PATH"

# RefCOCOg
deepspeed --include "$CUDA_DEVICES" --master_port="$MASTER_PORT" ./eval/single_refseg/infer_and_evaluate.py --version "$CKPT_PATH" --refer_seg_data "refcocog|val" --results_path "$RESULT_PATH"
deepspeed --include "$CUDA_DEVICES" --master_port="$MASTER_PORT" ./eval/single_refseg/infer_and_evaluate.py --version "$CKPT_PATH" --refer_seg_data "refcocog|test" --results_path "$RESULT_PATH"
