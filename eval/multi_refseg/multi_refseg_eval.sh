#!/bin/sh

## USAGE

## bash eval/multi_refseg/multi_refseg_eval.sh

MASTER_PORT=24997


CKPT_PATH=MTRAG-Model-Path
RESULT_PATH=./eval_results/multi_refseg/MTRAG-Model-Name
CUDA_DEVICES=localhost:0,1,2,3,4,5,6,7


# RefCOCO
deepspeed  --include "$CUDA_DEVICES" --master_port="$MASTER_PORT" ./eval/multi_refseg/multi_refseg.py --version "$CKPT_PATH" --refer_seg_data "refcoco|val" --results_path "$RESULT_PATH" #--workers 0
deepspeed --include "$CUDA_DEVICES" --master_port="$MASTER_PORT" ./eval/multi_refseg/multi_refseg.py --version "$CKPT_PATH" --refer_seg_data "refcoco|testA" --results_path "$RESULT_PATH"
deepspeed --include "$CUDA_DEVICES" --master_port="$MASTER_PORT" ./eval/multi_refseg/multi_refseg.py --version "$CKPT_PATH" --refer_seg_data "refcoco|testB" --results_path "$RESULT_PATH" 

# RefCOCO+
deepspeed --include "$CUDA_DEVICES" --master_port="$MASTER_PORT" ./eval/multi_refseg/multi_refseg.py --version "$CKPT_PATH" --refer_seg_data "refcoco+|val" --results_path "$RESULT_PATH"
deepspeed --include "$CUDA_DEVICES" --master_port="$MASTER_PORT" ./eval/multi_refseg/multi_refseg.py --version "$CKPT_PATH" --refer_seg_data "refcoco+|testA" --results_path "$RESULT_PATH"
deepspeed --include "$CUDA_DEVICES" --master_port="$MASTER_PORT" ./eval/multi_refseg/multi_refseg.py --version "$CKPT_PATH" --refer_seg_data "refcoco+|testB" --results_path "$RESULT_PATH"

# RefCOCOg
deepspeed --include "$CUDA_DEVICES" --master_port="$MASTER_PORT" ./eval/multi_refseg/multi_refseg.py --version "$CKPT_PATH" --refer_seg_data "refcocog|val" --results_path "$RESULT_PATH"
deepspeed --include "$CUDA_DEVICES" --master_port="$MASTER_PORT" ./eval/multi_refseg/multi_refseg.py --version "$CKPT_PATH" --refer_seg_data "refcocog|test" --results_path "$RESULT_PATH"
