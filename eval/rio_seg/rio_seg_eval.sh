#!/bin/sh

## USAGE

## bash eval/rio_seg/rio_seg_eval.sh <path to the HF checkpoints path> <path to the directory to save the evaluation results>

MASTER_PORT=24997


CKPT_PATH=MTRAG-Model-Path
RESULT_PATH=./eval_results/rio_seg/MTRAG-Model-Name

# deepspeed --include "localhost:0,1,2,3,4,5,6,7" --master_port="$MASTER_PORT" ./eval/rio_seg/rio_seg_eval.py --version "$CKPT_PATH" --refer_seg_data "RIOcomSegm" --results_path "$RESULT_PATH"
deepspeed --include "localhost:0,1,2,3,4,5,6,7" --master_port="$MASTER_PORT" ./eval/rio_seg/rio_seg_eval.py --version "$CKPT_PATH" --refer_seg_data "RIOUncomSegm" --results_path "$RESULT_PATH"

