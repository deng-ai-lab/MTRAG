#!/bin/sh

## USAGE

## bash eval/mtr-bench/mtr_bench_infer.sh

## USAGE
export CUDA_VISIBLE_DEVICES=6


CKPT_PATH=MTRAG-Model-Path
CHECKPOINT_FILE=MTRAG-Model-Name
# For Description
JSON_PATH=./eval/mtr-bench/description/questions.json
RESULT_PATH=./eval_results/mtr-bench/${CHECKPOINT_FILE}/mtr_description.json

# # For Relations
# JSON_PATH=./eval/mtr-bench/relations/questions.json
# RESULT_PATH=./eval_results/mtr-bench/${CHECKPOINT_FILE}/mtr_relations.json
# Run Inference
python  eval/mtr-bench/mtr_generate_gpt.py --model "$CKPT_PATH" --json "$JSON_PATH" --result_json "$RESULT_PATH"