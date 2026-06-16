#!/bin/bash

## USAGE

## bash eval/osprey_description/osprey_gpt_eval.sh


export CUDA_VISIBLE_DEVICES=3
# Positional arguments for the bash scripts
CKPT_PATH=MTRAG-Model-Path
CHECKPOINT_FILE=MTRAG-Model-Name
JSON_PATH=./eval/osprey_description/description/questions.json
RESULT_PATH=./eval_results/osprey_description/${CHECKPOINT_FILE}/osprey_description.json

# Run Inference
python  eval/osprey_description/osprey_generate_gpt_description_answer.py --model "$CKPT_PATH" --json "$JSON_PATH" --result_json "$RESULT_PATH"
# gpt_eval
python eval/osprey_description/eval_gpt.py \
    --question ./eval/osprey_description/description/questions.json \
    --context ./eval/osprey_description/description/prompt.json \
    --answer-list \
    ./eval/osprey_description/description/answers.json \
    ${RESULT_PATH}\
    --rule ./eval/osprey_description/rule.json \
    --output ./eval_results/osprey_description/score/${CHECKPOINT_FILE}/${CHECKPOINT_FILE}.jsonl

#summarize score
python eval/osprey_description/summarize_gpt_score.py --dir ./eval_results/osprey_description/score/${CHECKPOINT_FILE} --save

