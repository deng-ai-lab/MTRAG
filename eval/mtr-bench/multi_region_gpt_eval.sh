#!/bin/bash

CHECKPOINT_FILE=MTRAG-Model-Name

# gpt_eval_description
python eval/mtr-bench/eval_gpt.py \
    --question ./eval/mtr-bench/description/questions.json \
    --context ./eval/mtr-bench/description/prompt.json \
    --answer-list \
    ./eval/mtr-bench/description/answers.json \
    ./eval_results/mtr-bench/${CHECKPOINT_FILE}/mtr_description.json\
    --rule ./eval/mtr-bench/rule.json \
    --output ./eval_results/mtr-bench/score/${CHECKPOINT_FILE}/${CHECKPOINT_FILE}_description.jsonl &
python eval/mtr-bench/eval_gpt.py \
    --question ./eval/mtr-bench/relations/questions.json \
    --context ./eval/mtr-bench/relations/prompt.json \
    --answer-list \
    ./eval/mtr-bench/relations/answers.json \
    ./eval_results/mtr-bench/${CHECKPOINT_FILE}/mtr_relations.json\
    --rule ./eval/mtr-bench/rule.json \
    --output ./eval_results/multi_region/score/${CHECKPOINT_FILE}/${CHECKPOINT_FILE}_relations.jsonl
#summarize score
python eval/mtr-bench/summarize_gpt_score.py --dir ./eval_results/mtr-bench/score/${CHECKPOINT_FILE} --save

