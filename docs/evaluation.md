# Evaluating MTRAG 🔍
This guide provides instructions on evaluating the pretrained MTRAG models on the downstream tasks.


### 1) Grounded Conversation Generation (GCG)
```bash
bash eval/gcg/run_gcg.sh 
```

### 2) Region-Level Captioning
```bash
bash eval/region_captioning/refcocog_eval.sh
```

### 3) Osprey Detailed Description
```bash
bash eval/osprey_description/osprey_gpt_eval.sh
```

### 4) MTR-Bench
Inference:
```bash
bash eval/mtr-bench/mtr_bench_infer.sh
```
Evaluation:
```bash
bash eval/mtr-bench/multi_region_gpt_eval.sh
```

### 5) Referring Expression Segmentation (RES)
```bash
bash eval/single_refseg/single_refseg_eval.sh
```

### 6) Multi-Referring Expression Segmentation (MRES)
```bash
bash eval/multi_refseg/multi_refseg_eval.sh
```

### 7) Reasoning Intention-Oriented Objects (RIO)
```bash
bash eval/rio_seg/rio_seg_eval.sh
```