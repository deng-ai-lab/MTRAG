## MTRAG: Multi-Target Referring and Grounding via Hybrid Semantic-Spatial Integration (Under Review)
 Official implementation of ``MTRAG: Multi-Target Referring and Grounding via Hybrid Semantic-Spatial Integration''.

 As the paper is currently under peer review, we are releasing only the full-capability model MTRAG-Full (without additional fine-tuning) and the evaluation script for MTR-Bench at this stage. The complete codebase and model checkpoints will be made publicly available upon acceptance

## Installation
See [install](./docs/install.md) for details.

## Pre-trained weights

### Vicuna-7B-v1.5
MTRAG needs loading [vicuna-7b-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5/tree/main) pre-trained weights.
### Alpha-CLIP Encoder
Our Global Image Encoder is initialized with the pre-trained weights of [Alpha-CLIP-L/14@336px](https://drive.google.com/file/d/1dUq1deeLcou26RuxZbBG57m2ALPWev6-/view?usp=drive_link), which has been fine-tuned on the GRIT-20M dataset. Place the downloaded weights in the path `./alpha_clip`.
### SAM weights
Our grounding branch, including both the perception encoder and decoder, is initialized from the ViT-H backbone of the Segment Anything Model (SAM) [ViT-H SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth). The encoder is kept frozen during training. Place the downloaded weights in the path `./checkpoints`.

## Prepare Datasets
See [datasets](./docs/datasets.md) for details.

## Checkpoints 🤖
MTRAG-Full model🤗: [MTRAG-Full](https://huggingface.co/duujy/MTRAG-Full/tree/main)

## Evaluation 🔎
See [evaluation](./docs/evaluation.md) for details.



## Acknowledgement
Thanks for great works of [GLaMM](https://github.com/mbzuai-oryx/groundingLMM), [LLaVA](https://github.com/haotian-liu/LLaVA) and [SAM](https://github.com/facebookresearch/segment-anything). Our code is based on them.