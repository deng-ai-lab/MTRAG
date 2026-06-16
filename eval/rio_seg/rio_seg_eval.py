import os
import sys
import json
import tqdm
import torch
import argparse
import deepspeed
from functools import partial
from torch.utils.data import ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from mtrag.MTRAG import MTRAGForCausalLM
from mtrag.llava import conversation as conversation_lib
from dataset.dataset import custom_collate_fn
from tools.utils import (AverageMeter, Summary,intersectionAndUnionGPU, dict_to_cuda)
from transformers import AutoTokenizer
local_rank = None
from dataset.multiobject_datasets.RIO_Segm import RIOTrainSegmDataset, RIOUncomSegmDataset, RIOcomSegmDataset

def parse_args(args):
    parser = argparse.ArgumentParser(description="MTRAG Inference - RIO")

    parser.add_argument("--version", required=True, help="The model path in huggingface format.")
    parser.add_argument("--pretrained", default=True, type=bool)
    parser.add_argument("--vision_pretrained", default="./checkpoints/sam_vit_h_4b8939.pth", type=str)
    parser.add_argument("--vision_tower", default="openai/clip-vit-large-patch14-336", type=str)
    parser.add_argument("--vision_module", default="openai/clip-vit-large-patch14-336", type=str)
    parser.add_argument("--vision_tower_alpha", default="./alpha-clip/clip_l14@336_grit_20m_4xe.pth", type=str)
    parser.add_argument("--tune_mm_mlp_adapter", default=False, type=bool)
    parser.add_argument("--freeze_mm_mlp_adapter", default=False, type=bool)
    parser.add_argument("--mm_use_im_start_end", default=False, type=bool)
    parser.add_argument("--mm_use_im_patch_token", default=False, type=bool) 
    parser.add_argument("--model_max_length", default=2048, type=int)
    parser.add_argument("--conv_type", default="llava_v1", type=str, choices=["llava_v1", "llava_llama_2"])
    parser.add_argument("--precision", default='bf16', type=str)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--mm_vision_select_layer", default=-2, type=int)
    parser.add_argument("--pretrain_mm_mlp_adapter", default=None, type=str)
    parser.add_argument("--pretrain_region_fea_adapter", default=None, type=str)
    parser.add_argument("--mm_projector_type", default='SAMCLIP', type=str)
    parser.add_argument("--mm_patch_merge_type", default='flat', type=str)
    parser.add_argument("--initial_grounding", default=False, type=bool)
    parser.add_argument("--train_mask_decoder", default=False, type=bool)
    parser.add_argument("--use_mm_start_end", default=False, type=bool)
    parser.add_argument("--with_region", default=True, type=bool)
    parser.add_argument("--region_fea_adapter", default=True, type=bool)
    parser.add_argument("--select_feature_type", default="cls_patch", type=str)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)

    # Training settings
    parser.add_argument("--bf16", default=True, type=bool)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--auto_resume", action="store_true")
    parser.add_argument("--lr", default=0.0003, type=float)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--steps_per_epoch", default=500, type=int)
    parser.add_argument("--batch_size", default=2, type=int, help="batch size per device per step")
    parser.add_argument("--grad_accumulation_steps", default=10, type=int)
    parser.add_argument("--lora_r", default=16, type=int)
    parser.add_argument("--lora_alpha", default=32, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=0.5, type=float)
    parser.add_argument("--bce_loss_weight", default=2.0, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--gradient_checkpointing", default=True, type=bool)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--weight", default=None, type=str, help="load from other checkpoint")
    parser.add_argument("--save_freq", default=1000, type=int)

    # Dataset settings
    parser.add_argument("--global_image_encoder", default="openai/clip-vit-large-patch14-336")
    parser.add_argument("--num_classes_per_sample", default=3, type=int)
    parser.add_argument("--image_aspect_ratio", default="square", type=str)
    parser.add_argument("--split", default="train", type=str)
    parser.add_argument("--refer_seg_data", default="refcocog|val", type=str)
    parser.add_argument("--results_path", default="referring_seg_eval.json", type=str)
    parser.add_argument("--dataset_dir", default="./data", type=str)
    parser.add_argument("--image_size", default=1024, type=int, help="Image size for grounding image encoder")

    # Evaluation settings
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=2, type=int)
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")

    # Experiment settings
    parser.add_argument("--log_base_dir", default="./runs", type=str)
    parser.add_argument("--exp_name", default="mtrag_eval_rio", type=str)

    return parser.parse_args(args)


def initialize_environment(args):
    """ Set up logging and model directories. """
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    if args.local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        return SummaryWriter(args.log_dir)
    return None


def initialize_deepspeed(model, tokenizer, args):
    ds_config = {"train_micro_batch_size_per_gpu": args.batch_size,
                 "gradient_accumulation_steps": args.grad_accumulation_steps, "optimizer": {"type": "AdamW",
                                                                                            "params": {"lr": args.lr,
                                                                                                       "weight_decay": 0.0,
                                                                                                       "betas": (
                                                                                                           args.beta1,
                                                                                                           args.beta2)}},
                 "scheduler": {"type": "WarmupDecayLR",
                               "params": {"total_num_steps": args.epochs * args.steps_per_epoch, "warmup_min_lr": 0,
                                          "warmup_max_lr": args.lr, "warmup_num_steps": 100, "warmup_type": "linear"}},
                 "fp16": {"enabled": args.precision == "fp16"}, "bf16": {"enabled": args.precision == "bf16"},
                 "gradient_clipping": 1.0,
                 "zero_optimization": {"stage": 2, "contiguous_gradients": True, "overlap_comm": True,
                                       "reduce_scatter": True, "reduce_bucket_size": 5e8,
                                       "allgather_bucket_size": 5e8}, }

    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model, model_parameters=model.parameters(), collate_fn=partial(
            custom_collate_fn, tokenizer=tokenizer, use_mm_start_end=args.use_mm_start_end, local_rank=args.local_rank
        ), config=ds_config
    )
    return model_engine, optimizer, scheduler


def initialize_datasets_and_loaders(args, tokenizer):
    # Dataset settings for ReferSegDataset
    common_ds_args = {
        "dataset_dir": args.dataset_dir,
        "tokenizer": tokenizer,
        "global_image_encoder": args.global_image_encoder,
        "precision": args.precision,
        "image_size": args.image_size,
        "grounding":args.mm_projector_type == "SAMCLIP",
        "use_mm_start_end":args.use_mm_start_end
    }

    # Validation datasets
    datasets_config = {
            "RIOUncomSegm": RIOUncomSegmDataset,
            "RIOcomSegm": RIOcomSegmDataset,
    }
    val_datasets = [datasets_config[args.refer_seg_data](**common_ds_args)]
    return val_datasets


def setup_data_loaders(args, val_datasets, tokenizer):
    sampler_args = {"shuffle": False, "drop_last": False}
    val_loader_args = {"batch_size": args.val_batch_size, "shuffle": False, "num_workers": args.workers,
                       "pin_memory": False}
    collate_fn_args_val = partial(
        custom_collate_fn, tokenizer=tokenizer, local_rank=args.local_rank,
        inference=True
    )

    # Validation loader
    combined_val_datasets = ConcatDataset(val_datasets)
    val_loader = torch.utils.data.DataLoader(
        combined_val_datasets, sampler=torch.utils.data.distributed.DistributedSampler(
            combined_val_datasets, **sampler_args
        ), **val_loader_args, collate_fn=collate_fn_args_val, )

    return val_loader



def evaluate_model_performance(validation_loader, model, tokenizer, args):
    # Trackers for metrics
    trackers = {
        "intersection": AverageMeter("Intersec", ":6.3f", Summary.SUM),
        "union": AverageMeter("Union", ":6.3f", Summary.SUM),
        "gIoU": AverageMeter("gIoU", ":6.3f", Summary.SUM)
    }
    os.makedirs(os.path.join(args.results_path, args.refer_seg_data), exist_ok=True)
    model.eval()
    # global_step = 0 # for visualizing
    for data_batch in tqdm.tqdm(validation_loader):
        # Prepare data and convert relevant tensors to the appropriate type
        data_batch = dict_to_cuda(data_batch)
        # model_answer = {} # for visualizing
        for key in ["global_enc_images", "grounding_enc_images"]:
            data_batch[key] = data_batch[key].to(dtype=torch.bfloat16, device=args.local_rank)
        torch.cuda.empty_cache()
        # Model inference without gradient tracking
        with torch.no_grad():
            output_ids, pred_masks = model.evaluate(
                global_enc_images=data_batch["global_enc_images"],
                grounding_enc_images=data_batch["grounding_enc_images"],
                input_ids=data_batch["input_ids"],
                resize_list=data_batch["resize_list"],
                label_list=[item.shape for item in data_batch["label_list"]],
                max_tokens_new=512,
                region_masks=data_batch["region_masks"],
                )
        """batch_size = 1"""
        gt_masks = data_batch["masks_list"][0].sum(dim=0, keepdim=True).clamp(min=0.0, max=1.0).int()
        if pred_masks is None:
            predictions = [torch.zeros_like(gt_masks)]
            predicted_masks = (predictions[0] > 0).int()
        else:
            predictions = (pred_masks[0] > 0).int()
            predicted_masks = predictions.sum(dim=0, keepdim=True).clamp(min=0.0, max=1.0).int()  # Thresholding to get binary masks


        assert len(predicted_masks) == 1
        intersection, union, accuracy_iou = 0.0, 0.0, 0.0
        for target, prediction in zip(gt_masks, predicted_masks):
            intersect, union_, _ = intersectionAndUnionGPU(
                prediction.contiguous().clone(), target.contiguous(), 2, ignore_index=255
            )
            intersection += intersect
            union += union_
            accuracy_iou += intersect / (union_ + 1e-5)
            accuracy_iou[union_ == 0] += 1.0

    for meter in trackers.values():
        meter.all_reduce()

    iou_per_class = trackers["intersection"].sum / (trackers["union"].sum + 1e-10)
    class_iou = iou_per_class[1]
    global_iou = trackers["gIoU"].avg[1]

    return global_iou, class_iou


def main(args):
    if args.conv_type in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_type]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]
    global local_rank
    local_rank = args.local_rank
    tokenizer = AutoTokenizer.from_pretrained(args.version, cache_dir=None,
                                              model_max_length=args.model_max_length, padding_side="right",
                                              use_fast=False)
    torch_dtype = torch.bfloat16  # By default, using bf16
    kwargs = {"torch_dtype": torch_dtype}
    model = MTRAGForCausalLM.from_pretrained(args.version, low_cpu_mem_usage=True, **kwargs)
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.seg_token_idx = tokenizer.convert_tokens_to_ids(['[SEG]'])[0]
    model.config.bop_token_idx = tokenizer.convert_tokens_to_ids(['<p>'])[0]
    model.config.eop_token_idx = tokenizer.convert_tokens_to_ids(['</p>'])[0] 
    model.get_model().initialize_vision_modules(model.get_model().config, add_region_feature=True)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)
    model.requires_grad_(False)
    for param in model.get_model().region_fea_adapter.parameters():
        param.requires_grad = True
    model_engine, _, _ = initialize_deepspeed(model, tokenizer, args)
    val_datasets = initialize_datasets_and_loaders(args, tokenizer)
    val_loader = setup_data_loaders(args, val_datasets, tokenizer)
    giou, ciou = evaluate_model_performance(val_loader, model_engine, tokenizer, args)

    # torch.distributed.barrier()
    if args.local_rank == 0:
        # Update and save the results
        os.makedirs(args.results_path, exist_ok=True)
        if os.path.exists(f"{args.results_path}/stats.json"):
            with open(f"{args.results_path}/stats.json", 'r') as json_file:
                result_list = json.load(json_file)
        else:
            result_list = []
        result_dict = {"model": args.results_path, "dataset": args.refer_seg_data, "giou": str(giou), "ciou": str(ciou)}
        result_list.append(result_dict)

        with open(f"{args.results_path}/stats.json", 'w') as json_file:
            json.dump(result_list, json_file, indent=2)

        print(result_list)  # Print all the results


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
