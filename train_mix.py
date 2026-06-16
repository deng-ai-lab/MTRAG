"""
train.py - MTRAG Model Training on Mixed Datasets
"""
import os
import sys
import time
import tqdm
import random
import torch
import argparse
import deepspeed
import numpy as np
import transformers
from functools import partial
from torch.utils.data import ConcatDataset
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter
from mtrag.MTRAG import MTRAGForCausalLM
from mtrag.llava import conversation as conversation_lib
from dataset.dataset import custom_collate_fn, HybridSegDataset, HybridRegDataset, HybridCapDataset
from mtrag.llava.constants import DEFAULT_REGION_FEA_TOKEN
from tools.utils import (AverageMeter, ProgressMeter, dict_to_cuda, Summary, intersectionAndUnionGPU)

from dataset.singleobject_datasets.ReferExp_Segm import ReferExpSegmDataset
from dataset.singleregion_datasets.RefCOCORegion import RefCocoGRegDataset
from dataset.singleregion_datasets.VGRegion import VisualGenomeRegDataset
from dataset.image_caption_datasets.COCOCaption import CocoCapDataset
from dataset.multiobject_datasets.GCG import  OpenPsgGCGDataset, Flickr30kGCGDataset, RefCOCOgGCGDataset
import json
from types import SimpleNamespace
from mtrag.llava.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
replace_llama_attn_with_flash_attn()
local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

import tokenizers
from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')
print("IS_TOKENIZER_GREATER_THAN_0_14:", IS_TOKENIZER_GREATER_THAN_0_14)

def parse_args(args):
    parser = argparse.ArgumentParser(description="MTRAG Model Training")
    # Model-specific settings
    parser.add_argument("--version", default="/path/to/stage3-checkpoint")
    parser.add_argument("--vision_pretrained", default="./checkpoints/sam_vit_h_4b8939.pth", type=str)
    parser.add_argument("--vision_tower", default="openai/clip-vit-large-patch14-336", type=str)
    parser.add_argument("--vision_module", default="openai/clip-vit-large-patch14-336", type=str)
    parser.add_argument("--vision_tower_alpha", default="./alpha-clip/clip_l14@336_grit_20m_4xe.pth", type=str)
    parser.add_argument("--conv_type", default="llava_v1", type=str, choices=["llava_v1", "llava_llama_2"])
    parser.add_argument("--tune_mm_mlp_adapter", action="store_true")
    parser.add_argument("--freeze_mm_mlp_adapter", action="store_true")
    parser.add_argument("--mm_use_im_start_end", action="store_true", default=False)
    parser.add_argument("--mm_use_im_patch_token", action="store_true", default=False) 
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--grounding_hidden_size", default=256, type=int)
    parser.add_argument("--precision", default='bf16', type=str) 
    parser.add_argument("--add_region_feature", action="store_true",default=False)
    parser.add_argument("--pretrained", action="store_true", default=False)
    parser.add_argument("--mm_vision_select_layer", default=-2, type=int)
    parser.add_argument("--pretrain_mm_mlp_adapter", default=None, type=str)
    parser.add_argument("--pretrain_region_fea_adapter", default=None, type=str)
    parser.add_argument("--mm_projector_type", default='SAMCLIP', type=str)
    parser.add_argument("--mm_patch_merge_type", default='flat', type=str)
    parser.add_argument("--mm_vision_select_feature", default='patch', type=str)
    parser.add_argument("--initial_grounding", action="store_true", default=False)
    parser.add_argument("--train_mask_decoder", action="store_true", default=False)
    parser.add_argument("--use_mm_start_end", action="store_true")
    parser.add_argument("--with_region", action="store_true", default=False)
    parser.add_argument("--region_fea_adapter", action="store_true", default=False)
    parser.add_argument("--select_feature_type", default="cls_patch", type=str)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--model_max_length", default=2048, type=int)


    # Dataset settings
    parser.add_argument("--global_image_encoder", default="openai/clip-vit-large-patch14-336")
    parser.add_argument("--weight_cap", default=0.15, type=float, help="Sampling weight for caption data")
    parser.add_argument("--weight_reg", default=0.40, type=float, help="Sampling weight for region data")
    parser.add_argument("--weight_segm", default=0.45, type=float, help="Sampling weight for segmentation data")
    parser.add_argument("--dataset_dir", default="./data", type=str)
    # Segmentation datasets
    parser.add_argument("--dataset_seg", default=None, type=str, help="Choose from: gReferExpSegm||MOSemanticSegm||MOReferExpSegm||Flickr30kGCG||RefCOCOgGCG||OpenPsgGCG||GranDfGCG||RIOTrainSegm||ReferExpSegm||SemanticSegm")
    parser.add_argument("--sample_rate_seg", default="1,1,1,1,1,1,10,1,1,1", type=str)
    # Region datasets
    parser.add_argument("--dataset_reg", default=None,
                        type=str, help="Choose from: Flickr30kReg||MTRAGMultiturn||MTRAGRelations||RIOMultiReg||OspreyMultiReg||VGReg||RefCocoGReg||RefCocoPReg||RefCocoReg||OspreyConversations||OspreyDetailedDescription||OspreyPartLevel||MDVPReg")
    parser.add_argument("--sample_rate_reg", default=None, type=str)
    # Caption datasets
    parser.add_argument("--dataset_cap", default=None, type=str,
                        help="Choose from: COCOCap||LLaVA-VQA")
    parser.add_argument("--sample_rate_cap", default="1,1", type=str)
    parser.add_argument("--semantic_segm_data", default="ade20k||cocostuff||pascal_part||paco_lvis||mapillary", type=str)
    parser.add_argument("--refer_segm_data", default="refcoco||refcoco+||refcocog||refclef", type=str)
    parser.add_argument("--num_classes_per_sample", default=3, type=int)
    parser.add_argument("--image_size", default=1024, type=int)
    parser.add_argument("--image_aspect_ratio", default="square", type=str)
    parser.add_argument("--split", default="train", type=str)

    # Training settings
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--auto_resume", action="store_true")
    parser.add_argument("--lr", default=0.0003, type=float)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--steps_per_epoch", default=500, type=int)
    parser.add_argument("--batch_size", default=2, type=int, help="batch size per device per step")
    parser.add_argument("--grad_accumulation_steps", default=10, type=int)
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=2, type=int) # default=2
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=0.5, type=float)
    parser.add_argument("--bce_loss_weight", default=2.0, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument("--weight", default=None, type=str, help="load from other checkpoint")
    parser.add_argument("--save_freq", default=1000, type=int)

    # Evaluation settings
    parser.add_argument("--val_dataset", default=None, type=str,
                        help="Choose from: CocoCapVal, RefCOCOgRegVal, VisGenomeRegVal, RefCOCOgSegmVal, PsgGCGVal, "
                             "RefCocoGCGVal, FlickrGCGVal")
    parser.add_argument("--mask_validation", action="store_true") # SegVal, inference=True
    parser.add_argument("--no_eval", action="store_true")
    parser.add_argument("--eval_only", action="store_true")

    # Experiment settings
    parser.add_argument("--log_base_dir", default="./output", type=str)
    parser.add_argument("--exp_name", default="MTRAG-Full", type=str)

    return parser.parse_args(args)


def initialize_environment(args):
    """ Set up logging and model directories. """
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    if args.local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        return SummaryWriter(args.log_dir)
    return None

def initialize_model(args):
    """ Initialize the MTRAG model. """
    rank0_print('\033[92m' + "---- Initialized model from: {} ----".format(args.version) + '\033[0m')
    model_args_dict = {k: getattr(args, k) for k in
                  ["train_mask_decoder", "out_dim", "ce_loss_weight", "dice_loss_weight", "bce_loss_weight",
                   "vision_pretrained", "vision_tower", "vision_module", "use_mm_start_end", "mm_vision_select_layer",
                   "pretrain_mm_mlp_adapter", "freeze_mm_mlp_adapter", "mm_use_im_start_end","mm_use_im_patch_token",
                   "with_region", "vision_tower_alpha", "add_region_feature","region_fea_adapter","select_feature_type","grounding_hidden_size",
                   "pretrain_region_fea_adapter", "tune_mm_mlp_adapter","mm_projector_type","mm_patch_merge_type","mm_vision_select_feature"]}
    model_args = SimpleNamespace(**model_args_dict)
    model = MTRAGForCausalLM.from_pretrained(
        args.version, torch_dtype=(torch.bfloat16 if args.bf16 else None), attn_implementation="flash_attention_2", low_cpu_mem_usage=True, **model_args.__dict__
    )
    if args.initial_grounding:
        model.get_model().initialize_mtrag_model()
    model.config.use_cache = False
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version, model_max_length=args.model_max_length, padding_side="right", use_fast=False
    )
    rank0_print('\033[92m' + "---- Initialized tokenizer from: {} ----".format(args.version) + '\033[0m')
    tokenizer.pad_token = tokenizer.unk_token
    args.is_multimodal = True
    model.config.image_aspect_ratio = args.image_aspect_ratio
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length
    model.config.tune_mm_mlp_adapter = args.tune_mm_mlp_adapter
    model.config.region_fea_adapter = args.region_fea_adapter
    model.config.mm_use_im_start_end = args.mm_use_im_start_end
    model.config.mm_use_im_patch_token = args.mm_use_im_patch_token
    if model_args.train_mask_decoder:
        segmentation_tokens = ['[SEG]'] 
        # Adding tokens for GCG
        phrase_tokens = ['<p>', '</p>']
        special_tokens =  segmentation_tokens + phrase_tokens
        tokenizer.add_tokens(special_tokens, special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))
        # Configure model tokens
        model.config.seg_token_idx = tokenizer.convert_tokens_to_ids(['[SEG]'])[0] 
        model.config.bop_token_idx = tokenizer.convert_tokens_to_ids(['<p>'])[0] 
        model.config.eop_token_idx = tokenizer.convert_tokens_to_ids(['</p>'])[0] 
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.bos_token_id = tokenizer.bos_token_id
    if model_args.add_region_feature:
        tokenizer.add_tokens([DEFAULT_REGION_FEA_TOKEN], special_tokens=True)
        model.config.reg_token_idx = tokenizer.convert_tokens_to_ids([DEFAULT_REGION_FEA_TOKEN])[0]
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def prepare_model_for_training(model, tokenizer, args):
    # Enable input gradients
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    # Initialize vision tower
    rank0_print(
        '\033[92m' + "---- Initialized Global Image Encoder (vision tower) from: {} ----".format(
            args.vision_tower
        ) + '\033[0m'
    )
    model.get_model().initialize_vision_modules(
            model_args=model.get_model().config,
            add_region_feature=model.get_model().config.add_region_feature
        )
    vision_tower = model.get_vision_tower()
    vision_tower.to(dtype=torch.bfloat16 if args.bf16 else torch.float16, device=args.local_rank)
    vision_tower.requires_grad_(False)
    
    for param in model.get_model().grounding_encoder.parameters():
        param.requires_grad = False
    if model.get_model().config.train_mask_decoder:
        model.get_model().grounding_encoder.mask_decoder.train()
        for param in model.get_model().grounding_encoder.mask_decoder.parameters():
            param.requires_grad = True
        # Projection layer
        model.get_model().text_hidden_fcs.train()
        for param in model.get_model().text_hidden_fcs.parameters():
            param.requires_grad = True
    # Set requires_grad for vision tower and mm projector
    for p in vision_tower.parameters():
        p.requires_grad = False
    for p in model.get_model().mm_projector.parameters(): 
        p.requires_grad = False

    # Set requires_grad based on LoRA training
    lora_r = args.lora_r
    if lora_r == 0:
        for p in model.get_model().layers.parameters():
            p.requires_grad = True
        for p in model.get_model().mm_projector.parameters(): 
            p.requires_grad = True

    # Configure conversation library
    conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_type]

    # Configure LoRA if applicable
    if lora_r > 0:
        lora_config = setup_lora_config(model, args)
        model = get_peft_model(model, lora_config)

    # Resize token embeddings
    model.resize_token_embeddings(len(tokenizer)-1)
    if args.weight is not None:
        rank0_print('loading from ', args.weight)
        state_dict = torch.load(args.weight, map_location="cpu")["module"]
        updated_state_dict = {}
        for key in state_dict.keys():
            updated_key = f"base_model.model.{key}"
            updated_state_dict[updated_key] = state_dict[key]
        model.load_state_dict(updated_state_dict, strict=True)
    # Make certain modules trainable
    set_trainable_modules(model)
    

def setup_lora_config(model, args):
    """ Configure LoRA settings for the model. """
    def find_proj_layers(model, target_modules):
        """ Identify projection layers in the model for LoRA adaptation. """
        linear_cls = torch.nn.Linear
        lora_module_names = set()
        # When using LoRA, the linear layers in "grounding_encoder", "vision_tower", "mm_projector", and "text_hidden_fcs" are not adjusted.
        for name, module in model.named_modules():
            if (isinstance(module, linear_cls) and all(
                    x not in name for x in ["grounding_encoder", "vision_tower", "mm_projector", "text_hidden_fcs"]
            ) and any(x in name for x in target_modules)):
                lora_module_names.add(name)
        return sorted(list(lora_module_names))

    # Extracting LoRA target modules
    lora_target_modules = args.lora_target_modules.split(",")

    lora_module_names = find_proj_layers(model, lora_target_modules)

    # Configuring LoRA
    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=lora_module_names, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM"
    )
    return lora_config


def set_trainable_modules(model):
    """ Make specified modules in the model trainable. """
    trainable_modules = ["lm_head", "embed_tokens", "mask_decoder", "text_hidden_fcs", "region_fea_adapter"] # mm_projector untrainable
    for name, param in model.named_parameters():
        if any(module in name for module in trainable_modules):
            rank0_print(f"Making trainable: {name}, Shape: {param.shape}")
            param.requires_grad = True
    if model.config.tune_mm_mlp_adapter:
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True
    if not model.config.region_fea_adapter:
        for p in model.get_model().region_fea_adapter.parameters():
            p.requires_grad = False
    if not model.config.train_mask_decoder:
        for p in model.get_model().grounding_encoder.mask_decoder.parameters():
            p.requires_grad = False
        for p in model.get_model().text_hidden_fcs.parameters():
            p.requires_grad = False
    def count_parameters(model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        rank0_print('\033[92m' + "---- Total parameters: ----{}".format(total_params) + '\033[0m')
        rank0_print('\033[92m' + "---- Trainable parameters: ----{}".format(trainable_params) + '\033[0m')

    count_parameters(model)


def initialize_datasets_and_loaders(args, tokenizer):
    world_size = torch.cuda.device_count()
    args.distributed = world_size > 1
    # Common dataset arguments
    common_ds_args = {"dataset_dir": args.dataset_dir, "tokenizer": tokenizer,
                      "global_image_encoder": args.vision_tower,
                      "epoch_samples": args.batch_size * args.grad_accumulation_steps * args.steps_per_epoch * world_size,
                      "precision": args.precision, "image_size": args.image_size,
                      "num_classes_per_sample": args.num_classes_per_sample,
                      "grounding":args.mm_projector_type == "SAMCLIP",
                      "use_mm_start_end":args.use_mm_start_end
                      }

    # Training datasets
    cap_train_dataset = HybridCapDataset(
        **common_ds_args, dataset_cap=args.dataset_cap, sample_rate_cap=args.sample_rate_cap,
        batch_size=args.batch_size, ) if args.dataset_cap is not None else None
    reg_train_dataset = HybridRegDataset(
        **common_ds_args, dataset_reg=args.dataset_reg, sample_rate_reg=args.sample_rate_reg,
        ) if args.dataset_reg is not None else None
    seg_train_dataset = HybridSegDataset(
        **common_ds_args, dataset_seg=args.dataset_seg, sample_rate_seg=args.sample_rate_seg,
        semantic_segm_data=args.semantic_segm_data, refer_segm_data=args.refer_segm_data ) if args.dataset_seg is not None else None

    # Validation datasets
    val_datasets = []
    if not args.no_eval:
        val_dataset_classes = {'CocoCapVal': CocoCapDataset,
                               'RefCOCOgRegVal': RefCocoGRegDataset,
                               'VisGenomeRegVal': VisualGenomeRegDataset,
                               'RefCOCOgSegmVal': ReferExpSegmDataset,
                               'PsgGCGVal': OpenPsgGCGDataset,
                               'RefCocoGCGVal': RefCOCOgGCGDataset,
                               'FlickrGCGVal': Flickr30kGCGDataset,
                               }
        for val_dataset_name in args.val_dataset.split('|'):
            val_dataset_class = val_dataset_classes.get(val_dataset_name)
            if val_dataset_class:
                if val_dataset_class == ReferExpSegmDataset:
                    # Modify this if other datasets in refer_segm_data need to be included in val
                    refer_segm_data = 'refcocog'
                    all_datasets = refer_segm_data.split("||")
                    for d in all_datasets:
                        val_dataset_class = val_dataset_class(
                            **common_ds_args, validation=True, refer_segm_data=d, split='val'
                        )
                        val_dataset_class._set_len(len(val_dataset_class.refer_segm_data[d]['images']))
                        val_datasets.append(val_dataset_class)
                else:
                    val_datasets.append(val_dataset_class(**common_ds_args, validation=True))

    return cap_train_dataset, reg_train_dataset,seg_train_dataset, val_datasets


def setup_data_loaders(args, cap_train_dataset, reg_train_dataset, seg_train_dataset, val_datasets, tokenizer):
    sampler_args = {"shuffle": False, "drop_last": False}
    train_loader_args = {"batch_size": args.batch_size, "shuffle": False, "num_workers": args.workers,
                         "pin_memory": False}
    val_loader_args = {"batch_size": args.val_batch_size, "shuffle": False, "num_workers": args.workers,
                       "pin_memory": False}
    collate_fn_args_train = partial(
        custom_collate_fn, tokenizer=tokenizer, local_rank=args.local_rank,
        inference=False
    )
    inference_mode = args.mask_validation
    collate_fn_args_val = partial(
        custom_collate_fn, tokenizer=tokenizer, local_rank=args.local_rank,
        inference=inference_mode
    )

    # Training loaders
    cap_train_loader = torch.utils.data.DataLoader(
        cap_train_dataset, sampler=torch.utils.data.distributed.DistributedSampler(
            cap_train_dataset, **sampler_args
        ), collate_fn=collate_fn_args_train, **train_loader_args
    , ) if cap_train_dataset is not None else None
    reg_train_loader = torch.utils.data.DataLoader(
        reg_train_dataset, sampler=torch.utils.data.distributed.DistributedSampler(
            reg_train_dataset, **sampler_args
        ), collate_fn=collate_fn_args_train, **train_loader_args
    ) if reg_train_dataset is not None else None
    seg_train_loader = torch.utils.data.DataLoader(
        seg_train_dataset, sampler=torch.utils.data.distributed.DistributedSampler(
            seg_train_dataset, **sampler_args
        ), collate_fn=collate_fn_args_train, **train_loader_args
    ) if seg_train_dataset is not None else None

    # Validation loader
    val_loader = None
    if val_datasets:
        combined_val_datasets = ConcatDataset(val_datasets)
        val_loader = torch.utils.data.DataLoader(
            combined_val_datasets, **val_loader_args, collate_fn=collate_fn_args_val,
            sampler=torch.utils.data.distributed.DistributedSampler(combined_val_datasets, **sampler_args), )

    return cap_train_loader, reg_train_loader, seg_train_loader, val_loader


def initialize_deepspeed(model, tokenizer, args):
    ds_config = {"train_micro_batch_size_per_gpu": args.batch_size,
                 "gradient_accumulation_steps": args.grad_accumulation_steps,
                 "optimizer": {"type": "AdamW", "params": {"lr": args.lr, "weight_decay": 0.0,
                                                           "betas": (args.beta1, args.beta2)}},
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


def resume_training_from_checkpoint(model_engine, args):
    if args.auto_resume and not args.resume:
        resume = os.path.join(args.log_dir, "ckpt_model")
        if os.path.exists(resume):
            args.resume = resume

    if args.resume:
        load_path, client_state = model_engine.load_checkpoint(args.resume)
        with open(os.path.join(args.resume, "latest"), "r") as f:
            ckpt_dir = f.readlines()[0].strip()
        args.start_epoch = int(ckpt_dir.replace("global_step", "")) // args.steps_per_epoch
        rank0_print(f"Resume training from {args.resume}, start from epoch {args.start_epoch}")
def get_grad_norm(model):
    grad_sum = torch.tensor(0.0).to(model.local_rank)
    for name, param in model.named_parameters():
        with deepspeed.zero.GatheredParameters(param, modifier_rank=None):
            if param.grad is not None:
                rank0_print(f"Parameter: {name}, Gradient: {param.grad}")
                grad_sum += param.grad.sum()
            else:
                rank0_print(f"Parameter: {name} has no gradient.")
    return grad_sum
def main(args):
    global local_rank
    local_rank = args.local_rank
    model, tokenizer = initialize_model(args)
    prepare_model_for_training(model, tokenizer, args)
    model_engine, optimizer, scheduler = initialize_deepspeed(model, tokenizer, args)
    resume_training_from_checkpoint(model_engine, args)
    cap_train_dataset, reg_train_dataset, seg_train_dataset, val_datasets = (
        initialize_datasets_and_loaders(args, tokenizer))
    cap_train_loader, reg_train_loader, seg_train_loader, val_loader = (
        setup_data_loaders(args, cap_train_dataset, reg_train_dataset, seg_train_dataset, val_datasets, tokenizer))

    # Determine active datasets
    use_cap_data = False
    use_reg_data = False
    use_segm_data = False
    if args.dataset_cap is not None:
        use_cap_data = True
    if args.dataset_reg is not None:
        use_reg_data = True
    if args.dataset_seg is not None:
        use_segm_data = True

    active_dataloaders = {
        'cap': cap_train_loader,
        'reg': reg_train_loader,
        'seg': seg_train_loader,
    }

    # Assert that at least one dataset is active
    assert active_dataloaders, "Error: At least one dataset (segm, reg, or cap) must be active."

    dataset_iters = {'cap': iter(cap_train_loader) if use_cap_data else None,
                     'reg': iter(reg_train_loader) if use_reg_data else None,
                     'seg': iter(seg_train_loader) if use_segm_data else None, }

    writer = initialize_environment(args)

    # save configs and args
    save_args_path = os.path.join(args.log_dir, "args.json")
    os.makedirs(args.log_dir, exist_ok=True)
    with open(save_args_path, "w") as f:
        json.dump({"args": vars(args), "model_configs": model.config.to_dict()}, f, indent=4)


    if args.eval_only:
        cur_val_loss = validate_model_performance(val_loader, model_engine, 0, writer, args)[0]
        exit()

    epoch_seeds = [random.randint(0, 100000) for _ in range(args.epochs)]
    best_giou, best_ciou, best_val_loss = 0.0, 0.0, np.inf
    for epoch in range(args.start_epoch, args.epochs):
        random.seed(epoch_seeds[epoch])

        dataset_iters = train(
            active_dataloaders, model_engine, epoch, scheduler, writer, dataset_iters, args
        )

        if args.mask_validation:
            giou, ciou = validate_model_performance(val_loader, model_engine, epoch, writer, args)
            is_best = giou > best_giou
            best_giou = max(giou, best_giou)
            best_ciou = ciou if is_best else best_ciou
            if args.local_rank == 0:  # Log the progress
                rank0_print(f"Epoch: {epoch}, giou: {giou}, ciou: {ciou}, best_giou: {best_giou}, best_ciou: {best_ciou}")
            save_checkpoint(model_engine, args, epoch, 'giou-ciou', f"{giou:.4f}-{ciou:.4f}", is_best)
        else:
            cur_val_loss = validate_model_performance(val_loader, model_engine, epoch, writer, args)
            is_best = cur_val_loss < best_val_loss
            best_val_loss = min(cur_val_loss, best_val_loss)
            if args.local_rank == 0:  # Log the progress
                rank0_print(f"Epoch: {epoch}, Current Validation Loss: {cur_val_loss:.4f}, Best Validation Loss: {best_val_loss:}")
            save_checkpoint(model_engine, args, epoch, 'loss', f"{cur_val_loss:.4f}", is_best)


def save_checkpoint(model_engine, args, epoch, metric_name, metric_value, is_best):
    """ Saves the model checkpoint. """
    # If the checkpoint is the best, save it in ckpt_model_best, else in ckpt_model_last_epoch
    save_dir_name = "ckpt_model_best" if is_best else "ckpt_model_last_epoch"
    save_dir = os.path.join(args.log_dir, save_dir_name)
    # Ensure the directory exists
    if args.local_rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        ckpt_filename = f"epoch_{epoch}_val_{metric_name}_{metric_value}.pth"
        torch.save({"epoch": epoch, f"val_{metric_name}": metric_value}, os.path.join(save_dir, ckpt_filename))
    torch.distributed.barrier()
    model_engine.save_checkpoint(save_dir)


def train(active_datasets, model, epoch, scheduler, writer, dataset_iters, args):
    """Main training loop."""

    def get_next_input(iterator, data_loader):
        """Retrieve next input from the iterator, or reinitialize if necessary."""
        try:
            return next(iterator), iterator
        except StopIteration:
            new_iterator = iter(data_loader)
            return next(new_iterator), new_iterator

    def log_progress():
        """Log training progress."""
        if global_step % args.print_freq == 0:
            if args.distributed:
                for tracker in trackers.values():
                    tracker.all_reduce()

            if args.local_rank == 0:
                progress.display(global_step + 1)
                for key, tracker in trackers.items():
                    writer.add_scalar(f"train/{key}", tracker.avg, global_step)
                writer.add_scalar("metrics/total_secs_per_batch", batch_time.avg, global_step)
                writer.add_scalar("metrics/data_secs_per_batch", data_time.avg, global_step)
                for key, tracker in trackers.items():
                    writer.add_scalar(f"train/{key}_all_epoch", tracker.avg, global_step+epoch*args.steps_per_epoch)
            for tracker in trackers.values():
                tracker.reset()

    batch_time = AverageMeter("Time", ":.4f")
    data_time = AverageMeter("Data", ":.4f")
    trackers = {"loss": AverageMeter("Loss", ":.4f"),
                "ce_loss": AverageMeter("CeLoss", ":.4f"),
                "mask_bce_loss": AverageMeter("MaskBCELoss", ":.4f"),
                "mask_dice_loss": AverageMeter("MaskDICELoss", ":.4f"),
                "mask_loss": AverageMeter("MaskLoss", ":.4f")}
    progress = ProgressMeter(args.steps_per_epoch, list(trackers.values()), prefix=f"Epoch: [{epoch}]")

    model.train()
    end = time.time()
    for global_step in range(args.steps_per_epoch):
        if epoch == 0: # warmup for segm
            for _ in range(args.grad_accumulation_steps):
                # Select data loader based on step choice
                dataset_type = "seg"
                data_loader = active_datasets[dataset_type]
                data_batch, new_iter = get_next_input(dataset_iters[dataset_type], data_loader)
                dataset_iters[dataset_type] = new_iter

                data_time.update(time.time() - end)
                # Prepare data and convert relevant tensors to bfloat16
                data_batch = dict_to_cuda(data_batch)
                for key in ["global_enc_images", "grounding_enc_images"]:
                    if data_batch[key] is not None:
                        data_batch[key] = data_batch[key].bfloat16()

                output_dict = model(**data_batch)

                # Update training metrics
                for key, tracker in trackers.items():
                    if key in output_dict:
                        tracker.update(output_dict[key].item(), data_batch["global_enc_images"].size(0))
                model.backward(output_dict["loss"])
                model.step()
        else:
            if global_step % 2 == 0:
                for _ in range(args.grad_accumulation_steps):
                    # Select data loader based on step choice
                    dataset_type = "reg"
                    data_loader = active_datasets[dataset_type]
                    data_batch, new_iter = get_next_input(dataset_iters[dataset_type], data_loader)
                    dataset_iters[dataset_type] = new_iter

                    data_time.update(time.time() - end)
                    # Prepare data and convert relevant tensors to bfloat16
                    data_batch = dict_to_cuda(data_batch)
                    for key in ["global_enc_images", "grounding_enc_images"]:
                        if data_batch[key] is not None:
                            data_batch[key] = data_batch[key].bfloat16()

                    output_dict = model(**data_batch)

                    # Update training metrics
                    trackers["loss"].update(output_dict["loss"].item(), data_batch["global_enc_images"].size(0))
                    trackers["ce_loss"].update(output_dict["ce_loss"].item(), data_batch["global_enc_images"].size(0))
                     
                    model.backward(output_dict["loss"])
                    model.step()
            if global_step % 4 == 0:
                for _ in range(args.grad_accumulation_steps):
                     
                    # Select data loader based on step choice
                    dataset_type = "cap"
                    data_loader = active_datasets[dataset_type]
                    data_batch, new_iter = get_next_input(dataset_iters[dataset_type], data_loader)
                    dataset_iters[dataset_type] = new_iter

                    data_time.update(time.time() - end)
                    # Prepare data and convert relevant tensors to bfloat16
                    data_batch = dict_to_cuda(data_batch)
                    for key in ["global_enc_images", "grounding_enc_images"]:
                        if data_batch[key] is not None:
                            data_batch[key] = data_batch[key].bfloat16()

                    output_dict = model(**data_batch)

                    # Update training metrics
                    trackers["loss"].update(output_dict["loss"].item(), data_batch["global_enc_images"].size(0))
                    trackers["ce_loss"].update(output_dict["ce_loss"].item(), data_batch["global_enc_images"].size(0))
                    model.backward(output_dict["loss"])
                    model.step()
            if global_step % 1 == 0:
                for _ in range(args.grad_accumulation_steps):
                     
                    # Select data loader based on step choice
                    dataset_type = "seg"
                    data_loader = active_datasets[dataset_type]
                    data_batch, new_iter = get_next_input(dataset_iters[dataset_type], data_loader)
                    dataset_iters[dataset_type] = new_iter

                    data_time.update(time.time() - end)
                    # Prepare data and convert relevant tensors to bfloat16
                    data_batch = dict_to_cuda(data_batch)
                    for key in ["global_enc_images", "grounding_enc_images"]:
                        if data_batch[key] is not None:
                            data_batch[key] = data_batch[key].bfloat16()

                    output_dict = model(**data_batch)

                    # Update training metrics
                    for key, tracker in trackers.items():
                        if key in output_dict:
                            tracker.update(output_dict[key].item(), data_batch["global_enc_images"].size(0))

                    model.backward(output_dict["loss"])
                    model.step()

        batch_time.update(time.time() - end)
        end = time.time()
        log_progress()

        if global_step != 0:
            curr_lr = scheduler.get_last_lr()
            if args.local_rank == 0:
                writer.add_scalar("train/lr", curr_lr[0], global_step)
                writer.add_scalar("train/lr_all_epoch", curr_lr[0], global_step+epoch*args.steps_per_epoch) 


    return dataset_iters


def validate_model_performance(validation_loader, training_model, current_epoch, tensorboard_writer, args):
    if args.mask_validation:
        # For use with only segmentation/GCG type datasets
        trackers = {"intersection": AverageMeter("Intersec", ":.4f", Summary.SUM),
                    "union": AverageMeter("Union", ":.4f", Summary.SUM),
                    "gIoU": AverageMeter("gIoU", ":.4f", Summary.SUM)}

        training_model.eval()
        for data_batch in tqdm.tqdm(validation_loader):
            # Prepare data and convert relevant tensors to bfloat16
            data_batch = dict_to_cuda(data_batch)
            for key in ["global_enc_images", "grounding_enc_images"]:
                data_batch[key] = data_batch[key].bfloat16()
            torch.cuda.empty_cache()
            # Model inference without gradient tracking
            with torch.no_grad():
                results = training_model(**data_batch)

            predictions = results["pred_masks"]
            gt_masks = results["gt_masks"][0].int()
            # segmentation tasks. Ensure that the dataset is appropriate for segmentation analysis.
            predicted_masks = (predictions[0] > 0).int()
            assert len(predictions) == 1

            intersection, union, accuracy_iou = 0.0, 0.0, 0.0
            for target, prediction in zip(gt_masks, predicted_masks):
                intersect, union_, _ = intersectionAndUnionGPU(
                    prediction.contiguous().clone(), target.contiguous(), 2, ignore_index=255
                )
                intersection += intersect
                union += union_
                accuracy_iou += intersect / (union_ + 1e-5)
                # handles no-object targets
                accuracy_iou[union_ == 0] += 1.0

            intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
            accuracy_iou = accuracy_iou.cpu().numpy() / gt_masks.shape[0]
            trackers["intersection"].update(intersection)
            trackers["union"].update(union)
            trackers["gIoU"].update(accuracy_iou, n=gt_masks.shape[0])

        for meter in trackers.values():
            meter.all_reduce()

        iou_per_class = trackers["intersection"].sum / (trackers["union"].sum + 1e-10)
        class_iou = iou_per_class[1]
        global_iou = trackers["gIoU"].avg[1]

        if args.local_rank == 0:
            tensorboard_writer.add_scalar("val/giou", global_iou, current_epoch)
            tensorboard_writer.add_scalar("val/ciou", class_iou, current_epoch)
            rank0_print("giou: {:.4f}, ciou: {:.4f}".format(global_iou, class_iou))

        return global_iou, class_iou
    else:
        # Initializing performance trackers
        trackers = {"loss": AverageMeter("Loss", ":.4f"), "ce_loss": AverageMeter("CeLoss", ":.4f"),
                    "mask_bce_loss": AverageMeter("MaskBCELoss", ":.4f"),
                    "mask_dice_loss": AverageMeter("MaskDICELoss", ":.4f"),
                    "mask_loss": AverageMeter("MaskLoss", ":.4f")}

        # Prepare model for validation phase
        # Hack to get the loss
        training_model.train()

        for data_batch in tqdm.tqdm(validation_loader):
            # Prepare data and convert relevant tensors to bfloat16
            data_batch = dict_to_cuda(data_batch)
            for key in ["global_enc_images", "grounding_enc_images"]:
                if data_batch[key] is not None:
                    data_batch[key] = data_batch[key].bfloat16()
            torch.cuda.empty_cache()
            # Model inference without gradient tracking
            with torch.no_grad():
                predictions = training_model(**data_batch)
            # Update performance metrics)
            for key, tracker in trackers.items():
                tracker.update(predictions[key].item(), data_batch["global_enc_images"].size(0))

        # Synchronize metrics across processes
        for tracker in trackers.values():
            tracker.all_reduce()
        # Calculate average validation loss
        avg_val_loss = trackers["ce_loss"].avg
        # Tensorboard logging for primary process
        if args.local_rank == 0:
            tensorboard_writer.add_scalar("val/loss", avg_val_loss, current_epoch)
        return avg_val_loss


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
