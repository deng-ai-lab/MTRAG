import os
import torch
import argparse
from peft import get_peft_model
from train_mix import initialize_model, setup_lora_config

def rewrited_forward_embedding(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
    patch_embeds = self.patch_embedding(pixel_values[:,:3,:,:])  # shape = [B, width, grid, grid]
    _, dim, grid, _ = patch_embeds.shape
    if pixel_values.shape[1] >= 4:
        # print("[Warning] no mask specified!")
        all_mask = torch.ones_like(pixel_values[:, [3], :, :]) * 1.9231
        alpha = torch.concat([all_mask, pixel_values[:, 3:, :, :]],dim=1)
    else:
        alpha = torch.ones_like((pixel_values[:, [0], :, :])) * 1.9231
    B, M, H, W = alpha.shape  # M = 1 + NUM_REGION
    patch_embeds = patch_embeds.unsqueeze(1).repeat(1, M, 1, 1, 1).reshape(B*M, dim, grid, grid)
    patch_embeds = patch_embeds + self.patch_embedding_alpha(alpha.reshape(B*M, 1, H, W))
    patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
    class_embeds = self.class_embedding.expand(B*M, 1, -1)
    embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
    embeddings = embeddings + self.position_embedding(self.position_ids)
    return embeddings

def parse_args():
    parser = argparse.ArgumentParser(description="MTRAG: Merge lora weights and save model in hf format")
    parser.add_argument("--version", default="./path/to/stage3-checkpoint", help='Path to the base model.')
    parser.add_argument("--vision_pretrained", default="./checkpoints/sam_vit_h_4b8939.pth", type=str)
    parser.add_argument("--weight", default="./path/to/stage4-checkpoint", type=str, help="Path to the .bin model "
                                                                  "(generated using the script zero_to_fp32.py)")
    parser.add_argument("--save_path", default="./path/to/save_megered_model", type=str, help="Path to save the hf model.")
    # Model-specific settings
    parser.add_argument("--vision_tower", default="openai/clip-vit-large-patch14-336", type=str)
    parser.add_argument("--vision_module", default="openai/clip-vit-large-patch14-336", type=str)
    parser.add_argument("--vision_tower_alpha", default="./alpha-clip/clip_l14@336_grit_20m_4xe.pth", type=str)
    parser.add_argument("--conv_type", default="llava_v1", type=str, choices=["llava_v1", "llava_llama_2"])
    parser.add_argument("--tune_mm_mlp_adapter", action="store_true")
    parser.add_argument("--freeze_mm_mlp_adapter", action="store_true")
    parser.add_argument("--mm_use_im_start_end", default=False, type=bool)
    parser.add_argument("--mm_use_im_patch_token", default=False, type=bool) 
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--grounding_hidden_size", default=256, type=int)
    parser.add_argument("--precision", default='bf16', type=str)
    parser.add_argument("--add_region_feature", default=True, type=bool)
    parser.add_argument("--pretrained", default=False, type=bool)
    parser.add_argument("--mm_vision_select_layer", default=-2, type=int)
    parser.add_argument("--pretrain_mm_mlp_adapter", default=None, type=str)
    parser.add_argument("--pretrain_region_fea_adapter", default=None, type=str)
    parser.add_argument("--mm_projector_type", default='SAMCLIP', type=str)
    parser.add_argument("--mm_patch_merge_type", default='flat', type=str)
    parser.add_argument("--mm_vision_select_feature", default='patch', type=str)
    parser.add_argument("--initial_grounding", default=True, type=bool)
    parser.add_argument("--train_mask_decoder", default=True, type=bool)
    parser.add_argument("--use_mm_start_end", default=False, type=bool)
    parser.add_argument("--with_region", default=True, type=bool)
    parser.add_argument("--region_fea_adapter", default=True, type=bool)
    parser.add_argument("--select_feature_type", default="cls_patch", type=str)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--model_max_length", default=2048, type=int)
    parser.add_argument("--image_aspect_ratio", default="square", type=str)

    # Training settings
    parser.add_argument("--bf16", default=True, type=bool)
    parser.add_argument("--lora_r", default=16, type=int)
    parser.add_argument("--lora_alpha", default=32, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=0.5, type=float)
    parser.add_argument("--bce_loss_weight", default=2.0, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--gradient_checkpointing", default=True, type=bool)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    return parser.parse_args()


def main():
    args = parse_args()
    # Create output directory if not exists already
    os.makedirs(args.save_path, exist_ok=True)
    # Initialize the tokenizer and model
    model, tokenizer = initialize_model(args)
    model.get_model().initialize_vision_modules(model.get_model().config, add_region_feature=True)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch.bfloat16)
    lora_r = args.lora_r
    if lora_r > 0:
        lora_config = setup_lora_config(model, args)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    # Load the state-dict from --weights
    state_dict = torch.load(args.weight, map_location="cpu")
    updated_state_dict = {}
    for key in state_dict.keys():
        updated_key = f"base_model.model.{key}"
        updated_state_dict[updated_key] = state_dict[key]
    model.load_state_dict(updated_state_dict, strict=True)

    # Merge and save
    model = model.merge_and_unload()
    state_dict = {}
    for k, v in model.state_dict().items():
        if "vision_tower" not in k:
            state_dict[k] = v
    model.save_pretrained(args.save_path, state_dict=state_dict)
    tokenizer.save_pretrained(args.save_path)


if __name__ == "__main__":
    main()