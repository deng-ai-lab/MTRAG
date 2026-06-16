import os
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from tools.utils import IGNORE_INDEX, IMAGE_TOKEN_INDEX,DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN
from mtrag.llava.model.multimodal_encoder.builder import build_vision_tower
from mtrag.llava.model.multimodal_projector.builder import  build_vision_projector
import types
import wget
import collections
import torch.nn.functional as F
def rewrited_forward_embedding(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
    patch_embeds = self.patch_embedding(pixel_values[:,:3,:,:])  # shape = [B, width, grid, grid]
    _, dim, grid, _ = patch_embeds.shape
    if pixel_values.shape[1] >= 4:
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




class LlavaMetaModel:
    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)
        if hasattr(config, "mm_vision_tower"): 
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)
        if hasattr(config, "region_fea_adapter"): 
            modules = [nn.Linear(self.config.mm_hidden_size, self.config.hidden_size),
                           nn.GELU(),
                           nn.Linear(self.config.hidden_size, self.config.hidden_size)]
            self.region_fea_adapter = nn.Sequential(*modules)
            
    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, add_region_feature=False, fsdp=None):
        vision_tower = model_args.vision_tower
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        pretrain_region_fea_adapter = model_args.pretrain_region_fea_adapter
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        grounding_hidden_size = model_args.grounding_hidden_size
        mm_patch_merge_type = model_args.mm_patch_merge_type
        self.config.mm_vision_tower = vision_tower
        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)
            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()
        
        visual_encoder = vision_tower.vision_tower.vision_model
        visual_encoder.embeddings.patch_embedding_alpha = torch.nn.Conv2d(in_channels=1,
                                                        out_channels=visual_encoder.embeddings.patch_embedding.out_channels, 
                                                        kernel_size=visual_encoder.embeddings.patch_embedding.kernel_size, 
                                                        stride=visual_encoder.embeddings.patch_embedding.stride, 
                                                        bias=False)
        visual_encoder.embeddings.forward = types.MethodType(rewrited_forward_embedding, visual_encoder.embeddings)
        # visual_encoder.forward = types.MethodType(rewrited_forward_vision_model, visual_encoder)
        filename = getattr(model_args, "vision_tower_alpha", "clip_l14@336_grit_20m_fultune_4xe.pth")
         
        if not os.path.exists(filename):
            filename = wget.download("https://download.openxlab.org.cn/models/SunzeY/AlphaCLIP/weight//clip_l14_336_grit20m_fultune_4xe.pth")
        state_dict = torch.load(filename,  map_location='cpu')
        converted_dict = collections.OrderedDict()
        for k, v in state_dict.items():
            if 'transformer.resblocks' in k:
                new_key = k.replace('transformer.resblocks', 'encoder.layers').replace('attn', 'self_attn').replace('ln_1', 'layer_norm1').replace('ln_2', 'layer_norm2') \
                            .replace('c_fc', 'fc1').replace('c_proj', 'fc2')
                if ('self_attn' in new_key) and ('out' not in new_key): # split qkv attn
                    if 'weight' in new_key :
                        converted_dict[new_key.replace('in_proj', 'q_proj')] = v[:1024, :]
                        converted_dict[new_key.replace('in_proj', 'k_proj')] = v[1024:2048, :]
                        converted_dict[new_key.replace('in_proj', 'v_proj')] = v[2048:, :]
                    else:
                        assert 'bias' in new_key
                        converted_dict[new_key.replace('in_proj', 'q_proj')] = v[:1024]
                        converted_dict[new_key.replace('in_proj', 'k_proj')] = v[1024:2048]
                        converted_dict[new_key.replace('in_proj', 'v_proj')] = v[2048:]
                else:
                    converted_dict[new_key] = v
            else:
                new_key = k.replace('class_embedding', 'embeddings.class_embedding') \
                            .replace('conv1.weight', 'embeddings.patch_embedding.weight') \
                            .replace('positional_embedding', 'embeddings.position_embedding.weight') \
                            .replace('conv1_alpha.weight', 'embeddings.patch_embedding_alpha.weight') \
                            .replace('ln_pre.weight', 'pre_layrnorm.weight') \
                            .replace('ln_pre.bias', 'pre_layrnorm.bias') \
                            .replace('ln_post.weight', 'post_layernorm.weight') \
                            .replace('ln_post.bias', 'post_layernorm.bias')
                converted_dict[new_key] = v

        visual_encoder.load_state_dict(converted_dict, strict=False)
        self.config.use_mm_proj = True
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.grounding_hidden_size = grounding_hidden_size
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)
            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True


        # initialize the region feature adapter
        if add_region_feature:
            if not hasattr(self, 'region_fea_adapter'):
                modules = [nn.Linear(self.config.mm_hidden_size, self.config.hidden_size),
                           nn.GELU(),
                           nn.Linear(self.config.hidden_size, self.config.hidden_size)]
                self.region_fea_adapter = nn.Sequential(*modules)
            else:
                # In case it is frozen by LoRA
                for p in self.region_fea_adapter.parameters():
                    p.requires_grad = True

        if pretrain_region_fea_adapter is not None:
            region_fea_adapter_weights = torch.load(
                pretrain_region_fea_adapter, map_location="cpu"
            )

            def get_w(weights, keyword):
                return {
                    k.split(keyword + ".")[1]: v
                    for k, v in weights.items()
                    if keyword in k
                }

            self.region_fea_adapter.load_state_dict(
                get_w(region_fea_adapter_weights, "region_fea_adapter")
            )

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(
                pretrain_mm_mlp_adapter, map_location="cpu"
            )

            def get_w(weights, keyword):
                return {
                    k.split(keyword + ".")[1]: v
                    for k, v in weights.items()
                    if keyword in k
                }

            self.mm_projector.load_state_dict(
                get_w(mm_projector_weights, "mm_projector")
            )

class LlavaMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        pass

    
    def apply_merge(self, x, pooler_mode='mean', dim=2):
        if pooler_mode == 'mean':
            return x.mean(dim=dim)
        elif pooler_mode == 'max':
            return x.max(dim=dim).values
        else:
            raise NotImplementedError
    
    def extract_region_feature(self, image_features_mask, region_masks):
        # image_features_mask: [B, cls_patch, mm_hidden_size] or [B, mask+1, cls_patch, mm_hidden_size]
        assert len(image_features_mask) == len(region_masks)
        all_region_features = [None] * len(region_masks)
        if len(image_features_mask.shape) == 3:
            return all_region_features
        region_features = self.get_model().region_fea_adapter(image_features_mask[:, :, 0, :])
        for i, (region_feature, region_mask) in enumerate(zip(region_features, region_masks)):
            if region_mask is not None and region_mask.numel() > 0:
                all_region_features[i] = region_feature[1 : len(region_mask) + 1, :] 
        return all_region_features
    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, images, grounding_enc_images, region_masks
    ):
        pass
    

    def initialize_vision_tokenizer(self, model_args, tokenizer, add_region_feature=False):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))
        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

            self.resize_token_embeddings(len(tokenizer))
            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
