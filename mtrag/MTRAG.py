import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union
import torch.nn.functional as F
import math
import time
from mtrag.SAM import build_sam_vit_h
from mtrag.llava.model.language_model.llava_llama import LlavaLlamaForCausalLM, LlavaLlamaModel
from tools.utils import IGNORE_INDEX, IMAGE_TOKEN_INDEX,DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,DEFAULT_REGION_FEA_TOKEN

def calculate_dice_loss(predictions: torch.Tensor, ground_truth: torch.Tensor, mask_count: float, scale_factor=1000, epsilon=1e-6):
    """
    Calculate the DICE loss, a measure similar to generalized IOU for masks.
    """
    predictions = predictions.sigmoid()
    predictions = predictions.flatten(1, 2)
    ground_truth = ground_truth.flatten(1, 2)

    intersection = 2 * (predictions / scale_factor * ground_truth).sum(dim=-1)
    union = (predictions / scale_factor).sum(dim=-1) + (ground_truth / scale_factor).sum(dim=-1)

    dice_loss = 1 - (intersection + epsilon) / (union + epsilon)
    dice_loss = dice_loss.sum() / (mask_count + 1e-8)
    return dice_loss


def compute_sigmoid_cross_entropy(predictions: torch.Tensor, targets: torch.Tensor, mask_count: float):
    """
    Compute sigmoid cross-entropy loss for binary classification.
    """
    targets = targets.clamp(min=0.0, max=1.0)
    loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1)
    loss = loss.sum() / (mask_count + 1e-8)
    return loss


class MTRAGBaseModel:
    def __init__(self, config, **kwargs):
        super(MTRAGBaseModel, self).__init__(config)
        self.config = config
        # Set config attributes if they don't exist
        self.config.train_mask_decoder = getattr(
            self.config, "train_mask_decoder", kwargs.get("train_mask_decoder", False)
        )
        self.config.out_dim = getattr(self.config, "out_dim", kwargs.get("out_dim", 512))

        self.initialize_mtrag_model()

    def initialize_mtrag_model(self):
        # Initialize the visual model
        vision_pretrained = getattr(self.config, "vision_pretrained", None)
        self.grounding_encoder = build_sam_vit_h(vision_pretrained)
        self._configure_grounding_encoder(self.config)

        # Initialize the text projection layer
        self._initialize_text_projection_layer()

    def _configure_grounding_encoder(self, config):
        # Freezing visual model parameters
        for param in self.grounding_encoder.parameters():
            param.requires_grad = False

        # Training mask decoder if specified
        if config.train_mask_decoder:
            self._train_mask_decoder()

    def _train_mask_decoder(self):
        self.grounding_encoder.mask_decoder.train()
        for param in self.grounding_encoder.mask_decoder.parameters():
            param.requires_grad = True

    def _initialize_text_projection_layer(self):
        in_dim, out_dim = self.config.hidden_size, self.config.out_dim
        text_projection_layers = [nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True), nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0), ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_projection_layers)])
        self.text_hidden_fcs.train()
        self.text_hidden_fcs.train()


class MTRAGModel(MTRAGBaseModel, LlavaLlamaModel):
    def __init__(self, config, **kwargs):
        super(MTRAGModel, self).__init__(config, **kwargs)
        self._configure_model_settings(**kwargs)
    def _configure_model_settings(self,**kwargs):
        self.config.use_cache = kwargs.get("use_cache", True)
        self.config.mm_vision_tower = self.config.vision_tower
        self.config.select_feature_type = kwargs.get("select_feature_type", "cls_patch")
        self.config.image_aspect = kwargs.get("image_aspect_ratio", "square")
        self.config.tune_mm_mlp_adapter = kwargs.get("tune_mm_mlp_adapter", False)
        for key, default_value in kwargs.items():
            if not hasattr(self.config, key):
                setattr(self.config, key, default_value) 

class MTRAGForCausalLM(LlavaLlamaForCausalLM):

    def __init__(self, config, **kwargs):
        self._set_model_configurations(config, kwargs)
        super().__init__(config)
        self.model = MTRAGModel(config, **kwargs)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def encode_images(self, images, grounding_images):
        B, C, H, W = images.shape # c = 3 + NUM_MASK
        image_features, image_features_lastlayer= self.get_model().get_vision_tower()(images)
        use_samclip = getattr(self.config, 'mm_projector_type', 'mlp2x_gelu') == "SAMCLIP"
        if image_features.shape[0] == B and C == 3:
            raw_image_features = image_features[:,1:,:]
        else:
            image_features = image_features.reshape(B,C-2, image_features.shape[1], image_features.shape[2]).to(images.device)
            image_features_lastlayer = image_features_lastlayer.reshape(B,C-2, image_features_lastlayer.shape[1], image_features_lastlayer.shape[2]).to(images.device)
            raw_image_features = image_features[:, 0, 1:, :]
            # image_features
        if use_samclip:
            grounding_features = self.get_grounding_encoder_embs(grounding_images)
            projected_image_features = self.get_model().mm_projector(raw_image_features, grounding_features)
        else:
            projected_image_features = self.get_model().mm_projector(raw_image_features)
        return raw_image_features, projected_image_features, image_features_lastlayer

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, images, grounding_enc_images, region_masks
    ):
        """
        input_ids: [B, seq_len]
        attention_mask: [B, seq_len]
        labels: [B, seq_len]
        region_masks: [B, num_region_mask, (ori_w, orin_h)]
        images: [B,C,H,W]
        """
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if (
                past_key_values is not None
                and vision_tower is not None
                and images is not None
                and input_ids.shape[1] == 1
            ):
                attention_mask = torch.ones(
                    (attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
            return input_ids, attention_mask, past_key_values, None, labels
        
        if type(images) is list or images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            if type(grounding_enc_images) is list or grounding_enc_images.ndim == 5:
                concat_grounding_enc_images = torch.cat([image for image in grounding_enc_images], dim=0)
            raw_image_features, image_features, image_features_mask = self.encode_images(concat_images,  concat_grounding_enc_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1) for x in image_features]
        else:
            # Process for region
            raw_image_features, image_features, image_features_mask = self.encode_images(images, grounding_enc_images)
        if region_masks is not None and (len(region_masks) > 0) and not all(x is None for x in region_masks):
            mlvl_reg_features = self.extract_region_feature(image_features_mask, region_masks)
        else:
            mlvl_reg_features = [None for _ in range(len(input_ids))]
        _labels = labels
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]
        
        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0
        for batch_idx, (cur_input_ids, reg_feat) in enumerate(zip(input_ids, mlvl_reg_features)): # Adjusted the loop to include reg_feat
            curr_full_input_ids = []
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                cur_input_embeds = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = (
                    cur_input_embeds
                    + (
                        0.0 * image_features[cur_image_idx] # vision_tower.dummy_feature
                    ).sum()
                )
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape
            while image_token_indices.numel() > 0: 
                cur_image_features = image_features[cur_image_idx]  
                image_token_start = image_token_indices[0]
                if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(self.config, "mm_use_im_start_end", False):
                    # preparing input embedding
                    cur_new_input_embeds.append(
                        self.get_model()
                        .embed_tokens(cur_input_ids[: image_token_start - 1])
                        .detach()
                    )
                    # <img_st>
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(
                            cur_input_ids[image_token_start - 1 : image_token_start]
                        )
                    )
                    # <image>
                    cur_new_input_embeds.append(cur_image_features)
                    # <img_end>
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(
                            cur_input_ids[image_token_start + 1 : image_token_start + 2]
                        )
                    )
                    # preparing input_ids
                    curr_full_input_ids.append(cur_input_ids[: image_token_start - 1])
                    curr_full_input_ids.append(cur_input_ids[image_token_start - 1: image_token_start])
                    curr_full_image_token = torch.full((cur_image_features.shape[0],), image_token_start, dtype=torch.int64) 
                    curr_full_input_ids.append(curr_full_image_token)
                    curr_full_input_ids.append(cur_input_ids[image_token_start + 1: image_token_start + 2])
                    # preparing labels
                    if labels is not None:
                        # labels delay 1
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(
                            torch.full(
                                (cur_image_features.shape[0],),
                                IGNORE_INDEX,
                                device=labels[0].device,
                                dtype=labels[0].dtype,
                            )
                        )
                        cur_new_labels.append(
                            cur_labels[image_token_start : image_token_start + 1]
                        )
                        cur_labels = cur_labels[image_token_start + 2 :]
                elif getattr(self.config, "mm_use_im_start_end", False):
                    # preparing input embedding
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids[:image_token_start])
                    )
                    cur_new_input_embeds.append(cur_image_features)
                    # until image_token_start + 2, The remainder may have an <image>id
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(
                            cur_input_ids[image_token_start + 1 : image_token_start + 2]
                        )
                    )
                    # preparing input_ids
                    curr_full_input_ids.append(cur_input_ids[: image_token_start])
                    curr_full_image_token = torch.full((cur_image_features.shape[0],), image_token_start,
                                                       dtype=torch.int64)
                    curr_full_input_ids.append(curr_full_image_token)
                    # until image_token_start + 2, The remainder may have an <image>id
                    curr_full_input_ids.append(cur_input_ids[image_token_start + 1: image_token_start + 2]) 
                    # preparing labels
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(
                            torch.full(
                                (cur_image_features.shape[0],),
                                IGNORE_INDEX,
                                device=labels[0].device,
                                dtype=labels[0].dtype,
                            )
                        )
                        cur_new_labels.append(
                            cur_labels[image_token_start + 1 : image_token_start + 2]
                        )
                        cur_labels = cur_labels[image_token_start + 2 :] 
                else:
                    # preparing input embedding
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids[:image_token_start])
                    )
                    cur_new_input_embeds.append(cur_image_features)
                    # preparing input_ids
                    curr_full_input_ids.append(cur_input_ids[:image_token_start])
                    curr_full_image_token = torch.full((cur_image_features.shape[0],), image_token_start,
                                                       dtype=torch.int64)
                    curr_full_input_ids.append(curr_full_image_token)
                    # preparing labels
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(
                            torch.full(
                                (cur_image_features.shape[0],),
                                IGNORE_INDEX,
                                device=labels[0].device,
                                dtype=labels[0].dtype,
                            )
                        )
                        cur_labels = cur_labels[image_token_start + 1 :]

                cur_image_idx += 1
                if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(
                    self.config, "mm_use_im_start_end", False
                ):
                    cur_input_ids = cur_input_ids[image_token_start + 2 :]
                elif getattr(self.config, "mm_use_im_start_end", False):
                    cur_input_ids = cur_input_ids[image_token_start + 2 :]
                else:
                    cur_input_ids = cur_input_ids[image_token_start + 1 :]
                image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            
            if cur_input_ids.numel() > 0:
                if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(
                    self.config, "mm_use_im_start_end", False
                ):
                    if reg_feat is not None:
                        REG_TOKEN_ID = self.config.reg_token_idx
                        mask_idx = torch.nonzero(cur_input_ids==REG_TOKEN_ID)
                        assert len(mask_idx) == len(reg_feat), "mask num not equal to mask feats"
                        _l = 0
                        for i, idx in enumerate(mask_idx):
                            cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[_l:idx[0]]).detach())
                            ## mask
                            cur_new_input_embeds.append(reg_feat[i:i+1])
                            if labels is not None:
                                cur_labels[idx[0]:idx[0]+1] = torch.full((1,), IGNORE_INDEX, device=labels[0].device, dtype=labels[0].dtype)
                            _l = idx[0]+1
                        if _l< len(cur_input_ids):
                            cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[_l:]).detach())
                    else:
                        dummy_region_features = self.get_model().region_fea_adapter(torch.zeros(image_features_mask.shape[-2], image_features_mask.shape[-1], device=image_features_mask.device, dtype=image_features_mask.dtype).detach())
                        cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids).detach()+ (
                        0.0*dummy_region_features 
                        ).sum()
                        )
                else:
                    if reg_feat is not None:
                        REG_TOKEN_ID = self.config.reg_token_idx
                        mask_idx = torch.nonzero(cur_input_ids==REG_TOKEN_ID)
                        assert len(mask_idx) == len(reg_feat), "mask num not equal to mask feats"
                        _l = 0
                        for i, idx in enumerate(mask_idx):
                            cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[_l:idx[0]]))
                            ## mask
                            cur_new_input_embeds.append(reg_feat[i:i+1])
                            if labels is not None:
                                cur_labels[idx[0]:idx[0]+1] = torch.full((1,), IGNORE_INDEX, device=labels[0].device, dtype=labels[0].dtype)
                            _l = idx[0]+1
                        if _l< len(cur_input_ids):
                            cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[_l:])) 
                    else:
                        dummy_region_features = self.get_model().region_fea_adapter(torch.zeros(image_features_mask.shape[-2], image_features_mask.shape[-1], device=image_features_mask.device, dtype=image_features_mask.dtype).detach())
                        cur_new_input_embeds.append(
                            self.get_model().embed_tokens(cur_input_ids)+ (0.0*dummy_region_features
                        ).sum()
                        )
                curr_full_input_ids.append(cur_input_ids)
                if labels is not None:
                    cur_new_labels.append(cur_labels)


            cur_new_input_embeds = [
                x.to(device=self.device) for x in cur_new_input_embeds
            ]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            curr_full_input_ids = [x.to(device=self.device) for x in curr_full_input_ids]
            curr_full_input_ids = torch.cat(curr_full_input_ids, dim=0)
           
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)
        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]
        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)
        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True


        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        return None, attention_mask, past_key_values, new_input_embeds, new_labels
    def _set_model_configurations(self, config, kwargs):
        valid_keys = [
            "mm_use_image_start_end",
            "vision_module",
            "vision_tower",
            "vision_tower_alpha",
            "num_level_reg_features",
            "with_region",
            "pretrain_mm_mlp_adapter",
            "pretrain_region_fea_adapter",
            "add_region_feature",
            "mm_projector_type",
            "train_mask_decoder",
            "vision_pretrained",
        ]
        default_configurations = {
            k: kwargs[k] for k in valid_keys if k in kwargs
        }
        for key, default_value in default_configurations.items():
            setattr(config, key, default_value)
        self._initialize_loss_weights(kwargs)


    def _initialize_loss_weights(self, kwargs):
        self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
        self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
        self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)

    def get_grounding_encoder_embs(self, pixel_values: torch.FloatTensor):
        with torch.no_grad():
            outputs = []
            batch_size = len(pixel_values)//2
            if batch_size == 0:
                batch_size = 1
            for i in range(0, pixel_values.shape[0], batch_size):
                batch = pixel_values[i:i + batch_size].to(device=pixel_values.device, dtype=pixel_values.dtype)
                outputs.append(self.model.grounding_encoder.image_encoder(batch)) 
                # del output
            final_output = torch.cat(outputs, dim=0)
            return final_output
    def forward(self, 
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                images: Optional[torch.FloatTensor] = None,
                grounding_enc_images: Optional[torch.FloatTensor] = None,
                region_masks: Optional[List[torch.FloatTensor]] = None,
                **kwargs,):
        return super().forward(images=images, attention_mask=attention_mask, input_ids=input_ids,inputs_embeds=inputs_embeds, labels=labels,
            region_masks=region_masks, grounding_enc_images=grounding_enc_images, **kwargs) if "past_key_values" in kwargs else self.model_forward(images=images, attention_mask=attention_mask, input_ids=input_ids,inputs_embeds=inputs_embeds, labels=labels,
            region_masks=region_masks, grounding_enc_images=grounding_enc_images,**kwargs)

    def model_forward(self, global_enc_images: torch.FloatTensor, grounding_enc_images: torch.FloatTensor,
                      region_masks: List[torch.FloatTensor], input_ids: torch.LongTensor, labels: torch.LongTensor,
                      attention_masks: torch.LongTensor, offset: torch.LongTensor, masks_list: List[torch.FloatTensor],
                      label_list: List[torch.Tensor], resize_list: List[tuple], inference: bool = False, **kwargs, ):

        # Handle inference or training paths
        if inference:
            output_hidden_states = self._inference_path(input_ids, global_enc_images, grounding_enc_images, attention_masks)
        else:
            output, output_hidden_states = self._training_path(
                global_enc_images, grounding_enc_images,region_masks, input_ids, labels, attention_masks, offset
            )
        if not self.config.train_mask_decoder:
            pred_masks = None
            return self._calculate_losses(pred_masks, masks_list, output)
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        new_input_ids = input_ids[:, :tokenizer_model_max_length-575]
        if (new_input_ids[:, 1:] == self.config.seg_token_idx).any() and (new_input_ids[:, 1:] == IMAGE_TOKEN_INDEX).any():
            # Extract grounding encoder image embeddings
            image_embeddings = self.get_grounding_encoder_embs(grounding_enc_images)
            assert image_embeddings.shape[0] == len(offset) - 1
            # Create segmentation token mask
            seg_token_mask = self._create_seg_token_mask(new_input_ids)
            # Process hidden states
            hidden_states, pred_embeddings = self._process_hidden_states(output_hidden_states, seg_token_mask, offset)
            # Generate and post-process masks
             
            pred_masks = self._generate_and_postprocess_masks(
                pred_embeddings, image_embeddings, resize_list, label_list
            )

            if inference:
                return {"pred_masks": pred_masks, "gt_masks": masks_list, }
        else:
            pred_masks = None

        return self._calculate_losses(pred_masks, masks_list, output)

    def _create_seg_token_mask(self, input_ids):
        mask = input_ids[:, 1:] == self.config.seg_token_idx
        return torch.cat(
            [torch.zeros((mask.shape[0], 575)).bool().cuda(), mask, torch.zeros((mask.shape[0], 1)).bool().cuda()],
            dim=1
        ) 

    def _inference_path(self, input_ids, global_enc_images, grounding_enc_images, attention_masks):
        length = input_ids.shape[0]
        global_enc_images_extended = global_enc_images.expand(length, -1, -1, -1).contiguous()
        grounding_enc_images_extended = grounding_enc_images.expand(length, -1, -1, -1).contiguous()
        # Process and return inference output
        output_hidden_states = []
        for i in range(input_ids.shape[0]):
            output_i = super().forward(
                images=global_enc_images_extended[i:i + 1], attention_mask=attention_masks[i:i + 1],
                input_ids=input_ids[i:i + 1], output_hidden_states=True, grounding_enc_images=grounding_enc_images_extended[i:i + 1])
            output_hidden_states.append(output_i.hidden_states)
            torch.cuda.empty_cache()
        output_hidden_states = torch.cat(output_hidden_states, dim=0)
        output_hidden_states = [output_hidden_states]
        return output_hidden_states

    def _training_path(self, global_enc_images, grounding_enc_images, region_masks, input_ids, labels, attention_masks, offset):
        global_enc_images, grounding_enc_images = self._prepare_enc_image(global_enc_images, grounding_enc_images, offset)
        output = super().forward(
            images=global_enc_images, attention_mask=attention_masks, input_ids=input_ids, labels=labels,
            output_hidden_states=True, region_masks=region_masks, grounding_enc_images=grounding_enc_images)
        output_hidden_states = output.hidden_states
        return output, output_hidden_states

    def _prepare_enc_image(self, global_enc_image, grounding_enc_image, offset):
        global_enc_image_list = []
        grounding_enc_image_list = []
        for i in range(len(offset) - 1):
            start_i, end_i = offset[i], offset[i + 1]
            # A picture can appear in multiple conversations, extending it to multiple groups
            global_enc_image_i = global_enc_image[i].unsqueeze(0).expand(end_i - start_i, -1, -1, -1).contiguous()
            global_enc_image_list.append(global_enc_image_i)
            grounding_enc_image_i = grounding_enc_image[i].unsqueeze(0).expand(end_i - start_i, -1, -1, -1).contiguous()
            grounding_enc_image_list.append(grounding_enc_image_i)
        return torch.cat(global_enc_image_list, dim=0), torch.cat(grounding_enc_image_list, dim=0)    # [B*num_image, C, H, W]

    def _process_hidden_states(self, output_hidden_states, seg_token_mask, offset, infer=False):
        hidden_states = [self.model.text_hidden_fcs[0](output_hidden_states[-1])] 
        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
        pred_embeddings = last_hidden_state[seg_token_mask] # [num_seg, dim]
        seg_token_counts = seg_token_mask.int().sum(-1)

        seg_token_offset = seg_token_counts.cumsum(-1)  #  Each sample of dialog contains a different number of segmentation tokens
        seg_token_offset = torch.cat([torch.zeros(1).long().cuda(), seg_token_offset], dim=0)
        if not infer:
            seg_token_offset = seg_token_offset[offset]
        pred_embeddings_list = []
        for i in range(len(seg_token_offset) - 1):
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            pred_embeddings_list.append(pred_embeddings[start_i:end_i])
        return hidden_states, pred_embeddings_list

    def _generate_and_postprocess_masks(self, pred_embeddings, image_embeddings, resize_list, label_list, infer=False):
        pred_masks = []
        for i, pred_embedding in enumerate(pred_embeddings):
            if len(pred_embedding) == 0:
                pred_masks.append([])
                continue
            sparse_embeddings, dense_embeddings = self.model.grounding_encoder.prompt_encoder(
                points=None, boxes=None, masks=None, text_embeds=pred_embedding.unsqueeze(1)
            )
            sparse_embeddings = sparse_embeddings.to(pred_embedding.dtype)
            low_res_masks, _ = self.model.grounding_encoder.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=self.model.grounding_encoder.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings, dense_prompt_embeddings=dense_embeddings,
                multimask_output=False, )
            orig_size = label_list[i].shape if not infer else label_list[i]
            # During inference, we have original size list in place of label list
            pred_mask = self.model.grounding_encoder.postprocess_masks(
                low_res_masks, input_size=resize_list[i], original_size=orig_size, )
            pred_masks.append(pred_mask[:, 0])
        return pred_masks

    def _calculate_losses(self, pred_masks, masks_list, output):
        loss_components = self._compute_loss_components(pred_masks, masks_list, output)
        return loss_components

    def _compute_loss_components(self, pred_masks, masks_list, output):
        # Initialize loss components
        ce_loss = output.loss * self.ce_loss_weight
        mask_bce_loss = torch.tensor(0.0, device=ce_loss.device)
        mask_dice_loss = torch.tensor(0.0, device=ce_loss.device)
        num_masks = 0
        if pred_masks:
            # Iterate over batch and compute mask-related losses
            for batch_idx, pred_mask in enumerate(pred_masks):
                if len(pred_mask) == 0:
                    continue
                if pred_mask.numel() > 0:  # Ensure pred_mask is not empty
                    gt_mask = masks_list[batch_idx]
                    assert gt_mask.shape[0] > 0, f"Invalid mask_count: {gt_mask.shape[0]}"
                    # Resize gt_mask to match pred_mask if needed
                    if gt_mask.shape[0] != pred_mask.shape[0]:
                        gt_mask = gt_mask[:pred_mask.shape[0]]

                    assert gt_mask.shape[0] == pred_mask.shape[
                        0], f"Shape mismatch: gt_mask {gt_mask.shape}, pred_mask {pred_mask.shape}"
                    # Compute Binary Cross-Entropy Loss
                    mask_bce_loss += (compute_sigmoid_cross_entropy(pred_mask, gt_mask, mask_count=gt_mask.shape[0]) *
                                      gt_mask.shape[0])
                    # Compute Dice Loss
                    mask_dice_loss += (
                            calculate_dice_loss(pred_mask, gt_mask, mask_count=gt_mask.shape[0]) * gt_mask.shape[0])
                    num_masks += gt_mask.shape[0]

        # Normalize the losses
        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
        mask_loss = mask_bce_loss + mask_dice_loss

        # Aggregate all loss components
        total_loss = ce_loss + mask_loss

        return {"loss": total_loss, "ce_loss": ce_loss, "mask_bce_loss": mask_bce_loss,
                "mask_dice_loss": mask_dice_loss, "mask_loss": mask_loss, "logits":output["logits"]}


    def evaluate(self, global_enc_images, grounding_enc_images, input_ids, resize_list, label_list, max_tokens_new=32,
                 region_masks=None, **kwargs): # For evaluation Seg
        with torch.no_grad():
            generation_outputs = self.generate(
                images=global_enc_images, grounding_enc_images=grounding_enc_images, input_ids=input_ids, region_masks=region_masks, max_new_tokens=max_tokens_new, num_beams=1, output_hidden_states=True, return_dict_in_generate=True,  **kwargs)
            output_hiddens = generation_outputs.hidden_states[1:]  # tupe of hidden states
            output_hidden_states = [generation_outputs.hidden_states[0][:, -1:, :]]
            for i in range(len(output_hiddens)):
                output_hidden_states.append(output_hiddens[i])
            output_hidden_states = torch.cat(output_hidden_states, dim=1)
            output_hidden_states = [output_hidden_states]
            generated_output_ids = generation_outputs.sequences[:,1:]
            if getattr(self.config, "seg_token_idx", None):
                if (generated_output_ids == self.config.seg_token_idx).any():
                    seg_token_mask = generated_output_ids == self.config.seg_token_idx
                    # Process hidden states
                    hidden_states, predicted_embeddings = self._process_hidden_states(
                        output_hidden_states, seg_token_mask, None, infer=True
                    )
                    image_embeddings = self.get_grounding_encoder_embs(grounding_enc_images)
                    # Generate and post-process masks
                    pred_masks = self._generate_and_postprocess_masks(
                        predicted_embeddings, image_embeddings, resize_list, label_list, infer=True
                    )
                else:
                    pred_masks = None
            else:
                pred_masks = None
        return generated_output_ids, pred_masks
