import os
import re
import cv2
import json
import copy
import random
import numpy as np
import torch
from pycocotools import mask as maskUtils
import torch.nn.functional as F
from transformers import CLIPImageProcessor
from mtrag.llava import conversation as conversation_lib
from mtrag.SAM.utils.transforms import ResizeLongestSide
from tools.utils import DEFAULT_IMAGE_TOKEN
from dataset.utils.process import preprocess_multimodal, preprocess_v1
class MTRAGMultiRegDataset(torch.utils.data.Dataset):
    IMG_MEAN = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    IMG_STD = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    IMG_SIZE = 1024
    IGNORE_LABEL = 255
    def __init__(self, dataset_dir, tokenizer, global_image_encoder, precision="fp32",
                 image_size=1024, num_classes_per_sample=3, max_gt_per_img=10, validation=False, dataset_name='',
                 image_dir='', json_path='', random_sampling=True, use_mm_start_end=False, **kwargs):
        self.num_classes_per_sample = num_classes_per_sample
        self.dataset_dir = dataset_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.global_enc_processor = CLIPImageProcessor.from_pretrained(global_image_encoder)
        self.max_gt_per_img = max_gt_per_img
        self.validation = validation
        self.random_sampling = random_sampling
        self.use_mm_start_end = use_mm_start_end
        # Dataset type specific
        self.begin_str = f"{DEFAULT_IMAGE_TOKEN}\nThis provides an overview of the picture.\n"
        self.image_folder = os.path.join(dataset_dir, image_dir)
        self.base_dir = os.path.join(dataset_dir, dataset_name)
        self.ann_file = os.path.join(self.base_dir, json_path)
        self.data_infos = self._load_annotations(self.ann_file)
        self.grounding = kwargs.get("grounding", True)
    def annToMask(self, mask_ann, h, w):
        if isinstance(mask_ann, list):
            rles = maskUtils.frPyObjects(mask_ann, h, w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(mask_ann, h, w)
        else:
            # rle
            rle = mask_ann
        mask = maskUtils.decode(rle)
        return mask

    def _load_annotations(self, ann_file):
        data_infos = json.load(open(ann_file))
        data_infos_filter = [ann for ann in data_infos if len(ann["conversations"]) > 0 and min(ann['width'], ann['height']) >= 32]
        return data_infos_filter
            
    def _parse_annotations(self, ann):
        masks = []
        conversations = dict()
        conversations['conversations'] = []
        filename = ann['file_name']
        img_path = os.path.join(self.image_folder, filename)
        region_num = len(ann['annotation'])
        h, w = ann['height'], ann['width']
        region_string = ''
        for i in range(region_num):
            mask = ann['annotation'][i]['segmentation']
            masks.append(mask)
            region_string += f'region{i + 1} <region_fea>'
            if i < region_num -2:
                region_string += ', '
            elif i == region_num - 2:
                region_string += ' and '
        mid_str = "There are {} part regions in the picture: ".format(str(region_num))+region_string+'. '
        for  i in range(len(ann['conversations'])//2):
            if i==0:
                question = ann['conversations'][i*2]['value']
                question = re.sub(r'<region(\d+)>', lambda match: f'region{match.group(1)}', question)
                question = self.begin_str+mid_str+question
                conversations['conversations'].append({'from': 'human', 'value': question}) 
            else:
                question = ann['conversations'][i*2]['value']
                question = re.sub(r'<region(\d+)>', lambda match: f'region{match.group(1)}', question)
                conversations['conversations'].append({'from': 'human', 'value': question})         
            answer = ann['conversations'][i*2+1]['value']
            answer = re.sub(r'<region(\d+)>', lambda match: f'region{match.group(1)}', answer)
            conversations['conversations'].append({'from': 'gpt', 'value': answer}) 
        return dict(
                image_path = img_path,
                region_masks = masks,
                height = h,
                width = w,
                conversations = conversations
            )
  
    def __getitem__(self, index):
        while True:
            ann = self.data_infos[index]
            data_info = self._parse_annotations(ann)
            if min(data_info['width'], data_info['height']) >= 32 and len(ann["conversations"]) > 0:
                break
            index = random.randint(0, len(self.data_infos) - 1)
        image_path = data_info['image_path']
        height = data_info['height']
        width = data_info['width']
        masks_raw = data_info['region_masks']
        conversations = data_info['conversations']
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Prepare input for Global Image Encoder
        region_masks = []
        for mask_r in masks_raw:
            mask = self.annToMask(mask_r, height, width)
            region_masks.append(mask)
        region_masks = np.array(region_masks)
        global_enc_image, global_enc_image_mask = self.generate_alpha_for_feature(image, region_masks)
        global_enc_image = torch.cat([global_enc_image, global_enc_image_mask], dim=0)
        conv = conversation_lib.default_conversation.copy()
        assert conv.sep_style == conversation_lib.SeparatorStyle.TWO
        sources = preprocess_multimodal(copy.deepcopy([conversations["conversations"]]), use_im_start_end=self.use_mm_start_end)
        input_ids, targets, conversations = preprocess_v1(sources, self.tokenizer)
        assert len(conversations) == 1
        assert conversations[0].count("<region_fea>") == region_masks.shape[0]
        # Prepare input for Grounding Image Encoder
        if self.grounding:
            image = self.transform.apply_image(image)
            image_resize = image.shape[:2]
            grounding_enc_image = self.grounding_enc_processor(torch.from_numpy(image).permute(2, 0, 1).contiguous())
        else:
            image_resize = None
            grounding_enc_image = torch.zeros(3, 1024, 1024)
        masks = None
        label = None
        # image_resize = None
        return (image_path, global_enc_image, grounding_enc_image, region_masks, input_ids[0], targets[0], conversations, masks, label, image_resize)

    def __len__(self):
        return len(self.data_infos)
    
    @property
    def modality_lengths(self):
        length_list = []
        for i in range(len(self.data_infos)):
            ann = self.data_infos[i]
            data_info = self._parse_annotations(ann)
            cur_len = sum(len(conv['value'].split()) for conv in data_info['conversations']['conversations'])
            length_list.append(cur_len)
        return length_list

    def grounding_enc_processor(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.IMG_MEAN) / self.IMG_STD
        h, w = x.shape[-2:]
        x = F.pad(x, (0, self.IMG_SIZE - w, 0, self.IMG_SIZE - h))
        return x

    def generate_alpha_for_feature(self, image, masks):
        image_masks = np.where(masks > 0, 255.0, 0.0)
        rgb_masks = np.repeat(image_masks[:, :, :, np.newaxis], 3, axis=-1).astype(np.uint8)  # (batch, H, W, 3)
        global_enc_image_mask = self.global_enc_processor.preprocess(list(rgb_masks)+[image], return_tensors="pt")["pixel_values"]
        global_enc_image = global_enc_image_mask[-1]
        global_enc_image_mask = global_enc_image_mask[:-1].sum(dim=1, keepdim=True)
        global_enc_image_mask = (global_enc_image_mask > 6) * 1.9231 + (global_enc_image_mask <= 6) * (-1.9231)
        return global_enc_image, global_enc_image_mask.squeeze(1)
    

class MTRAGRelations(MTRAGMultiRegDataset):
    def __init__(self, dataset_dir, tokenizer, global_image_encoder, precision="fp32",
                 image_size=1024, num_classes_per_sample=3, max_gt_per_img=10, validation=False, random_sampling=True, use_mm_start_end=False, **kwargs):
        mode = "Val" if validation else "Train"
        json_path = "mtrag_relationalreasoning.json" 
        dataset_name = "mtrag_instruct"
        image_dir = "coco_2014/train2014"

        super().__init__(
            dataset_dir, tokenizer, global_image_encoder, precision, image_size, num_classes_per_sample,
            max_gt_per_img, validation, dataset_name, image_dir, json_path,
            random_sampling, use_mm_start_end, **kwargs)
        print('\033[92m' + "----MULTIREGION-{}: Loaded MTRAGRelations dataset ----".format(mode) + '\033[0m')

class MTRAGMultiturn(MTRAGMultiRegDataset):
    def __init__(self, dataset_dir, tokenizer, global_image_encoder, precision="fp32",
                 image_size=1024, num_classes_per_sample=3, max_gt_per_img=10, validation=False, random_sampling=True, use_mm_start_end=False, **kwargs):
        mode = "Val" if validation else "Train"
        json_path = "mtrag_multiturn.json"
        dataset_name = "mtrag_instruct"
        image_dir = "coco_2014/train2014"
        super().__init__(
            dataset_dir, tokenizer, global_image_encoder, precision, image_size, num_classes_per_sample,
            max_gt_per_img, validation, dataset_name, image_dir, json_path,
            random_sampling, use_mm_start_end, **kwargs)
        print('\033[92m' + "----MULTIREGION-{}: Loaded MTRAGMultiturn dataset ----".format(mode) + '\033[0m')