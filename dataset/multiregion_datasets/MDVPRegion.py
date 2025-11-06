import os
import cv2
import re
import copy
import random
import numpy as np
import torch
import json
import torch.nn.functional as F
from transformers import CLIPImageProcessor
from mtrag.llava import conversation as conversation_lib
from mtrag.SAM.utils.transforms import ResizeLongestSide
from tools.utils import DEFAULT_IMAGE_TOKEN
from dataset.utils.utils import REGION_SEPARATE_QUESTIONS, HUMANIZED_INTRODUCTIONS
from dataset.utils.process import preprocess_multimodal, preprocess_v1
class PointRegDataset(torch.utils.data.Dataset):
    IMG_MEAN = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    IMG_STD = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    IMG_SIZE = 1024
    IGNORE_LABEL = 255
    def __init__(self, dataset_dir, tokenizer, global_image_encoder, precision="fp32",
                 image_size=1024, num_classes_per_sample=3, max_gt_per_img=10, validation=False, dataset_name='',
                 image_dir='', json_path='', intro_string='', random_sampling=True, use_mm_start_end=False, **kwargs):
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
        self.begin_str = intro_string
        self.base_dir = os.path.join(dataset_dir, dataset_name)
        self.ann_file = os.path.join(self.base_dir, json_path)
        self.image_folder = os.path.join(dataset_dir, image_dir)
        self.data_infos = self._load_annotations(self.ann_file)
        self.data_infos = [ann for ann in self.data_infos if not len(ann['conversations'])//2 ==0 and len(ann['points']) > 0] 
        self.grounding = kwargs.get("grounding", True)
    def _generate_mask(self, shape, point):
        x, y = point
        span = 10
        x1 = max(0, x - span)
        x2 = min(shape[1], x + span + 1)
        y1 = max(0, y - span)
        y2 = min(shape[0], y + span + 1)
        mask = np.zeros(shape, dtype=np.uint8)
        mask[int(y1):int(y2), int(x1):int(x2)] = 1
        return mask
    def _load_annotations(self, ann_file):
        data_infos = json.load(open(ann_file))
        return data_infos
    def _parse_annotations(self, ann):
        conversations = dict()
        conversations['conversations'] = []
        filename = ann['image'].split("/")[-1]
        image_path = os.path.join(self.image_folder, filename)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        region_num = len(ann['points'])
        h, w = image.shape[:2]
        region_string = ''
        masks = []
        for i in range(region_num):
            mask = self._generate_mask(shape=(h, w), point=ann['points'][i])
            masks.append(mask)
            region_string += f'region{i + 1} <region_fea>'
            if i < region_num -2:
                region_string += ', '
            elif i == region_num - 2:
                region_string += ' and '
        question = random.choice(REGION_SEPARATE_QUESTIONS).strip().replace('<region>', region_string)
        answer = random.choice(HUMANIZED_INTRODUCTIONS).strip() + "\n"
        gpt_conversation = next(entry for entry in ann['conversations'] if entry['from'] == 'gpt')['value']
        gpt_conversation = re.sub(r'<Mark (\d+)>', lambda match: f'Mark {match.group(1)}', gpt_conversation)
        gpt_conversation = gpt_conversation.replace("Mark","region")

        conversations['conversations'].append({'from': 'human', 'value': self.begin_str + question})
        conversations['conversations'].append({'from': 'gpt', 'value': answer + "\n" + gpt_conversation})
        return dict(
            image = image,
            image_path = image_path,
            region_masks = masks,
            height = h,
            width = w,
            conversations = conversations
        )
    def __getitem__(self, index):
        while True:
            ann = self.data_infos[index]
            data_info = self._parse_annotations(ann)
            if min(data_info['width'], data_info['height']) >= 32:
                break
            index = random.randint(0, len(self.data_infos) - 1)
        image_path = data_info['image_path']
        region_masks = data_info['region_masks']
        conversations = data_info['conversations']
        image = data_info['image']
        # Prepare Conversations
        conv = conversation_lib.default_conversation.copy()
        assert conv.sep_style == conversation_lib.SeparatorStyle.TWO
        sources = preprocess_multimodal(copy.deepcopy([conversations["conversations"]]), use_im_start_end=self.use_mm_start_end)
        input_ids, targets, conversations = preprocess_v1(sources, self.tokenizer)
        assert len(conversations) == 1
        # Prepare alpha for Global Image Encoder
        region_masks = np.array(region_masks)
        global_enc_image, global_enc_image_mask = self.generate_alpha_for_feature(image, region_masks)
        global_enc_image = torch.cat([global_enc_image, global_enc_image_mask], dim=0)
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
        return (image_path, global_enc_image, grounding_enc_image, region_masks, input_ids[0], targets[0], conversations, masks, label, image_resize)

    def __len__(self):
        return len(self.data_infos)
    
    @property
    def modality_lengths(self):
        length_list = []
        for i in range(len(self.data_infos)):
            ann = self.data_infos[i]
            cur_len = sum(len(conv['value'].split()) for conv in ann['conversations'])
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

class ADE20KPointDetailedCap(PointRegDataset):
    def __init__(self, dataset_dir, tokenizer, global_image_encoder, precision="fp32",
                 image_size=1024, num_classes_per_sample=3, max_gt_per_img=20, validation=False, random_sampling=True, use_mm_start_end=False, **kwargs):
        intro_string = f"{DEFAULT_IMAGE_TOKEN}\nThis provides an overview of the picture.\n"
        json_path = f"gpt4v_ade20k_detailed_caption_point.json"
        dataset_name = "mdvp_instruct/detailed_caption"
        image_dir = f"ade20k/images/training"
        super().__init__(
            dataset_dir, tokenizer, global_image_encoder, precision, image_size, num_classes_per_sample,
            max_gt_per_img, validation, dataset_name, image_dir, json_path,
            intro_string, random_sampling,  use_mm_start_end, **kwargs)
        
class VGPointDetailedCap(PointRegDataset):
    def __init__(self, dataset_dir, tokenizer, global_image_encoder, precision="fp32",
                 image_size=1024, num_classes_per_sample=3, max_gt_per_img=20, validation=False, random_sampling=True, use_mm_start_end=False, **kwargs):
        intro_string = f"{DEFAULT_IMAGE_TOKEN}\nThis provides an overview of the picture.\n"
        json_path = f"gpt4v_vg_detailed_caption_point.json"
        dataset_name = "mdvp_instruct/detailed_caption"
        image_dir = f"vg"
        super().__init__(
            dataset_dir, tokenizer, global_image_encoder, precision, image_size, num_classes_per_sample,
            max_gt_per_img, validation, dataset_name, image_dir, json_path,
            intro_string, random_sampling,  use_mm_start_end, **kwargs)
    def _parse_annotations(self, ann):
        conversations = dict()
        conversations['conversations'] = []
        filename = ann['image'].split("/")[-1]
        subdirs = ['VG_100K', 'VG_100K_2']
        image_path = os.path.join(self.image_folder, subdirs[0], filename)
        if not os.path.exists(image_path):
            image_path = os.path.join(self.image_folder, subdirs[1], filename)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        region_num = len(ann['points'])
        h, w = image.shape[:2]
        region_string = ''
        masks = []
        for i in range(region_num):
            mask = self._generate_mask(shape=(h, w), point=ann['points'][i])
            masks.append(mask)
            region_string += f'region{i + 1} <region_fea>'
            if i < region_num -2:
                region_string += ', '
            elif i == region_num - 2:
                region_string += ' and '
        question = random.choice(REGION_SEPARATE_QUESTIONS).strip().replace('<region>', region_string)
        answer = random.choice(HUMANIZED_INTRODUCTIONS).strip() + "\n"
        gpt_conversation = next(entry for entry in ann['conversations'] if entry['from'] == 'gpt')['value']
        gpt_conversation = re.sub(r'<Mark (\d+)>', lambda match: f'Mark {match.group(1)}', gpt_conversation)
        gpt_conversation = gpt_conversation.replace("Mark","region")
        conversations['conversations'].append({'from': 'human', 'value': self.begin_str + question})
        conversations['conversations'].append({'from': 'gpt', 'value': answer + "\n" + gpt_conversation})
        return dict(
            image = image,
            image_path = image_path,
            region_masks = masks,
            height = h,
            width = w,
            conversations = conversations
        )
    
class LVISPointDetailedCap(PointRegDataset):
    def __init__(self, dataset_dir, tokenizer, global_image_encoder, precision="fp32",
                 image_size=1024, num_classes_per_sample=3, max_gt_per_img=20, validation=False, random_sampling=True, use_mm_start_end=False, **kwargs):
        intro_string = f"{DEFAULT_IMAGE_TOKEN}\nThis provides an overview of the picture.\n"
        json_path = f"gpt4v_lvis_detailed_caption_point.json"
        dataset_name = "mdvp_instruct/detailed_caption"
        image_dir = f"coco/train2017"
        super().__init__(
            dataset_dir, tokenizer, global_image_encoder, precision, image_size, num_classes_per_sample,
            max_gt_per_img, validation, dataset_name, image_dir, json_path,
            intro_string, random_sampling,  use_mm_start_end, **kwargs)
        
class COCOStuff10kPointDetailedCap(PointRegDataset):
    def __init__(self, dataset_dir, tokenizer, global_image_encoder, precision="fp32",
                 image_size=1024, num_classes_per_sample=3, max_gt_per_img=20, validation=False, random_sampling=True, use_mm_start_end=False, **kwargs):
        intro_string = f"{DEFAULT_IMAGE_TOKEN}\nThis provides an overview of the picture.\n"
        json_path = f"gpt4v_cocostuff_10k_detailed_caption_point.json"
        dataset_name = "mdvp_instruct/detailed_caption"
        image_dir = f"coco_2014/train2014"
        super().__init__(
            dataset_dir, tokenizer, global_image_encoder, precision, image_size, num_classes_per_sample,
            max_gt_per_img, validation, dataset_name, image_dir, json_path,
            intro_string, random_sampling,  use_mm_start_end, **kwargs)
        
class COCOStuff164kPointDetailedCap(PointRegDataset):
    def __init__(self, dataset_dir, tokenizer, global_image_encoder, precision="fp32",
                 image_size=1024, num_classes_per_sample=3, max_gt_per_img=20, validation=False, random_sampling=True, use_mm_start_end=False, **kwargs):
        intro_string = f"{DEFAULT_IMAGE_TOKEN}\nThis provides an overview of the picture.\n"
        json_path = f"gpt4v_cocostuff_164k_detailed_caption_point.json"
        dataset_name = "mdvp_instruct/detailed_caption"
        image_dir = f"coco/train2017"
        super().__init__(
            dataset_dir, tokenizer, global_image_encoder, precision, image_size, num_classes_per_sample,
            max_gt_per_img, validation, dataset_name, image_dir, json_path,
            intro_string, random_sampling,  use_mm_start_end, **kwargs)
        
class PascalPointDetailedCap(PointRegDataset):
    def __init__(self, dataset_dir, tokenizer, global_image_encoder, precision="fp32",
                 image_size=1024, num_classes_per_sample=3, max_gt_per_img=20, validation=False, random_sampling=True, use_mm_start_end=False, **kwargs):
        intro_string = f"{DEFAULT_IMAGE_TOKEN}\nThis provides an overview of the picture.\n"
        json_path = f"gpt4v_pascal_context_detailed_caption_point.json"
        dataset_name = "mdvp_instruct/detailed_caption"
        image_dir = f"pascal_part/VOCdevkit/VOC2010/JPEGImages"
        super().__init__(
            dataset_dir, tokenizer, global_image_encoder, precision, image_size, num_classes_per_sample,
            max_gt_per_img, validation, dataset_name, image_dir, json_path,
            intro_string, random_sampling,  use_mm_start_end, **kwargs)


class HybridDatasetBase(torch.utils.data.Dataset):
    PIXEL_MEAN = torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    PIXEL_STD = torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    IMG_SIZE = 1024
    IGNORE_LABEL = 255

    def __init__(self, dataset_dir, tokenizer, global_image_encoder, dataset, datasets_config,
                 precision="fp32", image_size=1024,
                 num_classes_per_sample=3, sample_rate=None, **kwargs):
        self.dataset_dir = dataset_dir
        self.tokenizer = tokenizer
        self.global_image_encoder = global_image_encoder
        self.dataset = dataset
        self.datasets_config = datasets_config
        self.precision = precision
        self.image_size = image_size
        self.num_classes_per_sample = num_classes_per_sample
        self.kwargs = kwargs
        self.dataset_list = dataset.split("||")
        self.sample_rate = np.array(sample_rate or [1] * len(self.dataset_list))
        self.sample_rate = self.sample_rate.astype(np.float64)
        self.sample_rate /= self.sample_rate.sum()
        self.all_datasets = self.create_datasets()
        self.epoch_samples = sum(len(item) for item in self.all_datasets)
    def create_datasets(self):
        datasets = []
        for ds in self.dataset_list:
            dataset_cls = self.datasets_config.get(ds)
            if dataset_cls:
                    datasets.append(
                        dataset_cls(
                            self.dataset_dir, self.tokenizer, self.global_image_encoder,
                            self.precision, self.image_size, self.num_classes_per_sample,**self.kwargs)
                        )
        return datasets

    def __len__(self):
        return self.epoch_samples
    @property
    def modality_lengths(self):
        length_list = []
        for dataset in self.all_datasets:
            length_list.extend(dataset.modality_lengths)
        return length_list
    def __getitem__(self, idx):
        dataset_idx = np.random.choice(len(self.dataset_list), p=self.sample_rate)
        selected_dataset = self.all_datasets[dataset_idx]
        index = np.random.choice(len(selected_dataset))
        data = selected_dataset[index]
        return (*data,)


class HybridMDVPRegDataset(HybridDatasetBase):
    def __init__(self, dataset_dir, tokenizer, global_image_encoder,
                 precision="fp32", image_size=1024, num_classes_per_sample=3,
                 dataset="ADE20K||VG||LVIS||COCOStuff10k||COCOStuff164k||Pascal", sample_rate=[1, 1, 1, 1, 1, 1], **kwargs):
        dataset="ADE20K||VG||LVIS||COCOStuff10k||COCOStuff164k||Pascal"
        sample_rate=[1, 1, 1, 1, 1, 1]
        datasets_config = {"ADE20K": ADE20KPointDetailedCap,
                           "VG": VGPointDetailedCap,
                           "LVIS": LVISPointDetailedCap,
                           "COCOStuff10k": COCOStuff10kPointDetailedCap,
                           "COCOStuff164k": COCOStuff164kPointDetailedCap,
                           "Pascal":PascalPointDetailedCap,
                           }
        super().__init__(
            dataset_dir, tokenizer, global_image_encoder, dataset, datasets_config,
            precision, image_size, num_classes_per_sample, sample_rate, **kwargs
        )
