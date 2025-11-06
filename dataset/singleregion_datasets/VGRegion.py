import os
import cv2
import copy
import random
import numpy as np
import torch
from pycocotools.coco import COCO
import torch.nn.functional as F
from transformers import CLIPImageProcessor
from mtrag.llava import conversation as conversation_lib
from mtrag.SAM.utils.transforms import ResizeLongestSide
from tools.utils import DEFAULT_IMAGE_TOKEN
from dataset.utils.process import preprocess_multimodal, preprocess_v1
from types import SimpleNamespace
class VisualGenomeRegDataset(torch.utils.data.Dataset):
    CLASSES = ('object',)
    IMG_MEAN = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    IMG_STD = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    IMG_SIZE = 1024
    IGNORE_LABEL = 255

    def __init__(self, dataset_dir, tokenizer, global_image_encoder,  precision="fp32",
                 image_size=1024, num_classes_per_sample=3, max_gt_per_img=10, validation=False, random_sampling=True, use_mm_start_end=False, **kwargs):
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
        self.begin_str = f""""{DEFAULT_IMAGE_TOKEN}\nThis provides an overview of the picture.\n"""
        self.question_templates = ["Please give me a short description of <region>",]
        json_files = {'validation': "vg_test_caption.json", 'training': "vg_train_caption.json"}
        json_path = json_files['validation'] if validation else json_files['training']
        self.base_dir = dataset_dir
        self.ann_file = os.path.join(self.base_dir,"region_cap", json_path)
        self.image_folder = os.path.join(dataset_dir, "vg")

        self.data_infos = self._load_annotations(self.ann_file)
        self.data_infos = [self.data_infos[i] for i in self._filter_images(min_size=32)]
        self.grounding = kwargs.get("grounding", True)
        self.model_cfg = SimpleNamespace(**kwargs)
    def _load_annotations(self, ann_file):
        self.coco = COCO(ann_file)
        img_ids = self.coco.getImgIds()
        data_infos = []
        for img_id in img_ids:
            if self.validation and len(data_infos) == 1000:
                # limited images during validation
                break
            info = self.coco.loadImgs([img_id])[0]
            info['filename'] = info['file_name']
            info['height'] = int(info['height'])
            info['width'] = int(info['width'])
            data_infos.append(info)
        return data_infos

    def _filter_images(self, min_size):
        return [i for i, info in enumerate(self.data_infos) if min(info['width'], info['height']) >= min_size]
    def _generate_mask(self, shape, bbox):
        x1, y1, x2, y2 = bbox
         
        mask = np.zeros(shape, dtype=np.uint8)
        mask[int(y1):int(y2), int(x1):int(x2)] = 1
        return mask
    def _parse_annotations(self, img_info, ann_info):
        annotations = {'region_masks': [], 'labels': [], }
        for ann in ann_info:
            if ann.get('ignore', False):
                continue
            # Check for valid area and dimensions
            if ann['area'] <= 0 or ann['bbox'][2] < 1 or ann['bbox'][3] < 1:
                continue
            bbox = self._get_valid_bbox(ann['bbox'], img_info['width'], img_info['height'])
            if bbox:
                mask = self._generate_mask((img_info['height'], img_info['width']), bbox)
                annotations['region_masks'].append(mask)
                annotations['labels'].append(ann['caption'].strip())
        return annotations
    def regulate_box(self, box, img_w, img_h):
        return [max(0, min(box[0], img_w-1)), max(0, min(box[1], img_h-1)), max(0, min(box[2], img_w-1)), max(0, min(box[3], img_h-1))]
    def _get_valid_bbox(self, bbox, img_width, img_height):
        x1, y1, w, h = bbox
        inter_w = max(0, min(x1 + w, img_width) - max(x1, 0))
        inter_h = max(0, min(y1 + h, img_height) - max(y1, 0))
        if inter_w * inter_h == 0:
            return None
        return self.regulate_box([x1, y1, x1 + w, y1 + h], img_width, img_height)

    def __getitem__(self, index):
        while True:
            img_info = self.data_infos[index] if (self.validation or not self.random_sampling) \
                else self.data_infos[random.randint(0, len(self.data_infos) - 1)]
            ann_info = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_info['id']))
            ann = self._parse_annotations(img_info, ann_info)
            if len(ann['region_masks']) > 0:
                break
            else:
                index = random.randint(0, len(self.data_infos) - 1)
        subdirs = ['VG_100K', 'VG_100K_2']
        image_path = os.path.join(self.image_folder, subdirs[0], img_info['filename'])
        if not os.path.exists(image_path):
            image_path = os.path.join(self.image_folder, subdirs[1], img_info['filename'])           
        data_item = {
            "image_path": image_path,
            "width": img_info['width'],
            "height": img_info['height'],
            "region_masks": ann['region_masks'],
            "labels": ann['labels'],
        }

        return self.process_data(data_item)

    def __len__(self):
        return len(self.data_infos)
    @property
    def modality_lengths(self):
        length_list = []
        for i in range(len(self.data_infos)):
            img_info = self.data_infos[i]
            ann_info = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_info['id']))
            ann = self._parse_annotations(img_info, ann_info)
            cur_len = sum(len(x) for x in ann['labels'])
            length_list.append(cur_len)
        return length_list
    def grounding_enc_processor(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.IMG_MEAN) / self.IMG_STD
        h, w = x.shape[-2:]
        x = F.pad(x, (0, self.IMG_SIZE - w, 0, self.IMG_SIZE - h))
        return x
    def create_conversations(self, ori_labels):
        conversations = dict()
        conversations['conversations'] = []
        for i, label in enumerate(ori_labels):
            question = random.choice(self.question_templates).strip().replace('<region>', f'region{i + 1} <region_fea>')
            if i == 0:
                question = self.begin_str + question
            conversations['conversations'].append({'from': 'human', 'value': question})
            conversations['conversations'].append({'from': 'gpt', 'value': label})
        return conversations

    def generate_alpha_for_feature(self, image, masks):
        image_masks = np.where(masks > 0, 255.0, 0.0)
        rgb_masks = np.repeat(image_masks[:, :, :, np.newaxis], 3, axis=-1).astype(np.uint8)  # (batch, H, W, 3)
        global_enc_image_mask = self.global_enc_processor.preprocess(list(rgb_masks)+[image], return_tensors="pt")["pixel_values"]
        # global_enc_image_mask = process_images(list(rgb_masks)+[image], self.global_enc_processor, self.model_cfg)
        global_enc_image = global_enc_image_mask[-1]
        global_enc_image_mask = global_enc_image_mask[:-1].sum(dim=1, keepdim=True)
        global_enc_image_mask = (global_enc_image_mask > 6) * 1.9231 + (global_enc_image_mask <= 6) * (-1.9231)
        return global_enc_image, global_enc_image_mask.squeeze(1)

    def process_data(self, data_item):
        data_labels = data_item['labels']
        data_masks = data_item['region_masks']
        image_path = data_item['image_path']
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Prepare conversations
        num_mask = random.choice(range(1, self.max_gt_per_img))
        shuffle_ids = torch.randperm(len(data_labels))
        if len(shuffle_ids) > num_mask :
            shuffle_ids = shuffle_ids[:num_mask]
        region_masks = [data_masks[i] for i in shuffle_ids]
        ori_labels = [data_labels[i] for i in shuffle_ids]
        conversations = self.create_conversations(ori_labels)
        conv = conversation_lib.default_conversation.copy()
        assert conv.sep_style == conversation_lib.SeparatorStyle.TWO
        sources = preprocess_multimodal(copy.deepcopy([conversations["conversations"]]), use_im_start_end=self.use_mm_start_end)
        input_ids, targets, conversations = preprocess_v1(sources, self.tokenizer)
        assert len(conversations) == 1
        # Prepare input for Region Image Encoder
        region_masks = np.array(region_masks)
        global_enc_image, global_enc_image_mask = self.generate_alpha_for_feature(image, region_masks)
        global_enc_image = torch.cat([global_enc_image, global_enc_image_mask], dim=0)

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
    

