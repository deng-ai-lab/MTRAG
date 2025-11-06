import os
import cv2
import copy
import numpy as np
import random
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from transformers import CLIPImageProcessor
from mtrag.llava import conversation as conversation_lib
from mtrag.SAM.utils.transforms import ResizeLongestSide
from tools.utils import DEFAULT_IMAGE_TOKEN
from dataset.utils.utils import  REGION_GROUP_QUESTIONS, REGION_SEPARATE_QUESTIONS, HUMANIZED_INTRODUCTIONS
from dataset.utils.process import preprocess_multimodal, preprocess_v1

class Flickr30kRegDataset(torch.utils.data.Dataset):
    CLASSES = ('object',)
    IMG_MEAN = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    IMG_STD = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    IMG_SIZE = 1024
    IGNORE_LABEL = 255

    def __init__(self, dataset_dir, tokenizer, global_image_encoder, precision="fp32",
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

        self.base_dir = dataset_dir
        self.image_folder = os.path.join(self.base_dir, "flickr30k")
        self.ann_file = os.path.join(dataset_dir, "region_cap", "final_flickr_mergedGT_train.json")
        self.data_infos = self._load_annotations(self.ann_file)
        self.data_infos = [self.data_infos[i] for i in self._filter_images(min_size=32)]
        self.id_cap_dict = dict()
        self.begin_str = f"{DEFAULT_IMAGE_TOKEN}\nThis provides an overview of the picture.\n"
        self.grounding = kwargs.get("grounding", True)
        print('\033[92m' + "----REGION-Train: Loaded Flickr30k dataset ----" + '\033[0m')

    def _load_annotations(self, ann_file):
        self.coco = COCO(ann_file)
        img_ids = self.coco.getImgIds()
        data_infos = []
        for img_id in img_ids:
            info = self.coco.loadImgs([img_id])[0]
            if len(info['caption'].split(' ')) < 3:
                continue
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
        annotations = {'region_masks': [], 'labels': [], 'masks_ann': []}
        self.cat_ids = self.coco.getCatIds(catNms=self.CLASSES)
        self.id_cap_dict = dict()
        self.id_cap_dict[img_info['file_name']] = img_info['caption']

        for ann in ann_info:
            if ann.get('ignore', False) or ann['area'] <= 0 or ann['bbox'][2] < 1 or ann['bbox'][3] < 1:
                continue
            bbox = self._get_valid_bbox(ann['bbox'], img_info['width'], img_info['height'])
            if bbox:
                if ann.get('iscrowd', False):
                    continue
                else:
                    mask = self._generate_mask((img_info['height'], img_info['width']), bbox)
                    annotations['region_masks'].append(mask)
                    gt_list = [img_info['caption'][atp[0]:atp[1]] for atp in ann['tokens_positive']]
                    annotations['labels'].append(gt_list[0])
                    annotations['masks_ann'].append(ann.get('segmentation', None))
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
        data_item = {
            "image_path": os.path.join(self.image_folder, img_info['file_name']),
            "filename": img_info['file_name'],
            "width": img_info['width'],
            "height": img_info['height'],
            "region_masks": ann['region_masks'],
            "caption": img_info['caption'],
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
            cur_len = len(img_info['caption'])
            cur_len += sum([len(x) for x in ann['labels']])
            length_list.append(cur_len)
        return length_list

    def grounding_enc_processor(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.IMG_MEAN) / self.IMG_STD
        h, w = x.shape[-2:]
        x = F.pad(x, (0, self.IMG_SIZE - w, 0, self.IMG_SIZE - h))
        return x

    def create_conversations(self, ori_labels, caption):
        conversations = dict()
        conversations['conversations'] = []
        region_string = ''
        region_string_reg = ''
        region_template = "region{num}:{content}"
        reg_answer = random.choice(HUMANIZED_INTRODUCTIONS).strip() + "\n"
        for i in range(len(ori_labels)):
            region_string_reg += f'region{i + 1} <region_fea>'
            region_string += f'region{i + 1}'
            reg_answer += region_template.format(num=i+1, content=ori_labels[i])+"\n"
            if i < len(ori_labels)-2:
                region_string_reg += ', '
                region_string += ', '
            elif i == len(ori_labels)-2:
                region_string_reg += ' and '
                region_string += ' and '
        reg_question = random.choice( REGION_SEPARATE_QUESTIONS).replace('<region>', region_string_reg).strip()
        question = random.choice(REGION_GROUP_QUESTIONS).strip()
        detailed_question = question.replace('<region>', region_string)
        conversations['conversations'].append(
            {'from': 'human', 'value': self.begin_str + reg_question})
        conversations['conversations'].append({'from': 'gpt', 'value': reg_answer})
        conversations['conversations'].append(
            {'from': 'human', 'value': detailed_question})
        conversations['conversations'].append({'from': 'gpt', 'value': caption})
        return conversations
    
    def generate_alpha_for_feature(self, image, masks):
        image_masks = np.where(masks > 0, 255.0, 0.0)
        rgb_masks = np.repeat(image_masks[:, :, :, np.newaxis], 3, axis=-1).astype(np.uint8)  # (batch, H, W, 3)
        global_enc_image_mask = self.global_enc_processor.preprocess(list(rgb_masks)+[image], return_tensors="pt")["pixel_values"]
        global_enc_image = global_enc_image_mask[-1]
        global_enc_image_mask = global_enc_image_mask[:-1].sum(dim=1, keepdim=True)
        global_enc_image_mask = (global_enc_image_mask > 6) * 1.9231 + (global_enc_image_mask <= 6) * (-1.9231)
        return global_enc_image, global_enc_image_mask.squeeze(1)

    def process_data(self, data_item):
        data_labels = data_item['labels']
        data_masks = data_item['region_masks']
        caption = data_item['caption']
        image_path = data_item['image_path']
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Prepare conversations
        shuffle_ids = torch.randperm(len(data_labels))
        if len(shuffle_ids) > self.max_gt_per_img:
            shuffle_ids = shuffle_ids[:self.max_gt_per_img]
        region_masks = [data_masks[i] for i in shuffle_ids]
        ori_labels = [data_labels[i] for i in shuffle_ids]
        conversations = self.create_conversations(ori_labels, caption)
        conv = conversation_lib.default_conversation.copy()
        assert conv.sep_style == conversation_lib.SeparatorStyle.TWO
        sources = preprocess_multimodal(copy.deepcopy([conversations["conversations"]]), use_im_start_end=self.use_mm_start_end)
        input_ids, targets, conversations = preprocess_v1(sources, self.tokenizer)
        assert len(conversations) == 1
        # Prepare input for Global Image Encoder
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
    