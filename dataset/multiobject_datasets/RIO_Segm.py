import os
import cv2
import json
import copy
import random
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask as maskUtils
from transformers import CLIPImageProcessor
from mtrag.llava import conversation as conversation_lib
from mtrag.SAM.utils.transforms import ResizeLongestSide
from tools.utils import DEFAULT_IMAGE_TOKEN
from dataset.utils.utils import RIO_QUESTIONS, RIO_ANSWERS
from dataset.utils.process import preprocess_multimodal, preprocess_v1
from mtrag.llava.mm_utils import tokenizer_image_token
from mtrag.llava.constants import IMAGE_TOKEN_INDEX
class RIOSegmDataset(torch.utils.data.Dataset):
    CLASSES = ('object',)
    IMG_MEAN = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    IMG_STD = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    IMG_SIZE = 1024
    IGNORE_LABEL = 255
    def __init__(self, dataset_dir, tokenizer, global_image_encoder, precision="fp32",
                 image_size=1024, num_classes_per_sample=3, validation=False, dataset_name='',
                 image_dir='', json_path='', random_sampling=True,use_mm_start_end=False, **kwargs):
        self.num_classes_per_sample = num_classes_per_sample
        self.dataset_dir = dataset_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.global_enc_processor = CLIPImageProcessor.from_pretrained(global_image_encoder)
        self.validation = validation
        self.random_sampling = random_sampling

        self.question_templates = RIO_QUESTIONS
        self.answer_templates = RIO_ANSWERS
        self.begin_str = f"""{DEFAULT_IMAGE_TOKEN}\nThis provides an overview of the picture.\n"""
        self.use_mm_start_end = use_mm_start_end

        # Defining paths
        self.base_dir = os.path.join(dataset_dir, dataset_name)
        self.image_folder = os.path.join(dataset_dir, image_dir)
        self.ann_file = os.path.join(self.base_dir, json_path)
        self.data_infos = self._load_annotations(self.ann_file)
        self.data_infos = [sample for sample in self.data_infos if len(sample['mask_list'])> 0 and min(sample['width'], sample['height']) >= 32]
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
    def get_mask_center_hw(self, mask):
        indices = np.argwhere(mask == 1)
        if len(indices) == 0:
            return None 
        h_y, w_x = indices.mean(axis=0).astype(int)
        return h_y, w_x
    def _load_annotations(self, ann_file):
        with open(ann_file, 'r') as f:
            data_infos = json.load(f)
        return data_infos

    def _parse_annotations(self, ann_info):
        image_path = os.path.join(self.image_folder, ann_info['file_name'])

        annotations = {'labels': [],'expressions':ann_info['expressions'][0], 'answer': '', 'masks': [],
                       'file_name': ann_info['file_name'], "height": ann_info['height'], "width": ann_info['width']}
        height, width = ann_info['height'], ann_info['width']
        categories = ann_info['category']
        segm_list = ann_info['mask_list']
        segm_list_mask = [self.annToMask(segm, height, width) for segm in segm_list]
        segm_list_mask = [segm for segm in segm_list_mask if segm.sum() > 0]
        center_points = [self.get_mask_center_hw(segm) for segm in segm_list_mask]
        sorted_indices = sorted(range(len(center_points)), key=lambda i: (center_points[i][0], center_points[i][1]) )
        objs = ''
        for i in range(len(segm_list_mask)):
            cat_i = categories[sorted_indices[i]]["name"]
            annotations['labels'].append(cat_i)
            binary_mask = segm_list_mask[sorted_indices[i]]
            annotations['masks'].append(binary_mask)
            objs += '<p> {}:ID{} </p> [SEG]'.format(cat_i, i)
            if i < len(segm_list_mask) - 2:
                objs += ', '
            elif i == len(segm_list_mask) - 2:
                objs += ' and '
        annotations['answer'] = objs
        return annotations

    def __getitem__(self, index):
        while True:
            ann_info = self.data_infos[index] if (self.validation or not self.random_sampling) \
                else self.data_infos[random.randint(0, len(self.data_infos) - 1)]
            # Parse annotation info
            ann = self._parse_annotations(ann_info)
            image_path = os.path.join(self.image_folder, ann['file_name'])
            if len(ann['masks']) > 0 and min(ann['width'], ann['height']) >= 32:
                break
            else:
                index = random.randint(0, len(self.data_infos) - 1)
        data_item = {"image_path": image_path, "expressions":ann["expressions"], "filename": ann['file_name'], "answer": ann['answer'],
            "labels": ann['labels'], "masks": ann['masks']}
        return self.process_data(data_item)

    def __len__(self):
        return len(self.data_infos)
    @property
    def modality_lengths(self):
        length_list = []
        for i in range(len(self.data_infos)):
            ann_info = self.data_infos[i]
            cur_len = len(ann_info['expressions'][0] + random.choice(self.question_templates).strip()) 
            length_list.append(cur_len)
        return length_list

    def grounding_enc_processor(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.IMG_MEAN) / self.IMG_STD
        h, w = x.shape[-2:]
        x = F.pad(x, (0, self.IMG_SIZE - w, 0, self.IMG_SIZE - h))
        return x

    def create_conversations(self, expressions, answer):
        question = random.choice(self.question_templates).strip()
        answer_template = random.choice(self.answer_templates)
        conversations = []
        conversations.append({'from': 'human', 'value': self.begin_str+question.format(sent=expressions)})
        conversations.append({'from': 'gpt', 'value':answer_template.format(object=answer)})
        return conversations

    def process_data(self, data_item):
         
        data_labels = data_item['labels']
        masks = data_item['masks']
        expressions = data_item['expressions']
        answer = data_item["answer"]
        image_path = data_item['image_path']

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Prepare input for Global Image Encoder
        global_enc_image = self.global_enc_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        # Prepare input for Grounding Image Encoder
        image = self.transform.apply_image(image)
        image_resize = image.shape[:2]
        grounding_enc_image = self.grounding_enc_processor(torch.from_numpy(image).permute(2, 0, 1).contiguous())
        region_masks = None

        conversations = self.create_conversations(expressions, answer)
        conv = conversation_lib.default_conversation.copy()
        assert conv.sep_style == conversation_lib.SeparatorStyle.TWO
        sources = preprocess_multimodal(copy.deepcopy([conversations]), use_im_start_end=self.use_mm_start_end)
        input_ids, targets, conversations = preprocess_v1(sources, self.tokenizer)
         
        assert len(conversations) == 1
        masks = np.stack(masks, axis=0)
        masks = torch.from_numpy(masks)
        assert conversations[0].count("[SEG]") == masks.shape[0]
        label = torch.ones(masks.shape[1:], dtype=torch.long) * self.IGNORE_LABEL
        return (
        image_path, global_enc_image, grounding_enc_image, region_masks, input_ids[0], targets[0], conversations, masks, label, image_resize)


class RIOTrainSegmDataset(RIOSegmDataset):
    def __init__(self, dataset_dir, tokenizer, global_image_encoder, precision="fp32",
                 image_size=1024, num_classes_per_sample=3, validation=False, random_sampling=True, use_mm_start_end=False, **kwargs):
        dataset_name = 'Reason/RIO'
        json_path = "RIO_train_all_thing_Cat.json"
        image_dir = os.path.join("coco_2014", "val2014") if validation else os.path.join("coco_2014", "train2014")
        mode = "Val" if validation else "Train"
        super().__init__(
            dataset_dir, tokenizer, global_image_encoder, precision, image_size, num_classes_per_sample,
            validation,  dataset_name, image_dir, json_path, random_sampling, use_mm_start_end, **kwargs)
        print('\033[92m' + "----RIOSegment-{}: RIOSegment dataset initialized----".format(mode) + '\033[0m')
