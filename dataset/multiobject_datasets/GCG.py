import os
import cv2
import json
import copy
import random
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from pycocotools import mask
from pycocotools.coco import COCO
from transformers import CLIPImageProcessor
from mtrag.llava import conversation as conversation_lib
from mtrag.SAM.utils.transforms import ResizeLongestSide
from tools.utils import DEFAULT_IMAGE_TOKEN
from dataset.utils.utils import GCG_QUESTIONS
from dataset.utils.process import preprocess_multimodal, preprocess_v1

class GCGBaseDataset(torch.utils.data.Dataset):
    """
    Dataset Class for Grounded Conversation Generation (GCG) proposed in GLaMM.
    """
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

        self.question_templates = GCG_QUESTIONS
        self.begin_str = f"""{DEFAULT_IMAGE_TOKEN}\nThis provides an overview of the picture.\n"""
        self.validation = validation
        self.use_mm_start_end = use_mm_start_end

        # Defining paths
        self.base_dir = os.path.join(dataset_dir, dataset_name)
        self.image_folder = os.path.join(dataset_dir, image_dir)
        self.ann_file = os.path.join(self.base_dir, "annotations", "train", json_path)
        self.data_infos = self._load_annotations(self.ann_file)

    def _load_annotations(self, ann_file):
        with open(ann_file, 'r') as f:
            data_infos = json.load(f)
        data_infos = data_infos[0: 1000] if self.validation else data_infos
        return data_infos

    def _parse_annotations(self, ann_info):
        image_path = os.path.join(self.image_folder, ann_info['file_name'])
        annotations = {'labels': [], 'caption': [], 'masks': [], 'tokens_positive': [],
                       'file_name': ann_info['file_name']}
        width, height = Image.open(image_path).size
        annotations['caption'] = ann_info['caption'].strip('"').strip()

        for word, grounding in ann_info["groundings"].items():
            # Convert segmentation to binary mask
            binary_mask = np.zeros((height, width), dtype=np.uint8)
            for rle in grounding["rle_masks"]:
                m = mask.decode(rle).astype(np.uint8)
                binary_mask += m.squeeze()
            if binary_mask.max() == 0:
                continue
            binary_mask = np.clip(binary_mask, 0, 1)
            annotations['labels'].append(word)
            annotations['tokens_positive'].append(grounding["token_positives"])
            annotations['masks'].append(binary_mask)
        return annotations

    def __getitem__(self, index):
        while True:
            ann_info = self.data_infos[index] if (self.validation or not self.random_sampling) \
                else self.data_infos[random.randint(0, len(self.data_infos) - 1)]
            # Parse annotation info
            ann = self._parse_annotations(ann_info)
            image_path = os.path.join(self.image_folder, ann['file_name'])
            if len(ann['labels']) > 0:
                break
            else:
                index = random.randint(0, len(self.data_infos) - 1)
        data_item = {"image_path": image_path, "filename": ann['file_name'], "caption": ann['caption'],
            "labels": ann['labels'], "masks": ann['masks'], "tokens_positive": ann['tokens_positive']}
        return self.process_data(data_item)

    def __len__(self):
        return len(self.data_infos)
    
    @property
    def modality_lengths(self):
        length_list = []
        for i in range(len(self.data_infos)):
            ann_info = self.data_infos[i]
            cur_len = len(ann_info['caption'].strip('"').strip()) + len(random.choice(self.question_templates).strip())
            length_list.append(cur_len)
        return length_list

    def grounding_enc_processor(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.IMG_MEAN) / self.IMG_STD
        h, w = x.shape[-2:]
        x = F.pad(x, (0, self.IMG_SIZE - w, 0, self.IMG_SIZE - h))
        return x

    def create_conversations(self, caption, tokens_positive):
        question = random.choice(self.question_templates).strip()
        # Prepare caption with tags
        def tag_caption(caption, tokens):
            for start, end in sorted(tokens, key=lambda x: x[0], reverse=True):
                caption = f"{caption[:start]}<p> {caption[start:end]} </p> [SEG]{caption[end:]}"
            return caption

        detailed_answer = tag_caption(caption, tokens_positive)

        conversations = []
        conversations.append({'from': 'human', 'value': self.begin_str + question})
        conversations.append({'from': 'gpt', 'value':detailed_answer})
        return conversations

    def process_data(self, data_item):
         
        data_labels = data_item['labels']
        masks = data_item['masks']
        caption = data_item['caption']
        tokens_positive = data_item['tokens_positive']
        image_path = data_item['image_path']

        # Function to sort elements based on the start index of each phrase
        def sort_by_start_index(items, order):
            return [items[i] for i in order]

        # Sort phrases based on their appearance in the sentence
        phrase_order = sorted(range(len(tokens_positive)), key=lambda x: tokens_positive[x][0])
        masks = sort_by_start_index(masks, phrase_order)
        data_labels = sort_by_start_index(data_labels, phrase_order)
        tokens_positive = sort_by_start_index(tokens_positive, phrase_order)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Prepare input for Global Image Encoder
        global_enc_image = self.global_enc_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        # Prepare input for Grounding Image Encoder
        image = self.transform.apply_image(image)
        image_resize = image.shape[:2]
        grounding_enc_image = self.grounding_enc_processor(torch.from_numpy(image).permute(2, 0, 1).contiguous())
        region_masks = None

        conversations = self.create_conversations(caption, tokens_positive)
        conv = conversation_lib.default_conversation.copy()
        assert conv.sep_style == conversation_lib.SeparatorStyle.TWO
        sources = preprocess_multimodal(copy.deepcopy([conversations]), use_im_start_end=self.use_mm_start_end)
        input_ids, targets, conversations = preprocess_v1(sources, self.tokenizer)
        assert len(conversations) == 1
        masks = np.stack(masks, axis=0)
        masks = torch.from_numpy(masks)
        label = torch.ones(masks.shape[1:], dtype=torch.long) * self.IGNORE_LABEL
        return (
        image_path, global_enc_image, grounding_enc_image, region_masks, input_ids[0], targets[0], conversations, masks, label, image_resize)


class GranDfGCGDataset(GCGBaseDataset):
    """
    Human annotated dataset proposed in GLaMM as part of GranDf dataset.
    """
    def __init__(self, dataset_dir, tokenizer, global_image_encoder, precision="fp32",
                 image_size=1024, num_classes_per_sample=3, validation=False, random_sampling=True, use_mm_start_end=False, **kwargs):
        dataset_name = 'GranDf'
        json_path = "GranDf_HA_GCG_train.json"
        image_dir = os.path.join("GranDf_HA_images", "train")
        mode = "Val" if validation else "Train"
        super().__init__(
            dataset_dir, tokenizer, global_image_encoder, precision, image_size, num_classes_per_sample,
            validation,  dataset_name, image_dir, json_path, random_sampling, use_mm_start_end, **kwargs)
        print('\033[92m' + "----GCG-{}: GranDf-GCG dataset initialized----".format(mode) + '\033[0m')


class OpenPsgGCGDataset(GCGBaseDataset):
    def __init__(self, dataset_dir, tokenizer, global_image_encoder, precision="fp32",
                 image_size=1024, num_classes_per_sample=3, validation=False, random_sampling=True, use_mm_start_end=False, **kwargs):
        dataset_name = 'GranDf'
        json_files = {'validation': "OpenPsgGCG_val.json", 'training': "OpenPsgGCG_train.json"}
        json_path = json_files['validation'] if validation else json_files['training']
        image_dir = "coco"
        mode = "Val" if validation else "Train"

        super().__init__(
            dataset_dir, tokenizer, global_image_encoder, precision, image_size, num_classes_per_sample,
            validation,  dataset_name, image_dir, json_path, random_sampling, use_mm_start_end, **kwargs)
        print('\033[92m' + "----GCG-{}: OpenPSG-GCG dataset initialized----".format(mode) + '\033[0m')


class Flickr30kGCGDataset(GCGBaseDataset):
    def __init__(self, dataset_dir, tokenizer, global_image_encoder, precision="fp32",
                 image_size=1024, num_classes_per_sample=3, validation=False, random_sampling=True, use_mm_start_end=False, **kwargs):
        dataset_name = 'GranDf'
        json_files = {'validation': "flickr_mergedGT_GCG_val.json", 'training': "flickr_mergedGT_GCG_train.json"}
        json_path = json_files['validation'] if validation else json_files['training']
        image_dir = "flickr30k"
        mode = "Val" if validation else "Train"

        super().__init__(
            dataset_dir, tokenizer, global_image_encoder, precision, image_size, num_classes_per_sample,
            validation,  dataset_name, image_dir, json_path, random_sampling, use_mm_start_end, **kwargs)
        # Filter out images smaller than the minimum size
        self.data_infos = [self.data_infos[i] for i in self._filter_images(min_size=32)]
        self.validation = validation
        print('\033[92m' + "----GCG-{}: Flickr30k-GCG dataset initialized----".format(mode) + '\033[0m')


    def _load_annotations(self, ann_file):
        # Load annotations and filter out images with very short captions
        self.coco = COCO(ann_file)
        self.image_ids = self.coco.getImgIds()
        data_infos = []
        total_ann_ids = []
        removed_img_count = 0
        for img_id in self.image_ids:
            if len(data_infos) == 1000 and self.validation:
                # Only limited images for validation
                break
            info = self.coco.loadImgs([img_id])[0]
            if len(info['caption'].split(' ')) < 3:
                removed_img_count += 1
                continue
            info['filename'] = info['file_name'].split('_')[-1]
            info['height'] = int(info['height'])
            info['width'] = int(info['width'])
            data_infos.append(info)
            ann_ids = self.coco.getAnnIds(imgIds=[img_id])
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        print(f'Removed {removed_img_count} images.')
        return data_infos

    def _filter_images(self, min_size):
        return [i for i, info in enumerate(self.data_infos) if min(info['width'], info['height']) >= min_size]

    def _parse_annotations(self, img_info, ann_info):
        annotations = {'labels': [], 'caption': img_info['caption'], 'masks': [],
                       'tokens_positive': []}
        for ann in ann_info:
            if ann.get('ignore', False):
                continue
            rle = ann['sam_mask']
            mask_decoded = mask.decode(rle).astype(np.uint8)
            mask_decoded = np.clip(mask_decoded, 0, 1)
            if mask_decoded.max() == 0:
                continue
            tokens_positive = ann['tokens_positive']
            gt_label = [img_info['caption'][span[0]:span[1]] for span in tokens_positive]
            annotations['labels'].append(gt_label[0])
            annotations['tokens_positive'].append(tokens_positive[0])
            annotations['masks'].append(mask_decoded)
        return annotations

    def __getitem__(self, index):
        img_info = self.data_infos[index] if (self.validation or not self.random_sampling) \
            else self.data_infos[random.randint(0, len(self.data_infos) - 1)]
        ann_ids = self.coco.getAnnIds(imgIds=img_info['id'])
        ann_info = self.coco.loadAnns(ann_ids)
        image_path = os.path.join(self.image_folder, img_info['file_name'])
        # Parse annotation info
        ann = self._parse_annotations(img_info, ann_info)
        while len(ann['labels']) == 0:
            index = random.randint(0, len(self.data_infos) - 1)
            img_info = self.data_infos[index]
            ann_ids = self.coco.getAnnIds(imgIds=img_info['id'])
            ann_info = self.coco.loadAnns(ann_ids)
            image_path = os.path.join(self.image_folder, img_info['file_name'])
            ann = self._parse_annotations(img_info, ann_info)
        data_item = {"image_path": image_path, "filename": img_info['file_name'], "width": img_info['width'],
                     "height": img_info['height'], "caption": ann['caption'],
                     "labels": ann['labels'], "masks": ann['masks'], "tokens_positive": ann['tokens_positive']}
        return self.process_data(data_item)


class RefCOCOgGCGDataset(GCGBaseDataset):
    def __init__(self, dataset_dir, tokenizer, global_image_encoder, precision="fp32",
                 image_size=1024, num_classes_per_sample=3, validation=False, random_sampling=True, use_mm_start_end=False, **kwargs):
        dataset_name = 'GranDf'
        json_files = {'validation': "RefCOCOg_GCG_val.json", 'training': "RefCOCOg_GCG_train.json"}
        json_path = json_files['validation'] if validation else json_files['training']
        image_dir = os.path.join("coco_2014", "train2014")
        mode = "Val" if validation else "Train"

        super().__init__(
            dataset_dir, tokenizer, global_image_encoder, precision, image_size, num_classes_per_sample,
            validation,  dataset_name, image_dir, json_path, random_sampling, use_mm_start_end, **kwargs)
        print('\033[92m' + "----GCG-{}: RefCOCOg-GCG dataset initialized----".format(mode) + '\033[0m')

    @property
    def modality_lengths(self):
        length_list = []
        for i in range(len(self.data_infos)):
            ann_dict = self.data_infos[i]
            ann_info = next(iter(ann_dict.values()))
            # ann = self._parse_annotations(ann_info)
            cur_len = len(ann_info ['caption'].strip('"').strip()) + len(random.choice(self.question_templates).strip())
            length_list.append(cur_len)
        return length_list

    def _parse_annotations(self, ann_info):
        image_path = os.path.join(self.image_folder, ann_info['img_file_name'])
        annotations = {'labels': [], 'caption': [], 'masks': [], 'tokens_positive': [],
                       'file_name': ann_info['img_file_name']}
        width, height = Image.open(image_path).size
        orig_caption = ann_info['caption'].strip('"').strip()
        annotations['caption'] = orig_caption.lower()
        for detail in ann_info['refs']:
            phrase = detail['sentence']
            if phrase.lower() in annotations['caption']:
                # Convert segmentation to binary mask
                binary_mask = np.zeros((height, width), dtype=np.uint8)
                for seg in detail["segmentation"]:
                    rles = mask.frPyObjects([seg], height, width)
                    m = mask.decode(rles)
                    m = m.astype(np.uint8)
                    binary_mask += m.squeeze()
                binary_mask = np.clip(binary_mask, 0, 1)
                if binary_mask.max() == 1:
                    annotations['labels'].append(phrase)
                    index = annotations['caption'].find(phrase)
                    end_index = index + len(phrase) if index != -1 else -1
                    annotations['tokens_positive'].append([index, end_index])
                    annotations['masks'].append(binary_mask)

        # Sort tokens_positive and corresponding lists
        tokens_positive = annotations['tokens_positive']
        sorted_indices = sorted(range(len(tokens_positive)), key=lambda i: tokens_positive[i][0])
        annotations['tokens_positive'] = [tokens_positive[i] for i in sorted_indices]
        annotations['masks'] = [annotations['masks'][i] for i in sorted_indices]
        annotations['labels'] = [annotations['labels'][i] for i in sorted_indices]
        tokens_positive = annotations['tokens_positive'] 
        # Trimming overlapping intervals
        for i in range(len(tokens_positive)):
            for j in range(i + 1, len(tokens_positive)):
                # If there is overlap
                if tokens_positive[i][1] >= tokens_positive[j][0]:
                    # Modify the end index of phrase i to be one less than the start index of phrase j
                    tokens_positive[i][1] = tokens_positive[j][0] - 1
                    # Modify the phrases to reflect the change in indices
                    annotations['labels'][i] = orig_caption[tokens_positive[i][0]:tokens_positive[i][1] + 1]
                    break  # Exit inner loop since i was modified
        return annotations

    def __getitem__(self, index):
        while True:
            ann_dict = self.data_infos[index] if (self.validation or not self.random_sampling) \
                else self.data_infos[random.randint(0, len(self.data_infos) - 1)]
            ann_info = next(iter(ann_dict.values()))
            # Parse annotation info
            ann = self._parse_annotations(ann_info)
            image_path = os.path.join(self.image_folder, ann['file_name'])
            # Check if len(gt_phrases) > 0 and if True, break the loop
            if len(ann['labels']) > 0:
                break
            else:
                index = random.randint(0, len(self.data_infos) - 1)
        data_item = {"image_path": image_path, "filename": ann['file_name'], "caption": ann['caption'],
                     "labels": ann['labels'], "masks": ann['masks'], "tokens_positive": ann['tokens_positive']}

        return self.process_data(data_item)

