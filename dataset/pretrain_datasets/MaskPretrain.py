import os
import cv2
import copy
import torch
import random
import numpy as np
import torch.nn.functional as F
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from transformers import CLIPImageProcessor
from mtrag.llava import conversation as conversation_lib
from mtrag.SAM.utils.transforms import ResizeLongestSide
from tools.utils import DEFAULT_IMAGE_TOKEN
from dataset.utils.process import preprocess_multimodal, preprocess_v1
class RegionBaseDataset(torch.utils.data.Dataset):
    CLASSES = ('object',)
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
        self.data_infos = [self.data_infos[i] for i in self._filter_images(min_size=32)]
        self.grounding = kwargs.get("grounding", True)
    def _load_annotations(self, ann_file):
        self.coco = COCO(ann_file)
        img_ids = self.coco.getImgIds()
        data_infos = []
        total_ann_ids = []
        for img_id in img_ids:
            if self.validation and len(data_infos) == 1000:
                # limited images during validation
                break
            info = self.coco.loadImgs([img_id])[0]
            info['filename'] = info['file_name']
            info['height'] = int(info['height'])
            info['width'] = int(info['width'])
            
            ann_ids = self.coco.getAnnIds(imgIds=[img_id])
            ann_info = self.coco.loadAnns(ann_ids)
            if len(ann_info)==0:
                continue
            data_infos.append(info)
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        return data_infos

    def _filter_images(self, min_size):
        return [i for i, info in enumerate(self.data_infos) if min(info['width'], info['height']) >= min_size]
    
    def regulate_box(self, box, img_w, img_h):
        return [max(0, min(box[0], img_w-1)), max(0, min(box[1], img_h-1)), max(0, min(box[2], img_w-1)), max(0, min(box[3], img_h-1))]
    def _get_valid_bbox(self, bbox, img_width, img_height):
        x1, y1, w, h = bbox
        inter_w = max(0, min(x1 + w, img_width) - max(x1, 0))
        inter_h = max(0, min(y1 + h, img_height) - max(y1, 0))
        if inter_w * inter_h == 0:
            return None
        return self.regulate_box([x1, y1, x1 + w, y1 + h], img_width, img_height)
    def _generate_mask(self, shape, bbox):
        x1, y1, x2, y2 = bbox
         
        mask = np.zeros(shape, dtype=np.uint8)
        mask[int(y1):int(y2), int(x1):int(x2)] = 1
        return mask

    def _parse_annotations(self, img_info, ann_info):
        annotations = {'labels': [], 'region_masks': []}
        for ann in ann_info:
            mask = self.annToMask(ann['segmentation'], img_info['height'], img_info['width'])
            if mask.sum().item()<=0:
                if "bbox" in ann:
                    bbox = ann["bbox"]
                    bbox_valid = self._get_valid_bbox(bbox, img_info['width'], img_info['height'])
                    if bbox_valid is not None:
                        annotations["region_masks"].append(self._generate_mask((img_info['height'], img_info['width']), bbox_valid))
                    else:
                        annotations["region_masks"].append(mask)
                else:
                    annotations["region_masks"].append(mask)
            else:
                annotations["region_masks"].append(mask)
            cat = self.coco.loadCats(ann['category_id'])
            annotations['labels'].append(cat[0]['name'])
        return annotations
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

    def __getitem__(self, index):
        img_info = self.data_infos[index]
        ann_info = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_info['id']))
        ann = self._parse_annotations(img_info, ann_info)
        while len(ann['region_masks']) == 0:
            index = random.randint(0, len(self.data_infos) - 1)
            img_info = self.data_infos[index]
            ann_info = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_info['id']))
            ann = self._parse_annotations(img_info, ann_info)

        data_item = {
            "image_path": os.path.join(self.image_folder, img_info['filename']),
            "width": img_info['width'],
            "height": img_info['height'],
            "labels": ann['labels'],
            "region_masks": ann['region_masks'],
        }

        return self.process_data(data_item)

    def __len__(self):
        return len(self.data_infos)

    def grounding_enc_processor(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.IMG_MEAN) / self.IMG_STD
        h, w = x.shape[-2:]
        x = F.pad(x, (0, self.IMG_SIZE - w, 0, self.IMG_SIZE - h))
        return x

    def create_conversations(self, ori_labels):
        conversations = dict()
        conversations['conversations'] = []
        for i in range(len(ori_labels)):
            question = '<region>'
            region_string = f'region{i + 1} <region_fea>.'
            question = question.replace('<region>', region_string)
            if i == 0:
                question = self.begin_str + question
            answer = ori_labels[i]
            conversations['conversations'].append(
                {'from': 'human', 'value': question})
            conversations['conversations'].append({'from': 'gpt', 'value': answer})
        return conversations
    
    #     return global_enc_image_mask
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
        image_path = data_item['image_path']
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Prepare input for Global Image Encoder
        shuffle_ids = torch.randperm(len(data_labels))
        if len(shuffle_ids) > self.max_gt_per_img:
            shuffle_ids = shuffle_ids[:self.max_gt_per_img]
        region_masks = [data_masks[i] for i in shuffle_ids]
        ori_labels = [data_labels[i] for i in shuffle_ids]
        
        conversations = self.create_conversations(ori_labels)
        conv = conversation_lib.default_conversation.copy()
        assert conv.sep_style == conversation_lib.SeparatorStyle.TWO
        sources = preprocess_multimodal(copy.deepcopy([conversations["conversations"]]), use_im_start_end=self.use_mm_start_end)
        input_ids, targets, conversations = preprocess_v1(sources, self.tokenizer)
        assert len(conversations) == 1
        region_masks = np.array(region_masks)
        global_enc_image, global_enc_image_mask = self.generate_alpha_for_feature(image, region_masks)
        global_enc_image = torch.cat([global_enc_image, global_enc_image_mask], dim=0)

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




class COCOPretrain(RegionBaseDataset):
    def __init__(self, dataset_dir, tokenizer, global_image_encoder, precision="fp32",
                 image_size=1024, num_classes_per_sample=3, max_gt_per_img=20, validation=False, random_sampling=True, use_mm_start_end=False,**kwargs):
        intro_string = DEFAULT_IMAGE_TOKEN + "\n" + ("In the conversation below, you simply answer the category name based on what you see "
                                                     "in the imagery inside a particular region. I will give you only one region each time.\n")
        mode = "val" if validation else "train"
        json_path = f"instances_{mode}2017.json"
        dataset_name = "coco/annotations"
        image_dir = f"coco/{mode}2017"
        super().__init__(
            dataset_dir, tokenizer, global_image_encoder, precision, image_size, num_classes_per_sample,
            max_gt_per_img, validation, dataset_name, image_dir, json_path,
            intro_string, random_sampling,  use_mm_start_end,**kwargs)


class PartImagenetPretrain(RegionBaseDataset):
    def __init__(self, dataset_dir, tokenizer, global_image_encoder, precision="fp32",
                 image_size=1024, num_classes_per_sample=3, max_gt_per_img=15, validation=False, random_sampling=True, use_mm_start_end=False,**kwargs):
        CAT_CLASSES = (
            'Bottle', 'Biped', 'Quadruped', 'Fish', 'Reptile', 'Bicycle', 'Bird', 'Car', 'Boat', 'Snake', 'Aeroplane'
        )

        SUB_CLASSES = (
            'Tier', 'Hand', 'Wing', 'Mouth', 'Tail', 'Side', 'Fin', 'Engine', 'Foot', 'Head', 'Body', 'Sail', 'Seat'
        )

        begin_str = DEFAULT_IMAGE_TOKEN + "\n"+ ('In the conversation below, you simply answer the category and subcategory name based on what you see ' 
                            'in the image inside a particular region. It maybe a subpart of an object. '
                            'I will give you only one region each time. Your answer should in the format of '
                            'category subcategory. ')
        class_str = 'Categories Containing '+', '.join(CAT_CLASSES)+ '. '
        subclass_str = 'Subcategories Containing ' + ','.join(SUB_CLASSES)
        intro_string = begin_str + class_str + subclass_str + '.\n'
        json_path = "partImagenet_train_format.json"
        dataset_name = "osprey_pretrain"
        mode = "val" if validation else "train"
        image_dir = f"partimagenet/{mode}"
        super().__init__(
            dataset_dir, tokenizer, global_image_encoder, precision, image_size, num_classes_per_sample,
            max_gt_per_img, validation, dataset_name, image_dir, json_path,
            intro_string, random_sampling,  use_mm_start_end,**kwargs)

class PascalPartPretrain(RegionBaseDataset):

    def __init__(self, dataset_dir, tokenizer, global_image_encoder, precision="fp32",
                 image_size=1024, num_classes_per_sample=3, max_gt_per_img=15, validation=False, random_sampling=True, use_mm_start_end=False,**kwargs):

        CAT_CLASSES = ('potted plant', 'aeroplane', 'cow', 'cat', 'bus', 'horse', 'car', 
                    'dog', 'bicycle', 'person', 'bird', 'bottle', 'sheep', 'motorbike')

        SUB_CLASSES = ('eye', 'window', 'cap', 'headlight', 'hand', 'mirror', 'arm', 'plant', 
                    'wheel', 'ear', 'pot', 'foot', 'leg', 'nose', 'body', 'horn', 'handlebar', 
                    'neck', 'license plate', 'paw', 'saddle', 'head', 'muzzle', 'tail', 'wing', 
                    'beak', 'hair', 'torso', 'door', 'mouth')

        begin_str = DEFAULT_IMAGE_TOKEN + "\n"+ ('In the conversation below, you simply answer the category and subcategory name based on what you see ' 
                            'in the image inside a particular region. It maybe a subpart of an object. '
                            'I will give you only one region each time. Your answer should in the format of '
                            'category:subcategory. ')
        class_str = 'Categories Containing '+', '.join(CAT_CLASSES)+ '. '
        subclass_str = 'Subcategories Containing ' + ','.join(SUB_CLASSES)
        intro_string = begin_str + class_str + subclass_str + '.\n'

        json_path = "pascalpart_train.json"
        dataset_name = "osprey_pretrain"
        image_dir = f"pascal_part/VOCdevkit/VOC2010/JPEGImages"
        super().__init__(
            dataset_dir, tokenizer, global_image_encoder, precision, image_size, num_classes_per_sample,
            max_gt_per_img, validation, dataset_name, image_dir, json_path,
            intro_string, random_sampling,  use_mm_start_end, **kwargs)


class RefCOCOPretrain(RegionBaseDataset):

    def __init__(self, dataset_dir, tokenizer, global_image_encoder, precision="fp32",
                 image_size=1024, num_classes_per_sample=3, max_gt_per_img=15, validation=False, random_sampling=True, use_mm_start_end=False,**kwargs):

        intro_string =  DEFAULT_IMAGE_TOKEN + "\n"+ ('I will provide you with only one region ' 
                         'containing only one object, although there may be other ' 
                         'objects present in the image. It is recommended that you ' 
                         "describe the object's relative position with respect to other " 
                         'objects in the image, as well as its position within ' 
                         'the image and its basic attributes.')
        json_path = "finetune_refcoco_train_with_mask.json"
        dataset_name = "osprey_pretrain"
        image_dir = f"coco_2014/train2014"
        super().__init__(
            dataset_dir, tokenizer, global_image_encoder, precision, image_size, num_classes_per_sample,
            max_gt_per_img, validation, dataset_name, image_dir, json_path,
            intro_string, random_sampling,  use_mm_start_end, **kwargs)


    def _filter_images(self, min_size):
        return [i for i, info in enumerate(self.data_infos) if min(info['width'], info['height']) >= min_size]

    def _parse_annotations(self, img_info, ann_info):
        annotations = {'labels': [], 'region_masks': []}

        for ann in ann_info:
            mask = self.annToMask(ann['segmentation'], img_info['height'], img_info['width'])
            # annotations["region_masks"].append(mask)
            if mask.sum().item()<=0:
                if "bbox" in ann:
                    bbox = ann["bbox"]
                    bbox_valid = self._get_valid_bbox(bbox, img_info['width'], img_info['height'])
                    if bbox_valid is not None:
                        annotations["region_masks"].append(self._generate_mask((img_info['height'], img_info['width']), bbox_valid))
                    else:
                        annotations["region_masks"].append(mask)
                else:
                    annotations["region_masks"].append(mask)
            else:
                annotations["region_masks"].append(mask)
            cat = self.coco.loadCats(ann['category_id'])
            annotations['labels'].append(img_info["caption"])
        return annotations

class RefCOCOPPretrain(RegionBaseDataset):
    def __init__(self, dataset_dir, tokenizer, global_image_encoder, precision="fp32",
                 image_size=1024, num_classes_per_sample=3, max_gt_per_img=15, validation=False, random_sampling=True, use_mm_start_end=False, **kwargs):
        intro_string =  DEFAULT_IMAGE_TOKEN + "\n"+ ('I will provide you with only one region ' 
                         'containing only one object, although there may be other ' 
                         'objects present in the image. It is recommended that you ' 
                         "describe the object's relative position with respect to other " 
                         'objects in the image and its basic attibuts, you should not ' 
                         'give its position within the image.')
        json_path = "finetune_refcoco+_train_with_mask.json"
        dataset_name = "osprey_pretrain"
        image_dir = f"coco_2014/train2014"
        super().__init__(
            dataset_dir, tokenizer, global_image_encoder, precision, image_size, num_classes_per_sample,
            max_gt_per_img, validation, dataset_name, image_dir, json_path,
            intro_string, random_sampling,  use_mm_start_end, **kwargs)


    def _filter_images(self, min_size):
        return [i for i, info in enumerate(self.data_infos) if min(info['width'], info['height']) >= min_size]

    def _parse_annotations(self, img_info, ann_info):
        annotations = {'labels': [], 'region_masks': []}

        for ann in ann_info:
            mask = self.annToMask(ann['segmentation'], img_info['height'], img_info['width'])
            if mask.sum().item()<=0:
                if "bbox" in ann:
                    bbox = ann["bbox"]
                    bbox_valid = self._get_valid_bbox(bbox, img_info['width'], img_info['height'])
                    if bbox_valid is not None:
                        annotations["region_masks"].append(self._generate_mask((img_info['height'], img_info['width']), bbox_valid))
                    else:
                        annotations["region_masks"].append(mask)
                else:
                    annotations["region_masks"].append(mask)
            else:
                annotations["region_masks"].append(mask)
            cat = self.coco.loadCats(ann['category_id'])
            annotations['labels'].append(img_info["caption"])
        return annotations
        

