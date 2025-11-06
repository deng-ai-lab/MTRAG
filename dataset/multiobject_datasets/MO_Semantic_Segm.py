import os
import cv2
import copy
import glob
import json
import random
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from pycocotools.coco import COCO
from transformers import CLIPImageProcessor
from mtrag.llava import conversation as conversation_lib
from mtrag.SAM.utils.transforms import ResizeLongestSide
from tools.utils import DEFAULT_IMAGE_TOKEN
from dataset.utils.utils import MO_SEG_ANSWER_TEMPLATE, MO_SEG_QUESTIONS
from dataset.utils.process import preprocess_multimodal, preprocess_v1

def load_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def init_ade20k(dataset_dir):
    ade20k_classes = load_json_file("./dataset/utils/ade20k_classes.json")
    ade20k_image_dir = os.path.join(dataset_dir, "ade20k", "images", "training")
    ade20k_images = [os.path.join(ade20k_image_dir, img) for img in os.listdir(ade20k_image_dir) if
                     img.endswith('.jpg')]
    ade20k_labels = [img.replace(".jpg", ".png").replace("images", "annotations") for img in ade20k_images]
    return np.array(ade20k_classes), ade20k_images, ade20k_labels


def init_cocostuff(dataset_dir):
    with open("./dataset/utils/cocostuff_classes.txt") as file:
        cocostuff_classes = [line.strip().split(": ")[-1] for line in file.readlines()[1:]]
    # Annotations
    cocostuff_labels = glob.glob(os.path.join(dataset_dir, "cocostuff", "train2017", "*.png"))
    # Images are obtained from COCO 2017 images
    cocostuff_images = [label.replace(".png", ".jpg").replace("cocostuff", "coco") for label in cocostuff_labels]
    return np.array(cocostuff_classes), cocostuff_images, cocostuff_labels


def init_paco_lvis(dataset_dir):
    paco_lvis_api = COCO(os.path.join(dataset_dir, "paco_lvis", "annotations", "paco_lvis_v1_train.json"))
    all_classes = paco_lvis_api.loadCats(paco_lvis_api.getCatIds())
    class_map_paco_lvis = {}
    for cat in all_classes:
        cat_split = cat["name"].strip().split(":")
        if len(cat_split) == 1:
            name = cat_split[0].split("_(")[0]
        else:
            assert len(cat_split) == 2
            obj, part = cat_split
            obj = obj.split("_(")[0]
            part = part.split("_(")[0]
            name = (obj, part)
        class_map_paco_lvis[cat["id"]] = name

    img_ids = paco_lvis_api.getImgIds()
    return class_map_paco_lvis, img_ids, paco_lvis_api


def init_pascal_part(dataset_dir):
    pascal_part_api = COCO(os.path.join(dataset_dir, "pascal_part", "train.json"))
    all_classes = pascal_part_api.loadCats(pascal_part_api.getCatIds())
    class_map_pascal_part = {}
    for cat in all_classes:
        cat_main, cat_part = cat["name"].strip().split(":")
        name = (cat_main, cat_part)
        class_map_pascal_part[cat["id"]] = name
    img_ids = pascal_part_api.getImgIds()
    return class_map_pascal_part, img_ids, pascal_part_api


def init_mapillary(dataset_dir):
    mapillary_path = os.path.join(dataset_dir, "mapillary")
    mapillary_classes = [cls["readable"].lower() for cls in
                         load_json_file(os.path.join(mapillary_path, "config_v2.0.json"))["labels"]]
    mapillary_labels = sorted(glob.glob(os.path.join(mapillary_path, "training", "v2.0", "labels", "*.png")))
    mapillary_images = [label.replace(".png", ".jpg").replace("v2.0/labels", "images") for label in mapillary_labels]
    return np.array(mapillary_classes), mapillary_images, mapillary_labels


class MOSemanticSegmDataset(torch.utils.data.Dataset):
    CLASSES = ('object',)
    IMG_MEAN = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    IMG_STD = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    IMG_SIZE = 1024
    IGNORE_LABEL = 255
    def __init__(self, dataset_dir, tokenizer, global_image_encoder, epoch_samples=500 * 8 * 4 * 5,
                 precision: str = "fp32", image_size: int = 1024, num_classes_per_sample: int = 8,
                 semantic_segm_data="ade20k||cocostuff||pascal_part||paco_lvis||mapillary", validation=False,
                 random_sampling=True, use_mm_start_end=False, **kwargs):
        self.epoch_samples = epoch_samples
        # self.epoch_samples = epoch_samples if epoch_samples is not None else sum(len(item) for item in self.all_datasets)
        self.num_classes_per_sample = num_classes_per_sample
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.global_enc_processor = CLIPImageProcessor.from_pretrained(global_image_encoder)

        self.question_templates = MO_SEG_QUESTIONS
        self.answer_list = MO_SEG_ANSWER_TEMPLATE
        self.begin_str = f"""{DEFAULT_IMAGE_TOKEN}\nThis provides an overview of the picture.\n"""
        self.validation = validation
        self.random_sampling = random_sampling
        self.use_mm_start_end = use_mm_start_end
        self.data2list = {}
        self.data2classes = {}
        self.dataset_dir = dataset_dir
        self.semantic_seg_ds_list = semantic_segm_data.split("||")
        self.samples = 0
        for ds in self.semantic_seg_ds_list:
            classes, images, labels = eval("init_{}".format(ds))(self.dataset_dir)
            self.data2list[ds] = (images, labels)
            self.samples += len(images) if epoch_samples is None else 0
            self.data2classes[ds] = classes
            print(f'\033[92m----SEG-{"Val" if validation else "Train"}: Loaded SemanticSeg - {ds} dataset ----\033[0m')
        if epoch_samples is not None:
            self.samples = epoch_samples
        if "cocostuff" in self.semantic_seg_ds_list:
            self.cocostuff_class2index = {c: i for i, c in enumerate(self.data2classes["cocostuff"])}
        
    def __len__(self):
        return self.samples

    def _set_len(self, length):
        self.samples = length

    @property
    def modality_lengths(self):
        length_list = []
        for i in range(self.samples):
            length_list.append(100)
        return length_list

    def grounding_enc_processor(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.IMG_MEAN) / self.IMG_STD
        h, w = x.shape[-2:]
        x = F.pad(x, (0, self.IMG_SIZE - w, 0, self.IMG_SIZE - h))
        return x

    def create_conversations(self, labels, dataset_name):
        class_label = ''
        class_label_seg = ''
        class_ids = []
        for i, label in enumerate(labels):
            label = label.strip()
            assert len(label.split("||")) == 1
            class_label += label
            class_label_seg += f"<p> {label} </p> [SEG]"
            if i < len(labels) - 2:
                class_label += ", "
                class_label_seg += ", "
            elif i == len(labels) - 2:
                class_label += " and "
                class_label_seg += " and "
            if dataset_name in ["paco_lvis", "pascal_part"]:
                continue
            class_id = self.data2classes[dataset_name].tolist().index(label)
            class_ids.append(class_id)
        question_template = random.choice(self.question_templates)
        answer_tempalte = random.choice(self.answer_list)
        conversations = dict()
        conversations['conversations'] = []
        question = self.begin_str + question_template.format(class_name=class_label)
        answer = answer_tempalte.format(class_name=class_label_seg)
        conversations['conversations'].append({'from': 'human', 'value': question})
        conversations['conversations'].append({'from': 'gpt', 'value': answer})
        return conversations, class_ids

    def __getitem__(self, idx):
        dataset_idx = random.randint(0, len(self.semantic_seg_ds_list) - 1)
        dataset_name = self.semantic_seg_ds_list[dataset_idx]

        if dataset_name in ["paco_lvis", "pascal_part"]:
            class_map = self.data2classes[dataset_name]
            img_ids, coco_api = self.data2list[dataset_name]
            random_idx = random.randint(0, len(img_ids) - 1)
            img_info = coco_api.loadImgs([img_ids[random_idx]])[0]
            file_name = img_info["file_name"]
            image_path = os.path.join(self.dataset_dir, dataset_name, "VOCdevkit", "VOC2010", "JPEGImages", file_name) \
                          if dataset_name == "pascal_part" else  os.path.join(self.dataset_dir, "coco", file_name)

            annotation_ids = coco_api.getAnnIds(imgIds=img_info["id"])
            annotations = coco_api.loadAnns(annotation_ids)
            if not annotations or len(annotations) < 3:
                return self.__getitem__(0)
            sampled_anns = np.random.choice(annotations, self.num_classes_per_sample, replace=False) if len(
                annotations
                ) >= self.num_classes_per_sample else annotations
            # sampled_anns = np.random.choice(annotations, self.num_classes_per_sample, replace=False)
            selected_labels = []
            for ann in sampled_anns:
                category_id = ann["category_id"]
                sampled_cls = class_map[category_id]
                if isinstance(sampled_cls, tuple):
                    obj, part = sampled_cls
                    name = f"{obj} {part}" if random.random() < 0.5 else f"the {part} of the {obj}"
                else:
                    name = sampled_cls
                selected_labels.append(name)
        elif dataset_name in ["ade20k", "cocostuff", "mapillary"]:
            images, labels = self.data2list[dataset_name]
            idx = idx if (self.validation or not self.random_sampling) else random.randint(0, len(images) - 1)
            image_path, label_path = images[idx], labels[idx]
            label = np.array(Image.open(label_path))
            if dataset_name == "ade20k":
                label = np.where(label == 0, 255, label - 1)
            elif dataset_name == "cocostuff":
                ignored_classes = [index for class_name, index in self.cocostuff_class2index.items() if
                                   "-" in class_name]
                label = np.where(np.isin(label, ignored_classes), 255, label)
            unique_labels = [lbl for lbl in np.unique(label) if lbl != 255]
            if not unique_labels:
                return self.__getitem__(0)

            classes = [self.data2classes[dataset_name][lbl] for lbl in unique_labels]
            if len(classes) < 3:
                return self.__getitem__(0)
            selected_labels = np.random.choice(
                classes, min(len(classes), self.num_classes_per_sample), replace=False
                )
        # Load and process the image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        global_enc_image = self.global_enc_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        # Prepare input for Grounding Image Encoder
        image = self.transform.apply_image(image)
        image_resize = image.shape[:2]
        grounding_enc_image = self.grounding_enc_processor(torch.from_numpy(image).permute(2, 0, 1).contiguous())
        # Generate conversations
        conversations, class_ids = self.create_conversations(selected_labels, dataset_name)
        conv = conversation_lib.default_conversation.copy()
        assert conv.sep_style == conversation_lib.SeparatorStyle.TWO
        sources = preprocess_multimodal(copy.deepcopy([conversations["conversations"]]), use_im_start_end=self.use_mm_start_end)
        input_ids, targets, conversations = preprocess_v1(sources, self.tokenizer)
        assert len(conversations) == 1
        if dataset_name in ["paco_lvis", "pascal_part"]:
            try:
                masks = [coco_api.annToMask(ann) for ann in sampled_anns]
            except Exception as e:
                print(f"Error generating mask: {e}")
                return self.__getitem__(0)

            masks = np.stack(masks, axis=0)
            masks = torch.from_numpy(masks)
            label = torch.ones(masks.shape[1], masks.shape[2]) * self.IGNORE_LABEL
        else:
            label = torch.from_numpy(label).long()
            masks = torch.stack([label == class_id for class_id in class_ids], dim=0)

        assert len(conversations) == 1
        assert conversations[0].count("[SEG]") == masks.shape[0]
        # set region_masks to None for segmentation datasets
        region_masks = None

        return (image_path, global_enc_image, grounding_enc_image, region_masks, input_ids[0], targets[0], conversations, masks, label, image_resize)
