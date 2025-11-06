import os
import cv2
import copy
import random
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask
from transformers import CLIPImageProcessor
from mtrag.llava import conversation as conversation_lib
from mtrag.SAM.utils.transforms import ResizeLongestSide
from dataset.utils.refcoco_refer import REFER
from tools.utils import DEFAULT_IMAGE_TOKEN
from dataset.utils.utils import MO_SEG_QUESTIONS, MO_SEG_ANSWER_TEMPLATE
from dataset.utils.process import preprocess_multimodal, preprocess_v1

class MOReferExpSegmDataset(torch.utils.data.Dataset):
    CLASSES = ('object',)
    IMG_MEAN = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    IMG_STD = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    IMG_SIZE = 1024
    IGNORE_LABEL = 255

    def __init__(self, dataset_dir, tokenizer, global_image_encoder, epoch_samples=500 * 8 * 4 * 5,
                 precision: str = "fp32", image_size: int = 1024, num_classes_per_sample: int = 8,
                 refer_segm_data="refcoco||refcoco+||refcocog||refclef", validation=False, split='train',
                 random_sampling=True, inference=False, use_mm_start_end=False, **kwargs):
        self.num_classes_per_sample = num_classes_per_sample
        self.dataset_dir = dataset_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.global_enc_processor = CLIPImageProcessor.from_pretrained(global_image_encoder)
        self.question_templates = MO_SEG_QUESTIONS
        self.answer_list = MO_SEG_ANSWER_TEMPLATE  
        self.begin_str = f"""{DEFAULT_IMAGE_TOKEN}\nThis provides an overview of the picture.\n"""
        self.validation = validation
        self.split = split
        self.random_sampling = random_sampling
        self.use_mm_start_end = use_mm_start_end
        self.base_dir =  os.path.join(dataset_dir, 'referexp_segm')
        self.image_folder = os.path.join(dataset_dir, 'coco_2014/train2014')
        self.samples = 0
        self.initialize_refer_segm_data(refer_segm_data, inference)
        if epoch_samples is not None:
            self.samples = epoch_samples

    def initialize_refer_segm_data(self, refer_segm_data, inference=False):
        self.refer_seg_ds_list = refer_segm_data.split("||")
        self.refer_segm_data = {}
        for dataset_name in self.refer_seg_ds_list:
            splitBy = "umd" if dataset_name == "refcocog" else "unc"
            refer_api = REFER(self.base_dir, dataset_name, splitBy)
            ref_ids_train = refer_api.getRefIds(split=self.split)
            images_ids_train = refer_api.getImgIds(ref_ids=ref_ids_train)
            refs_train = refer_api.loadRefs(ref_ids=ref_ids_train)
            refer_seg_ds = {
                "images": self.load_images(refer_api, images_ids_train, self.base_dir, dataset_name, inference=inference),
                "annotations": refer_api.Anns,
                "img2refs": self.create_img_to_refs_mapping(refs_train)
            }
            self.samples += len(refer_seg_ds["images"])

            print(f"dataset {dataset_name} (refs {splitBy}) ({self.split} split) has {len(refer_seg_ds['images'])} "
                  f"images and {len(refer_seg_ds['annotations'])} annotations.")
            print(f'\033[92m----SEG-{"Val" if self.validation else "Train"}:'
                  f' Loaded ReferExpSeg - {dataset_name} dataset ----\033[0m')
            self.refer_segm_data[dataset_name] = refer_seg_ds

    def load_images(self, refer_api, images_ids_train, dataset_dir, dataset_name, inference=False):
        images = []
        loaded_images = refer_api.loadImgs(image_ids=images_ids_train)
        # Limiting images to 1000(optional) for validation
        loaded_images = loaded_images[:1000] if (self.validation and not inference) else loaded_images
        for item in loaded_images:
            item = item.copy()
            if dataset_name == 'refclef':
                item["file_name"] = os.path.join(self.base_dir, "images", "saiapr_tc-12", item["file_name"])
            else:
                item["file_name"] = os.path.join(self.image_folder, item["file_name"])
            images.append(item)
        return images

    def create_img_to_refs_mapping(self, refs_train):
        img2refs = {}
        for ref in refs_train:
            img2refs[ref["image_id"]] = img2refs.get(ref["image_id"], []) + [ref, ]
        return img2refs


    def __len__(self):
        return self.samples
    
    @property
    def modality_lengths(self):
        length_list = []
        for i in range(self.samples):
            length_list.append(100)
        return length_list

    def _set_len(self, length):
        self.samples = length

    def grounding_enc_processor(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.IMG_MEAN) / self.IMG_STD
        h, w = x.shape[-2:]
        x = F.pad(x, (0, self.IMG_SIZE - w, 0, self.IMG_SIZE - h))
        return x

    def create_conversations(self, labels):
        class_label = ''
        class_label_seg = ''
        for i in range(len(labels)):
            label = labels[i].strip()
            assert len(label.split("||")) == 1
            class_label += label
            class_label_seg += f"<p> {label} </p> [SEG]"
            if i < len(labels) - 2:
                class_label += ", "
                class_label_seg += ", "
            elif i == len(labels) - 2:
                class_label += " and "
                class_label_seg += " and "
        question_template = random.choice(self.question_templates)
        answer_tempalte = random.choice(self.answer_list)
        conversations = dict()
        conversations['conversations'] = []
        question = self.begin_str + question_template.format(class_name=class_label)
        answer = answer_tempalte.format(class_name=class_label_seg)
        conversations['conversations'].append({'from': 'human', 'value': question})
        conversations['conversations'].append({'from': 'gpt', 'value': answer})
        return conversations

    def __getitem__(self, idx):
        dataset_idx = random.randint(0, len(self.refer_seg_ds_list) - 1)
        dataset_name = self.refer_seg_ds_list[dataset_idx]
        refer_seg_ds = self.refer_segm_data[dataset_name]
        images = refer_seg_ds["images"]
        annotations = refer_seg_ds["annotations"]
        img2refs = refer_seg_ds["img2refs"]
        idx = idx if (self.validation or not self.random_sampling) else random.randint(0, len(images) - 1)
        image_info = images[idx]
        image_id = image_info["id"]
        refs = img2refs[image_id]
        if len(refs) == 0:
            return self.__getitem__(random.randint(0, len(images) - 1))
        sents = []
        ann_ids = []
        for ref in refs:
            for sent in ref["sentences"]:
                text = sent["sent"]
                sents.append(text)
                ann_ids.append(ref["ann_id"])
        if len(sents) >= 3:
            sampled_inds = np.random.choice(
                list(range(len(sents))), size=self.num_classes_per_sample, replace=False
            ) if len(sents) > self.num_classes_per_sample else list(range(len(sents)))
            
        else:
            return self.__getitem__(random.randint(0, len(images) - 1))
        sampled_sents = np.vectorize(sents.__getitem__)(sampled_inds).tolist()
        sampled_ann_ids = [ann_ids[ind] for ind in sampled_inds]
        selected_labels = sampled_sents
        image_path = image_info["file_name"]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        global_enc_image = self.global_enc_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        
        # Prepare input for Grounding Image Encoder
        image = self.transform.apply_image(image)
        image_resize = image.shape[:2]
        grounding_enc_image = self.grounding_enc_processor(torch.from_numpy(image).permute(2, 0, 1).contiguous())
        # Generate conversations
        conversations = self.create_conversations(selected_labels)
        conv = conversation_lib.default_conversation.copy()
        assert conv.sep_style == conversation_lib.SeparatorStyle.TWO
        sources = preprocess_multimodal(copy.deepcopy([conversations["conversations"]]), use_im_start_end=self.use_mm_start_end)
        input_ids, targets, conversations = preprocess_v1(sources, self.tokenizer)
        assert len(conversations) == 1
        flag = False
        masks = []
        for ann_id in sampled_ann_ids:
            if isinstance(ann_id, list):
                flag = True
                if -1 in ann_id:
                    assert len(ann_id) == 1
                    m = np.zeros((image_info["height"], image_info["width"])).astype(
                        np.uint8
                    )
                else:
                    m_final = np.zeros(
                        (image_info["height"], image_info["width"])
                    ).astype(np.uint8)
                    for ann_id_i in ann_id:
                        ann = annotations[ann_id_i]

                        if len(ann["segmentation"]) == 0:
                            m = np.zeros(
                                (image_info["height"], image_info["width"])
                            ).astype(np.uint8)
                        else:
                            if type(ann["segmentation"][0]) == list:  # polygon
                                rle = mask.frPyObjects(
                                    ann["segmentation"], image_info["height"], image_info["width"], )
                            else:
                                rle = ann["segmentation"]
                                for i in range(len(rle)):
                                    if not isinstance(rle[i]["counts"], bytes):
                                        rle[i]["counts"] = rle[i]["counts"].encode()
                            m = mask.decode(rle)
                            m = np.sum(
                                m, axis=2
                            )  # sometimes there are multiple binary map (corresponding to multiple segs)
                            m = m.astype(np.uint8)  # convert to np.uint8
                        m_final = m_final | m
                    m = m_final
                masks.append(m)
                continue

            ann = annotations[ann_id]

            if len(ann["segmentation"]) == 0:
                m = np.zeros((image_info["height"], image_info["width"])).astype(
                    np.uint8
                )
                masks.append(m)
                continue

            if type(ann["segmentation"][0]) == list:  # polygon
                rle = mask.frPyObjects(
                    ann["segmentation"], image_info["height"], image_info["width"]
                )
            else:
                rle = ann["segmentation"]
                for i in range(len(rle)):
                    if not isinstance(rle[i]["counts"], bytes):
                        rle[i]["counts"] = rle[i]["counts"].encode()
            m = mask.decode(rle)
            m = np.sum(m, axis=2)  # sometimes there are multiple binary map (corresponding to multiple segs)
            m = m.astype(np.uint8)  # convert to np.uint8
            masks.append(m)

        masks = np.stack(masks, axis=0)
        masks = torch.from_numpy(masks)
        assert conversations[0].count("[SEG]") == masks.shape[0]
        label = torch.ones(masks.shape[1], masks.shape[2]) * self.IGNORE_LABEL
        # set region_masks to None for segmentation datasets
        region_masks = None

        return (image_path, global_enc_image, grounding_enc_image, region_masks, input_ids[0], targets[0], conversations, masks, label, image_resize)
