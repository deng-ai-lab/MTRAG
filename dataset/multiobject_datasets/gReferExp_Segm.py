import os
import cv2
import copy
import random
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor
from mtrag.llava import conversation as conversation_lib
from mtrag.SAM.utils.transforms import ResizeLongestSide
from dataset.utils.grefer import G_REFER
from tools.utils import DEFAULT_IMAGE_TOKEN
from dataset.utils.utils import ANSWER_LIST, SEG_QUESTIONS
from dataset.utils.process import preprocess_multimodal, preprocess_v1

class gReferExpSegmDataset(torch.utils.data.Dataset):
    CLASSES = ('object',)
    IMG_MEAN = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    IMG_STD = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    IMG_SIZE = 1024
    IGNORE_LABEL = 255

    def __init__(self, dataset_dir, tokenizer, global_image_encoder, 
                 precision: str = "fp32", image_size: int = 1024, num_classes_per_sample: int = 3,
                 grefer_segm_data="grefcoco", validation=False, split='train',
                 random_sampling=True, inference=False, use_mm_start_end=False, **kwargs):
        self.num_classes_per_sample = num_classes_per_sample
        self.dataset_dir = dataset_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.global_enc_processor = CLIPImageProcessor.from_pretrained(global_image_encoder)
        self.question_templates = SEG_QUESTIONS
        self.answer_list = ANSWER_LIST
        self.begin_str = f"""{DEFAULT_IMAGE_TOKEN}\nThis provides an overview of the picture.\n"""
        self.validation = validation
        self.split = split
        self.random_sampling = random_sampling
        self.use_mm_start_end = use_mm_start_end
        self.base_dir =  os.path.join(dataset_dir, 'referexp_segm')
        self.image_folder = os.path.join(dataset_dir, 'coco_2014/train2014')
        self.initialize_refer_segm_data(grefer_segm_data, inference)

    def initialize_refer_segm_data(self, refer_segm_data, inference=False):
        dataset_name = refer_segm_data
        splitBy = "unc"
        self.refer_api = G_REFER(self.base_dir, dataset_name, splitBy)
        ref_ids_train = self.refer_api.getRefIds(split=self.split)
        images_ids_train = self.refer_api.getImgIds(ref_ids=ref_ids_train)
        refs_train = self.refer_api.loadRefs(ref_ids=ref_ids_train)
        refer_seg_ds = {
            "images": self.load_images(self.refer_api, images_ids_train, self.base_dir, dataset_name, inference=inference),
            "refs": self.refer_api.loadRefs(ref_ids=ref_ids_train),
            "img2refs": self.create_img_to_refs_mapping(refs_train)
        }
        self.samples = len(refer_seg_ds["images"])

        print(f"dataset {dataset_name} (refs {splitBy}) ({self.split} split) has {len(refer_seg_ds['images'])} "
                f"images and {len(refer_seg_ds['refs'])} refs.")
        print(f'\033[92m----SEG-{"Val" if self.validation else "Train"}:'
                f' Loaded gReferExpSeg - {dataset_name} dataset ----\033[0m')
        self.refer_segm_data = refer_seg_ds

    def load_images(self, refer_api, images_ids_train, dataset_dir, dataset_name, inference=False):
        images = []
        loaded_images = refer_api.loadImgs(image_ids=images_ids_train)
        # Limiting images to 1000(optional) for validation
        loaded_images = loaded_images[:1000] if (self.validation and not inference) else loaded_images
        for item in loaded_images:
            item = item.copy()
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

    def create_conversations(self, labels):
        questions = []
        answers = []
        for i, label in enumerate(labels):
            label = label.strip()
            assert len(label.split("||")) == 1
            question_template = random.choice(self.question_templates)
            questions.append(question_template.format(class_name=label.lower()))
            answers.append(random.choice(self.answer_list))

        conversations = dict()
        conversations['conversations'] = []
        for i, (question, answer) in enumerate(zip(questions, answers)):
            if i == 0:
                question = self.begin_str + question
            conversations['conversations'].append({'from': 'human', 'value': question})
            conversations['conversations'].append({'from': 'gpt', 'value': answer})
        return conversations

    def __getitem__(self, idx):
        refer_seg_ds = self.refer_segm_data
        images = refer_seg_ds["images"]
        img2refs = refer_seg_ds["img2refs"]
        image_info = images[idx]
        image_id = image_info["id"]
        refs = img2refs[image_id]
        if len(refs) == 0:
            return self.__getitem__(random.randint(0, len(images) - 1))
        sents_img = []
        masks = []
        for ref in refs:
            m = self.refer_api.getMaskByRef(ref=ref, merge=True)['mask']
            sents = []
            for sent in ref["sentences"]:
                text = sent["sent"]
                sents.append(text)
            if len(sents) == 0:
                continue
            masks.append(m)
            sampled_inds = np.random.choice(
                list(range(len(sents))), size=1, replace=False
                )
            sampled_sents = sents[sampled_inds[0]]
            sents_img.append(sampled_sents)
        
        selected_labels = sents_img
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
        

        masks = np.stack(masks, axis=0)
        masks = torch.from_numpy(masks)
        assert conversations[0].count("[SEG]") == masks.shape[0]
        label = torch.ones(masks.shape[1], masks.shape[2]) * self.IGNORE_LABEL
        # set region_masks to None for segmentation datasets
        region_masks = None

        return (image_path, global_enc_image, grounding_enc_image, region_masks, input_ids[0], targets[0], conversations, masks, label, image_resize)
