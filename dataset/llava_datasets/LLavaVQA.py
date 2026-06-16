import os
import json
import copy
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from transformers import CLIPImageProcessor
from mtrag.llava import conversation as conversation_lib
from mtrag.SAM.utils.transforms import ResizeLongestSide
from dataset.utils.process import preprocess_multimodal, preprocess_v1

class LLaVAVQADataset(torch.utils.data.Dataset):
    IMG_MEAN = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    IMG_STD = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    IMG_SIZE = 1024
    IGNORE_LABEL = 255
    def __init__(self, dataset_dir, tokenizer, global_image_encoder, epoch_samples=10000, precision="fp32",
                 image_size=1024, num_classes_per_sample=3, validation=False, random_sampling=True, use_mm_start_end=False, **kwargs):

        self.dataset_dir = dataset_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.global_enc_processor = CLIPImageProcessor.from_pretrained(global_image_encoder)
        self.epoch_samples = epoch_samples
        self.num_classes_per_sample = num_classes_per_sample
        self.validation = validation
        self.random_sampling = random_sampling
        self.use_mm_start_end = use_mm_start_end

        self.cfgs = kwargs

        # Defining paths
        mode = "val" if validation else "train"
        self.base_dir = os.path.join(dataset_dir, "llava_dataset")
        self.image_folder = os.path.join(dataset_dir, "coco/train2017")
        annotations_file = os.path.join(self.base_dir, "llava_instruct_150k.json")
        self.data_infos = self._load_annotations(annotations_file)
        self.grounding = kwargs.get("grounding", True)
        print('\033[92m' + "----CAP----: LLaVA-Instruct VQA dataset initialized----".format(mode) + '\033[0m')
    @property
    def lengths(self):
        length_list = []
        for sample in self.data_infos:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.data_infos:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list
    def _load_annotations(self, ann_file):
        with open(ann_file, 'r') as f:
            data_infos = json.load(f)
        data_infos = data_infos[0: 1000] if self.validation else data_infos
        return data_infos

    def __len__(self):
        return len(self.data_infos)

    def grounding_enc_processor(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.IMG_MEAN) / self.IMG_STD
        h, w = x.shape[-2:]
        x = F.pad(x, (0, self.IMG_SIZE - w, 0, self.IMG_SIZE - h))
        return x

    def process(self, conv_ann, has_image=True):
        conv = conversation_lib.default_conversation.copy()
        assert conv.sep_style == conversation_lib.SeparatorStyle.TWO
        sources = preprocess_multimodal(conv_ann, use_im_start_end=self.use_mm_start_end)
        input_ids, targets, conversations = preprocess_v1(sources, self.tokenizer, has_image=has_image)
        assert len(conversations) == 1
        return input_ids[0], targets[0], conversations

    def __getitem__(self, idx):
        ann_info = self.data_infos[idx]
        conv_ann = ann_info["conversations"]
        input_ids, targets, conversations = self.process(copy.deepcopy([conv_ann]),has_image=("image" in ann_info))
        image_path = os.path.join(self.image_folder, ann_info["image"])
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)
        global_enc_image = self.global_enc_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        if self.grounding:
            image = self.transform.apply_image(image)
            image_resize = image.shape[:2]
            grounding_enc_image = self.grounding_enc_processor(torch.from_numpy(image).permute(2, 0, 1).contiguous())
        else:
            image_resize = None
            grounding_enc_image = torch.zeros(3, 1024, 1024)
        region_masks = None
        masks = None
        label = None
        
        return (image_path, global_enc_image, grounding_enc_image, region_masks, input_ids, targets, conversations, masks, label, image_resize)
    
