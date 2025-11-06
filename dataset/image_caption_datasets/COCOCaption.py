import os
import cv2
import random
import copy
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from transformers import CLIPImageProcessor
from mtrag.llava import conversation as conversation_lib
from mtrag.SAM.utils.transforms import ResizeLongestSide
from tools.utils import DEFAULT_IMAGE_TOKEN
from dataset.utils.utils import CAPTION_QUESTIONS
from dataset.utils.process import preprocess_multimodal, preprocess_v1
class CocoCapDataset(torch.utils.data.Dataset):
    IMG_MEAN = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    IMG_STD = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    IMG_SIZE = 1024
    IGNORE_LABEL = 255

    def __init__(self, dataset_dir, tokenizer, global_image_encoder, epoch_samples=10000, precision="fp32",
                 image_size=1024, num_classes_per_sample=3, max_gt_per_img=10, validation=False, random_sampling=True, use_mm_start_end=False, **kwargs):
        self.epoch_samples = epoch_samples
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
        self.cfgs = kwargs.get("model_config", None)
        # Defining paths
        mode = "val" if validation else "train"
        self.base_dir = os.path.join(dataset_dir, "coco")
        self.image_folder = os.path.join(dataset_dir, f"coco/{mode}2017")
        json_files = {'validation': "captions_val2017.json", 'training': "captions_train2017.json"}
        annotations_file = os.path.join(self.base_dir, "annotations",
                                        json_files['validation'] if validation else json_files['training'])
        self.data_infos = self._load_annotations(annotations_file)

        self.begin_str = f"""{DEFAULT_IMAGE_TOKEN}\nThis provides an overview of the picture.\n"""
        mode = "Val" if validation else "Train"
        self.grounding = kwargs.get("grounding", True)
        print('\033[92m' + "----CAP-{}: COCO Caption dataset initialized----".format(mode) + '\033[0m')

    def _load_annotations(self, annotation_file):
        self.coco_api = COCO(annotation_file)
        ann_ids = self.coco_api.getAnnIds()
        ann_ids = ann_ids[:1000] if self.validation else ann_ids
        data_infos = []
        for i, id in enumerate(ann_ids):
            annotation = self.coco_api.loadAnns([id])[0]
            data_infos.append(annotation)
        return data_infos

    def _parse_ann_info(self, annotation):
        return {'caption': annotation['caption'].strip()}

    def __getitem__(self, idx):
        annotation = self.data_infos[idx]
        image_info = self.coco_api.loadImgs([annotation['image_id']])[0]
        image_info['filename'] = image_info['file_name'].split('_')[-1]
        # Extract caption from annotation
        caption_info = self._parse_ann_info(annotation)

        data = {
            "image_path": os.path.join(self.image_folder, image_info['filename']),
            "filename": image_info['file_name'],
            "caption": caption_info['caption'],
            }

        processed_data = self.process_data(data)
        return processed_data

    def __len__(self):
        return len(self.data_infos) 
    def modality_lengths(self):
        length_list = []
        for sample in self.data_infos:
            cur_len = len(sample['caption'].strip())
            length_list.append(cur_len)
        return length_list

    def grounding_enc_processor(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.IMG_MEAN) / self.IMG_STD
        h, w = x.shape[-2:]
        x = F.pad(x, (0, self.IMG_SIZE - w, 0, self.IMG_SIZE - h))
        return x

    def create_conversations(self, labels):
        conversations = []
        question = random.choice(CAPTION_QUESTIONS).strip()
        answer = labels
        conversations.append({'from': 'human', 'value': self.begin_str + question})
        conversations.append({'from': 'gpt', 'value': answer})
        return conversations

    def process_data(self, data_item):
        caption = data_item['caption']
        image_path = data_item['image_path']
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Prepare input for Global Image Encoder
        global_enc_image =  self.global_enc_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        if self.grounding:
            image = self.transform.apply_image(image)
            image_resize = image.shape[:2]
            grounding_enc_image = self.grounding_enc_processor(torch.from_numpy(image).permute(2, 0, 1).contiguous())
        else:
            image_resize = None
            grounding_enc_image = torch.zeros(3, 1024, 1024)
        region_masks = None
        masks = None
        conversations = self.create_conversations(caption)
        conv = conversation_lib.default_conversation.copy()
        assert conv.sep_style == conversation_lib.SeparatorStyle.TWO
        sources = preprocess_multimodal(copy.deepcopy([conversations]), use_im_start_end=self.use_mm_start_end)
        input_ids, targets, conversations = preprocess_v1(sources, self.tokenizer)
        assert len(conversations) == 1

        label = None
        return (image_path, global_enc_image, grounding_enc_image, region_masks, input_ids[0], targets[0], conversations, masks, label, image_resize)
