import numpy as np
from dataclasses import dataclass, field
import torch
from dataset.pretrain_datasets.MaskPretrain import RefCOCOPPretrain, COCOPretrain, PascalPartPretrain, RefCOCOPretrain, PartImagenetPretrain
from dataset.pretrain_datasets.LLaVAPretrain import LLaVAPretrainDataset
from dataset.pretrain_datasets.PointPretrain import ADE20KPointPretrain, VOCPointPretrain, COCOStuff10kPointPretrain

from dataset.llava_datasets.LLavaInstruct import LLaVAInstructDataset
from dataset.llava_datasets.LLavaVQA import LLaVAVQADataset

from dataset.image_caption_datasets.COCOCaption import CocoCapDataset

from dataset.singleregion_datasets.VGRegion import VisualGenomeRegDataset
from dataset.singleregion_datasets.OspreyRegion import OspreyConversations, OspreyDetailedDescription, OspreyPartLevel
from dataset.singleregion_datasets.RefCOCORegion import RefCocoPRegDataset, RefCocoGRegDataset, RefCocoRegDataset
from dataset.multiregion_datasets.MDVPRegion import HybridMDVPRegDataset

from dataset.multiregion_datasets.FlickrRegion import Flickr30kRegDataset
from dataset.multiregion_datasets.MTRAGMultiRegion import MTRAGMultiturn, MTRAGRelations
from dataset.multiregion_datasets.OspreyMultiRegion import OspreyMultiRegDataset
from dataset.multiregion_datasets.RIOMultiRegion import RIOTrainMultiRegDataset

from dataset.singleobject_datasets.ReferExp_Segm import ReferExpSegmDataset
from dataset.singleobject_datasets.Semantic_Segm import SemanticSegmDataset

from dataset.multiobject_datasets.GCG import GranDfGCGDataset, OpenPsgGCGDataset, RefCOCOgGCGDataset, Flickr30kGCGDataset
from dataset.multiobject_datasets.gReferExp_Segm import gReferExpSegmDataset
from dataset.multiobject_datasets.MO_ReferExp_Segm import MOReferExpSegmDataset
from dataset.multiobject_datasets.MO_Semantic_Segm import MOSemanticSegmDataset
from dataset.multiobject_datasets.RIO_Segm import RIOTrainSegmDataset
from tools.utils import  IGNORE_INDEX
import transformers
import copy
from packaging import version
import tokenizers
from torch.utils.data import ConcatDataset
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')
def pad_to_max_channels(tensor, max_channels):
    current_channels = tensor.shape[0]
    if current_channels < max_channels:
        # Pad to max_channels
        padding = max_channels - current_channels
        # Pad with zeros
        tensor = torch.cat([tensor, torch.zeros(padding, tensor.shape[1], tensor.shape[2])], dim=0)
    return tensor

class HybridDatasetBase(torch.utils.data.Dataset):
    PIXEL_MEAN = torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    PIXEL_STD = torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    IMG_SIZE = 1024
    IGNORE_LABEL = 255

    def __init__(self, dataset_dir, tokenizer, global_image_encoder, dataset, datasets_config,
                 epoch_samples=500 * 8 * 2 * 10, precision="fp32", image_size=1024,
                 num_classes_per_sample=3, sample_rate=None, **kwargs):
        self.dataset_dir = dataset_dir
        self.tokenizer = tokenizer
        self.global_image_encoder = global_image_encoder
        self.dataset = dataset
        self.datasets_config = datasets_config
        self.precision = precision
        self.image_size = image_size
        self.num_classes_per_sample = num_classes_per_sample
        self.kwargs = kwargs
        self.dataset_list = dataset.split("||")
        self.sample_rate = np.array([float(x) for x in sample_rate.split(",")]) if sample_rate is not None else np.array([1] * len(self.dataset_list))
        self.sample_rate = self.sample_rate.astype(np.float64)
        self.sample_rate /= self.sample_rate.sum()
        self.all_datasets = self.create_datasets()
        self.epoch_samples = epoch_samples if epoch_samples is not None else sum(len(item) for item in self.all_datasets)
    def create_datasets(self):
        datasets = []
        for ds in self.dataset_list:
            dataset_cls = self.datasets_config.get(ds)
            if dataset_cls:
                    datasets.append(
                        dataset_cls(
                            dataset_dir=self.dataset_dir, tokenizer=self.tokenizer,  global_image_encoder=self.global_image_encoder,
                            precision=self.precision, image_size=self.image_size, num_classes_per_sample=self.num_classes_per_sample, **self.kwargs)
                        ) 
        return datasets
    
    @property
    def modality_lengths(self):
        length_list = []
        for dataset in self.all_datasets:
            length_list.extend(dataset.modality_lengths)
        return length_list

    def __len__(self):
        return self.epoch_samples
    def __getitem__(self, idx):
        dataset_idx = np.random.choice(len(self.dataset_list), p=self.sample_rate)
        selected_dataset = self.all_datasets[dataset_idx]
        index = np.random.choice(len(selected_dataset))
        data = selected_dataset[index]
        return data

class HybridCapDataset(HybridDatasetBase):
    def __init__(self, dataset_dir, tokenizer, global_image_encoder, epoch_samples=500 * 8 * 2 * 10,
                 precision="fp32", image_size=1024, num_classes_per_sample=3,
                 dataset_cap=None, sample_rate_cap=None, **kwargs):
        if dataset_cap is None:
            dataset = "COCOCap||LLaVA-VQA"
            sample_rate = "1,1"
        else:
            dataset = dataset_cap
            sample_rate = None if sample_rate_cap is None else sample_rate_cap
        datasets_config = {
            "COCOCap": CocoCapDataset,
            "LLaVA-VQA": LLaVAVQADataset
            }
        super().__init__(
            dataset_dir, tokenizer, global_image_encoder, dataset, datasets_config, epoch_samples,
            precision, image_size, num_classes_per_sample, sample_rate, **kwargs
        )


class HybridSegDataset(HybridDatasetBase):
    def __init__(self, dataset_dir, tokenizer, global_image_encoder, epoch_samples=None,
                 precision="fp32", image_size=1024, num_classes_per_sample=3,
                 dataset_seg=None, sample_rate_seg=None, **kwargs):
        if dataset_seg is None:
            dataset = "gReferExpSegm||MOSemanticSegm||MOReferExpSegm||Flickr30kGCG||RefCOCOgGCG||OpenPsgGCG||GranDfGCG||RIOTrainSegm||ReferExpSegm||SemanticSegm"
            sample_rate = "1,1,1,1,1,1,10,1,1,1"
        else:
            dataset = dataset_seg
            sample_rate = None if sample_rate_seg is None else sample_rate_seg
        datasets_config = {
            "VQA":LLaVAVQADataset,
            "gReferExpSegm": gReferExpSegmDataset,
            "MOSemanticSegm":MOSemanticSegmDataset,
            "MOReferExpSegm": MOReferExpSegmDataset,
            "Flickr30kGCG":Flickr30kGCGDataset,
            "RefCOCOgGCG":RefCOCOgGCGDataset,
            "OpenPsgGCG":OpenPsgGCGDataset,
            "GranDfGCG":GranDfGCGDataset,
            "RIOTrainSegm":RIOTrainSegmDataset,
            "ReferExpSegm":ReferExpSegmDataset,
            "SemanticSegm":SemanticSegmDataset
            }
        super().__init__(
            dataset_dir, tokenizer, global_image_encoder, dataset, datasets_config, epoch_samples,
            precision, image_size, num_classes_per_sample, sample_rate, **kwargs
        )

class HybridRegDataset(HybridDatasetBase):
    def __init__(self, dataset_dir, tokenizer, global_image_encoder, epoch_samples=None,
                 precision="fp32", image_size=1024, num_classes_per_sample=3,
                 dataset_reg=None, sample_rate_reg=None, **kwargs):
        if dataset_reg is None:
            dataset="Flickr30kReg||MTRAGMultiturn||MTRAGRelations||RIOMultiReg||OspreyMultiReg||VGReg||RefCocoGReg||RefCocoPReg||RefCocoReg||OspreyConversations||OspreyDetailedDescription||OspreyPartLevel||MDVPReg"
            sample_rate = None
        else:
            dataset = dataset_reg
            sample_rate = None if sample_rate_reg is None else sample_rate_reg
        datasets_config = {
            "Flickr30kReg":Flickr30kRegDataset,
            "MTRAGMultiturn":MTRAGMultiturn,
            "MTRAGRelations":MTRAGRelations,
            "RIOMultiReg":RIOTrainMultiRegDataset,
            "OspreyMultiReg":OspreyMultiRegDataset,
            "HybridCap":HybridCapDataset,
            "VGReg":VisualGenomeRegDataset,
            "RefCocoGReg":RefCocoGRegDataset,
            "RefCocoPReg":RefCocoPRegDataset,
            "RefCocoReg":RefCocoRegDataset,
            "OspreyConversations":OspreyConversations,
            "OspreyDetailedDescription":OspreyDetailedDescription,
            "OspreyPartLevel":OspreyPartLevel,
            "MDVPReg":HybridMDVPRegDataset,
            }
        super().__init__(
            dataset_dir, tokenizer, global_image_encoder, dataset, datasets_config, epoch_samples,
            precision, image_size, num_classes_per_sample, sample_rate, **kwargs
        )

@dataclass
class DataCollatorForMrgarDataset(object):
    tokenizer: transformers.PreTrainedTokenizer
    inference: bool = field(default=False)
    def __call__(self, instances):
        image_path_list, global_enc_image_list, grounding_enc_image_list = [], [], []
        region_masks_list, input_ids_list, targets_list, conversation_list, masks_list = [], [], [], [], []
        label_list, resize_list = [], []
        offset_list, inferences = [0], []
        cnt = 0
        max_channels = max(len(sample[3]) if sample[3] is not None else 0 for sample in instances) + 3
        # Iterating through the batch
        for (image_path, global_enc_image, grounding_enc_image, region_masks, input_ids, targets, conversations, masks, label, resize) in instances:
            image_path_list.append(image_path)
            global_enc_image_list.append(pad_to_max_channels(global_enc_image, max_channels) if max_channels > 3 else global_enc_image)
            grounding_enc_image_list.append(grounding_enc_image)
            region_masks_list.append(torch.tensor(region_masks) if region_masks is not None else None) 
            input_ids_list.append(input_ids)
            targets_list.append(targets)
            conversation_list.extend(conversations)
            masks_list.append([] if masks is None else masks.float())
            label_list.append(label)
            resize_list.append(resize)
            offset_list.append(cnt := cnt + len(conversations))
            inferences.append(self.inference)
        # Padding input ids and targets
        input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids_list,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id)
        targets = torch.nn.utils.rnn.pad_sequence(targets_list,
                                                    batch_first=True,
                                                    padding_value=IGNORE_INDEX)
        attention_masks = input_ids.ne(self.tokenizer.pad_token_id)
        if not inferences[0]:
            truncate_len = self.tokenizer.model_max_length
            if input_ids.shape[1] > truncate_len:
                input_ids, targets, attention_masks = map(
                    lambda x: x[:, :truncate_len], [input_ids, targets, attention_masks]
                    )
        batch = {
            "image_paths": image_path_list,
            "global_enc_images": torch.stack(global_enc_image_list, dim=0),
            "grounding_enc_images": torch.stack(grounding_enc_image_list, dim=0).bfloat16(),
            "region_masks": region_masks_list,
            "input_ids": input_ids,
            "labels": targets,
            "attention_masks": attention_masks,
            "masks_list": masks_list,
            "label_list": label_list,
            "resize_list": resize_list,
            "offset": torch.LongTensor(offset_list),
            "inference": inferences[0],
            "conversation_list": conversation_list
        }
        return batch
def make_multitask_data_module(data_args):
    dataset_configs = data_args.dataset_config.split("||")
    dataset_config = dataset_configs[0] if len(dataset_configs) == 1 else dataset_configs
    train_dataset = build_mtrag_dataset(dataset_config, data_args=data_args)
    data_collator = DataCollatorForMrgarDataset(tokenizer=data_args.tokenizer, inference=data_args.inference)
    
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def build_mtrag_dataset(dataset_config,
                  data_args=None,
                  **kwargs):
    if isinstance(dataset_config, list):
        datasets = []
        for cfg in dataset_config:
            temp_dataset = build_mtrag_dataset(cfg, data_args=data_args, **kwargs)
            datasets.append(temp_dataset)
        for dataset in datasets:
            print(type(dataset), f'len = {len(dataset)}')
        return ConcatDataset(datasets)
    dataset_type = dataset_config
    params_dict = copy.deepcopy(data_args.__dict__)
    if dataset_type == "Stage1": # LLaVA pretrain
        dataset = LLaVAPretrainDataset(**params_dict)
    elif dataset_type =="COCOPretrain":
        dataset = COCOPretrain(**params_dict)
    elif dataset_type =="RefCOCOPPretrain":
        dataset = RefCOCOPPretrain(**params_dict)
    elif dataset_type =="PascalPartPretrain":
        dataset = PascalPartPretrain(**params_dict)
    elif dataset_type =="RefCOCOPretrain":
        dataset = RefCOCOPretrain(**params_dict)
    elif dataset_type =="PartImagenetPretrain":
        dataset = PartImagenetPretrain(**params_dict)
    elif dataset_type =="ADE20KPoint":
        dataset = ADE20KPointPretrain(**params_dict)
    elif dataset_type =="COCOStuff10kPoint":
        dataset = COCOStuff10kPointPretrain(**params_dict)
    elif dataset_type =="VOCPoint":
        dataset = VOCPointPretrain(**params_dict)
    elif dataset_type == "Stage3": # LLaVA Instruct
        dataset = LLaVAInstructDataset(**params_dict)
    else:
        raise NotImplementedError  

    return dataset  


class ConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super().__init__(datasets)

    @property
    def modality_lengths(self):
        length_list = []
        for dataset in self.datasets:
            length_list.extend(dataset.modality_lengths)
        return length_list

def custom_collate_fn(batch, tokenizer=None, inference=False, local_rank=-1):
    # Initializing lists and counters
    image_path_list, global_enc_image_list, grounding_enc_image_list = [], [], []
    region_masks_list, input_ids_list, targets_list,conversation_list, masks_list = [], [], [], [], []
    label_list, resize_list = [], []
    offset_list, inferences = [0], []
    cnt = 0
    max_channels = max(len(sample[3]) if sample[3] is not None else 0 for sample in batch) + 3
    # Iterating through the batch
    for (image_path, global_enc_image, grounding_enc_image, region_masks, input_ids, targets, conversations, masks, label, resize) in batch:
        image_path_list.append(image_path)
        global_enc_image_list.append(pad_to_max_channels(global_enc_image,max_channels))
        grounding_enc_image_list.append(grounding_enc_image)
        region_masks_list.append(torch.tensor(region_masks) if region_masks is not None else None) 
        input_ids_list.append(input_ids)
        targets_list.append(targets)
        conversation_list.extend(conversations)
        masks_list.append([] if masks is None else masks.float())
        label_list.append(label)
        resize_list.append(resize)
        offset_list.append(cnt := cnt + len(conversations))
        inferences.append(inference)
    # Padding input ids and targets
    input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list,
            batch_first=True,
            padding_value=tokenizer.pad_token_id)
    targets = torch.nn.utils.rnn.pad_sequence(targets_list,
                                                batch_first=True,
                                                padding_value=IGNORE_INDEX)
    attention_masks = input_ids.ne(tokenizer.pad_token_id)
    if not inferences[0]:
        truncate_len = tokenizer.model_max_length
        if input_ids.shape[1] > truncate_len:
            input_ids, targets, attention_masks = map(
                lambda x: x[:, :truncate_len], [input_ids, targets, attention_masks]
                )

    return {
        "image_paths": image_path_list,
        "global_enc_images": torch.stack(global_enc_image_list, dim=0),
        "grounding_enc_images":torch.stack(grounding_enc_image_list, dim=0).bfloat16(),
        "region_masks": region_masks_list,
        "input_ids": input_ids,
        "labels": targets,
        "attention_masks": attention_masks,
        "masks_list": masks_list,
        "label_list": label_list,
        "resize_list": resize_list,
        "offset": torch.LongTensor(offset_list),
        "inference": inferences[0],
        "conversation_list": conversation_list
    }