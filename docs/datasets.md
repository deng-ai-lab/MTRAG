# Prepare Dataset 🚀
This guide outlines the datasets required for opensource fine-tuning of MTRAG. 
Overall, they must be arranged in the following format:
```
├── ade20k
│   ├── annotations
│   │   ├── training
│   │   │   ├── ADE_train_00000001.png
│   │   │   ├── ADE_train_00000002.png
│   ├── images
│   │   ├── training
│   │   │   ├── ADE_train_00000001.jpg
│   │   │   ├── ADE_train_00000002.jpg
│
├── coco
│   ├── train2017
│   │   ├── 000000000009.jpg
│   │   ├── 000000000025.jpg
│   ├── annotations
│   │   ├── instances_train2017.json
│   │   ├── captions_train2017.json
│   │   ├── captions_val2017.json
│   ├── val2017
│   │   ├── 000000000139.jpg
│   │   ├── 000000000285.jpg
│
├── coco_2014
│   ├── train2014
│   │   ├── COCO_train2014_000000000009.jpg
│   │   ├── COCO_train2014_000000000025.jpg
│   ├── val2014
│   │   ├── COCO_val2014_000000000042.jpg
│   │   ├── COCO_val2014_000000000073.jpg
│
├── coco_stuff
│   │   ├── train2017
│   │   │   ├── 000000000009.png
│   │   │   ├── 000000000025.png
│
├── flikcr_30k
│   ├── 1000092795.jpg
│   ├── 10002456.jpg
│
├── gqa
│   ├── images
│   │   ├── 1.jpg
│
├── GranDf
│   ├── annotations
│   │   ├── train
│   │   │   ├── GranDf_HA_GCG_train.json
│   │   │   ├── OpenPsgGCG_train.json
│   │   │   ├── OpenPsgGCG_val.json
│   │   │   ├── RefCOCOg_GCG_train.json
│   │   │   ├── RefCOCOg_GCG_val.json
│   │   │   ├── flickr_mergedGT_GCG_train.json
│   │   │   ├── flickr_mergedGT_GCG_val.json
│   │   ├── val_test
│   │   │   ├── test_gcg_coco_caption_gt.json
│   │   │   ├── test_gcg_coco_mask_gt.json
│   │   │   ├── val_gcg_coco_caption_gt.json
│   │   │   ├── val_gcg_coco_mask_gt.json
│
├── GranDf_HA_images
│   ├── train
│   │   ├── sa_10010541.jpg
│   │   ├── sa_10014079.jpg
│   ├── val_test
│   │   ├── sa_10010541.jpg
│   │   ├── sa_10014079.jpg
│
├── llava_dataset
│   ├── blip_laion_cc_sbu_558k.json
│   ├── llava_instruct_150k.json
│   ├── llava_v1_5_mix665k.json
│   ├── images
│   │   ├── 00000
│
├── mapillary
│   ├── config_v2.0.json
│   ├── training
│   │   ├── v2.0
│   │   │   ├── labels
│   │   │   │   ├── 0035fkbjWljhaftpVM37-g.png
│   │   │   │   ├── 00qclUcInksIYnm19b1Xfw.png
│   │   ├── images
│   │   │   ├── 0035fkbjWljhaftpVM37-g.jpg
│   │   │   ├── 00qclUcInksIYnm19b1Xfw.jpg
│
├── mdvp_instruct
│   ├── detailed_caption
│   │   ├── gpt4v_ade20k_detailed_caption_point.json
│   │   ├── gpt4v_cocostuff_10k_detailed_caption_point.json
│   │   ├── gpt4v_cocostuff_164k_detailed_caption_point.json
│   │   ├── gpt4v_lvis_detailed_caption_point.json
│   │   ├── gpt4v_pascal_context_detailed_caption_point.json
│   │   ├── gpt4v_vg_detailed_caption_point.json
│
├── mdvp_pretrain
│   ├── ADE20K_point2label.json
│   ├── cocostuff_10k_point2label.json
│   ├── cocostuff_164k_point2label.json
│   ├── LVIS_point2label.json
│   ├── VG_point2label.json
│   ├── VOC_point2label.json
│
├── mtrag_instruct
│   ├── mtrag_multiturn.json
│   ├── mtrag_relationalreasoning.json
│
├── ocr_vqa
│   ├── images
│   │   ├── 000195850X.jpg
│   ├── dataset.json
│
├── osprey_instruct
│   ├── osprey_conversation.json
│   ├── osprey_detail_description.json
│   ├── osprey_part_level.json
│
├── osprey_pretrain
│   ├── finetune_refcoco+_train_with_mask.json
│   ├── finetune_refcoco_train_with_mask.json
│   ├── partImagenet_train_format.json
│   ├── pascalpart_train.json
│   ├── vg_train_with_mask.json
│
├── paco_lvis
│   ├── annotations
│   │   ├── paco_lvis_v1_train.json
│
├── partimagenet
│   ├── train
│   │   ├── n01440764
│   ├── test
│   │   ├── n01491361
│   ├── val
│   │   ├── n01484850
│   ├── train.json
│   ├── test.json
│   ├── val.json
│
├── pascal_part
│   ├── train.json
│   ├── VOCdevkit
│   │   │   ├── VOC2010
│   │   │   │   ├── JPEGImages
│   │   │   │   │   ├── 2007_000027.jpg
│   │   │   │   │   ├── 2007_000032.jpg
│
├── Reason
│   ├── RIO
│   │   ├── RIO_common_all_thing.json
│   │   ├── RIO_train_all_thing.json
│   │   ├── RIO_uncommon_all_thing.json
│
├── referexp_segm
│   ├── grefcoco
│   ├── refcoco
│   ├── refcoco+
│   ├── refcocog
│   ├── refclef
│   ├── images
│   │   ├── saiapr_tc-12
│   │   │   ├── 00
│   │   │   ├── 01
│
├── region_cap
│   ├── finetune_refcoco_train_with_mask.json
│   ├── finetune_refcocog_train_with_mask.json
│   ├── finetune_refcocog_val_with_mask.json
│   ├── finetune_refcoco+_train_with_mask.json
│   ├── final_flickr_mergedGT_train.json
│   ├── vg_train_caption.json
│   ├── vg_test_caption.json
│
├── textvqa
│   ├── train_images
│   │   ├── 0000599864fd15b3.jpg
│
├── vg
│   ├── VG_100K
│   ├── VG_100K_2
```


## (1) Stage1 Pretraining Datasets
For stage1 pretraining, we following LLaVA-v1.5 to use the filtered CC3M dataset.

Download links and structure:
- Annotations: [blip_laion_cc_sbu_558k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/blob/main/blip_laion_cc_sbu_558k.json) 
- images: [`images.zip`](https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K/blob/main/images.zip) 

Download the data from the source links, and arrange as follows:
```
├── llava_dataset
│   ├── blip_laion_cc_sbu_558k.json
│   ├── images
│   │   ├── 00000
```
## (2) Stage2 Pretraining Datasets
For stage2 pretraining, we curate a collection of short-text and free-form visual prompt pairs from several publicly available datasets: (i)mask-annotated samples from COCO; (ii) object-level and part-level mask-prompt pretraining data from Osprey; (iii) point-prompt pretraining data from MDVP.
Download links and structure:
- COCO: [Annotations](http://images.cocodataset.org/annotations/annotations_trainval2017.zip), [images](http://images.cocodataset.org/zips/train2017.zip)
- Osprey Pretraining Datasets:
  - pascal_part: [train.json](https://huggingface.co/datasets/sunshine-lwt/Osprey-TrainingData/resolve/main/pascalpart_train.json?download=true), [VOCdevkit](http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar).
  - partImagenet: [train_format.json](https://huggingface.co/datasets/sunshine-lwt/Osprey-TrainingData/resolve/main/partImagenet_train_format.json?download=true), [PartImageNet_OOD](https://drive.google.com/file/d/19kA8-pAxssQI0GD5H8y8KESGaALwChtx/view?usp=sharing).
  - refcocos: [refcoco](https://huggingface.co/datasets/sunshine-lwt/Osprey-TrainingData/resolve/main/finetune_refcoco_train_with_mask.json?download=true), [refcoco+](https://huggingface.co/datasets/sunshine-lwt/Osprey-TrainingData/resolve/main/finetune_refcoco%2B_train_with_mask.json?download=true).
  - RefCOCO images: `coco_2014` - COCO-2014 ([train2014](http://images.cocodataset.org/zips/train2014.zip))
- MDVP Pretraining Datasets: A subset of the point2label pretraining dataset [Annotations](https://huggingface.co/datasets/Afeng-x/Draw-and-Understand/tree/main/stage_1_pre-training/point2label)
  - coco: [train2017](http://images.cocodataset.org/zips/train2017.zip
  - VisualGenome: [part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip), [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip)
  - [ADE20K](http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip)
  - [COCO-Stuff](http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip)
  - [PASCAL-Part](https://www.mapillary.com/dataset/vistas)
Download the data from the source links, and arrange as follows:
```
├── ade20k
│   ├── annotations
│   │   ├── training
│   │   │   ├── ADE_train_00000001.png
│   │   │   ├── ADE_train_00000002.png
│   ├── images
│   │   ├── training
│   │   │   ├── ADE_train_00000001.jpg
│   │   │   ├── ADE_train_00000002.jpg
├── partimagenet
│   ├── train
│   │   ├── n01440764
│   ├── test
│   │   ├── n01491361
│   ├── val
│   │   ├── n01484850
│   ├── train.json
│   ├── test.json
│   ├── val.json
├── pascal_part
│   ├── train.json
│   ├── VOCdevkit
│   │   │   ├── VOC2010
│   │   │   │   ├── JPEGImages
│   │   │   │   │   ├── 2007_000027.jpg
│   │   │   │   │   ├── 2007_000032.jpg
├── vg
│   ├── VG_100K
│   ├── VG_100K_2
├── coco
│   ├── train2017
│   │   ├── 000000000009.jpg
│   │   ├── 000000000025.jpg
│   ├── annotations
│   │   ├── instances_train2017.json
├── coco_stuff
│   │   ├── train2017
│   │   │   ├── 000000000009.png
│   │   │   ├── 000000000025.png
├── osprey_pretrain
│   ├── finetune_refcoco+_train_with_mask.json
│   ├── finetune_refcoco_train_with_mask.json
│   ├── partImagenet_train_format.json
│   ├── pascalpart_train.json
├── mdvp_pretrain
│   ├── ADE20K_point2label.json
│   ├── cocostuff_10k_point2label.json
│   ├── cocostuff_164k_point2label.json
│   ├── LVIS_point2label.json
│   ├── VG_point2label.json
│   ├── VOC_point2label.json
├── coco_2014
│   ├── train2014
│   │   ├── COCO_train2014_000000000009.jpg
│   │   ├── COCO_train2014_000000000025.jpg
│   ├── val2014
│   │   ├── COCO_val2014_000000000042.jpg
│   │   ├── COCO_val2014_000000000073.jpg
```

## (3) Stage3 Finetuning Datasets
For stage3 finetuning, we following LLaVA-v1.5 to use the LLaVA-v1.5-
mix665k.
Download links and structure:
- Annotations: [llava_v1_5_mix665k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json) 
- images: 
  - COCO: [train2017](http://images.cocodataset.org/zips/train2017.zip)
  - GQA: [images](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip)
  - OCR-VQA: [download script](https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_?usp=sharing), **we save all files as `.jpg`**
  - TextVQA: [train_val_images](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip)
  - VisualGenome: [part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip), [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip)
Download the data from the source links, and arrange as follows:
```
├── llava_dataset
│   ├── llava_v1_5_mix665k.json
├── gqa
│   ├── images
│   │   ├── 1.jpg
├── ocr_vqa
│   ├── images
│   │   ├── 000195850X.jpg
│   ├── dataset.json
├── textvqa
│   ├── train_images
│   │   ├── 0000599864fd15b3.jpg
├── vg
│   ├── VG_100K
│   ├── VG_100K_2
```

## (4) Stage4 Finetuning Datasets

Stage4 Finetuning Datasets encompass tasks like Single-Region Referring (SRR), Single Object Grounding (SOG), Multi-Region Referring (MRR), Multi-Object Grounding (MOG), and Image-Level
Understanding (ILU).

### 1) Single-Region Referring (SRR) Datasets
For single-region referring , we use five open-source datasets with region(bbox, mask) annotations: RefCOCO, RefCOCOg, RefCOCO+, Visual Genome(V1.2) and Osprey.


Download links and structure:
- Annotations - mdetr_annotations: [Download](https://drive.google.com/file/d/1gvH5ToNtmIr3qz7C9lNi_fDmElwAANsI/view?usp=drive_link)
- Visual Genome: [train.json](https://datarelease.blob.core.windows.net/grit/VG_preprocessed_annotations/train.json), [test_caption.json](https://drive.google.com/file/d/1zF3UGHU1rvgTujinqJ-hZtrCBVsfsuel/view?usp=sharing) [images](https://nlp.stanford.edu/data/gqa/images.zip)
- Osprey: Download the train images from the [Osprey-724K]((https://huggingface.co/datasets/AntGroup-MI/Osprey-724K)).
- RefCOCO images: `coco_2014` - COCO-2014 ([train2014](http://images.cocodataset.org/zips/train2014.zip))
Download the data from the source links, and arrange as follows:

```
├── vg
│   ├── VG_100K
│   ├── VG_100K_2
├── region_cap
│   ├── finetune_refcoco_train_with_mask.json
│   ├── finetune_refcocog_train_with_mask.json
│   ├── finetune_refcocog_val_with_mask.json
│   ├── finetune_refcoco+_train_with_mask.json
│   ├── final_flickr_mergedGT_train.json
│   ├── vg_train_caption.json
│   ├── vg_test_caption.json
├── osprey_instruct
│   ├── osprey_conversation.json
│   ├── osprey_detail_description.json
│   ├── osprey_part_level.json
├── coco_2014
│   ├── train2014
│   │   ├── COCO_train2014_000000000009.jpg
│   │   ├── COCO_train2014_000000000025.jpg
```
### 2) Single Object Grounding (SOG) Datasets
For single-object grounding, we use open-source semantic segmentation and referring expression comprehension datasets.

For semantic segmentation, we use five open-source datasets providing segmentation masks and semantic class labels: - ADE20K, COCO-Stuff, PASCAL-Part, PACO-LVIS, and Mapillary. 

Download links and structure:
- [ADE20K](http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip)
- [COCO-Stuff](http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip)
- [PASCAL-Part](https://www.mapillary.com/dataset/vistas)
- [PACO-LVIS](https://github.com/facebookresearch/paco/tree/main#dataset-setup)
- [Mapillary](https://github.com/facebookresearch/VLPart/tree/main/datasets#pascal-part)
- COCO images: `coco_2017` - COCO-2017 ([train2017](http://images.cocodataset.org/zips/train2017.zip))

Download and arrange as shown in the directory structure below.
```
├── ade20k
│   ├── annotations
│   │   ├── training
│   │   │   ├── ADE_train_00000001.png
│   │   │   ├── ADE_train_00000002.png
│   ├── images
│   │   ├── training
│   │   │   ├── ADE_train_00000001.jpg
│   │   │   ├── ADE_train_00000002.jpg
├── coco_stuff
│   │   ├── train2017
│   │   │   ├── 000000000009.png
│   │   │   ├── 000000000025.png
├── mapillary
│   ├── config_v2.0.json
│   ├── training
│   │   ├── v2.0
│   │   │   ├── labels
│   │   │   │   ├── 0035fkbjWljhaftpVM37-g.png
│   │   │   │   ├── 00qclUcInksIYnm19b1Xfw.png
│   │   ├── images
│   │   │   ├── 0035fkbjWljhaftpVM37-g.jpg
│   │   │   ├── 00qclUcInksIYnm19b1Xfw.jpg
├── paco_lvis
│   ├── annotations
│   │   ├── paco_lvis_v1_train.json
├── pascal_part
│   ├── train.json
│   ├── VOCdevkit
│   │   │   ├── VOC2010
│   │   │   │   ├── JPEGImages
│   │   │   │   │   ├── 2007_000027.jpg
│   │   │   │   │   ├── 2007_000032.jpg
```

For Referring Expression segmentation - we use COCO referring expression comprehension datasets: gRefCOCO, RefCOCO, RefCOCO+, RefCOCOg, and RefCLEF.

Download links and structure:
- [RefCOCO](https://web.archive.org/web/20220413011718/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip)
- [RefCOCO+](https://web.archive.org/web/20220413011656/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip)
- [RefCOCOg](https://web.archive.org/web/20220413012904/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip)
- [RefCLEF](https://web.archive.org/web/20220413011817/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refclef.zip)
- RefCOCO images: `coco_2014` - COCO-2014 ([train2014](http://images.cocodataset.org/zips/train2014.zip))
- For RefCLEF, you need images `[saiapr_tc-12](https://web.archive.org/web/20220515000000/http://bvisionweb1.cs.unc.edu/licheng/referit/data/images/saiapr_tc-12.zip)`

Download the data from the source links, and arrange as follows:
```
├── referexp_segm
│   ├── grefcoco
│   ├── refcoco
│   ├── refcoco+
│   ├── refcocog
│   ├── refclef
│   ├── images
│   │   ├── saiapr_tc-12
│   │   │   ├── 00
│   │   │   ├── 01
├── coco_2014
│   ├── train2014
│   │   ├── COCO_train2014_000000000009.jpg
│   │   ├── COCO_train2014_000000000025.jpg
```
### 3) Multi-Region Referring Datasets
For multi-region referring, we first aggregated multiple region-level descriptions into single instruction-response pairs using samples from Osprey, Flickr30K, MDVP, and RIO. In addition, we synthesized two specialized instruction datasets based on RefCOCO: a multi-turn conversational dataset (MTRAG-MT) and a relational reasoning dataset (MTRAG-RR).

Download links and structure:
- RIO: [Annotations](https://drive.google.com/drive/folders/1IAvh8tBGS3WWgV4SbVoqhwCkmyoSFffh)
- Flickr30k: Download the train images from the [Flickr30K webpage](https://shannon.cs.illinois.edu/DenotationGraph/) or use download from the following [link](https://drive.google.com/file/d/1iomUn-Ht0OBfieMuyoVqEFj5PEmXfQ0U/view?usp=drive_link).
- MTRAG-MT and MTRAG-RR: [Download](https://drive.google.com/file/d/1Kt7w4XlN90oJfB3NzWehHaKtbTvL793r/view?usp=sharing)
- COCO images: coco_2014 - COCO-2014 (train2014)([train2014](http://images.cocodataset.org/zips/train2014.zip))
- MDVP: A subset of the Detailed_caption dataset [Annotations](https://huggingface.co/datasets/Afeng-x/Draw-and-Understand/tree/main/stage_2_fine-tuning/MDVP-Data/detailed_caption)
  - coco: [train2017](http://images.cocodataset.org/zips/train2017.zip
  - VisualGenome: [part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip), [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip)
  - [ADE20K](http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip)
  - [COCO-Stuff](http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip)
  - [PASCAL-Part](https://www.mapillary.com/dataset/vistas)
  -
```
├── ade20k
│   ├── annotations
│   │   ├── training
│   │   │   ├── ADE_train_00000001.png
│   │   │   ├── ADE_train_00000002.png
│   ├── images
│   │   ├── training
│   │   │   ├── ADE_train_00000001.jpg
│   │   │   ├── ADE_train_00000002.jpg
├── partimagenet
│   ├── train
│   │   ├── n01440764
│   ├── test
│   │   ├── n01491361
│   ├── val
│   │   ├── n01484850
│   ├── train.json
│   ├── test.json
│   ├── val.json
├── pascal_part
│   ├── train.json
│   ├── VOCdevkit
│   │   │   ├── VOC2010
│   │   │   │   ├── JPEGImages
│   │   │   │   │   ├── 2007_000027.jpg
│   │   │   │   │   ├── 2007_000032.jpg
├── vg
│   ├── VG_100K
│   ├── VG_100K_2
├── coco_stuff
│   │   ├── train2017
│   │   │   ├── 000000000009.png
│   │   │   ├── 000000000025.png
├── mdvp_instruct
│   ├── detailed_caption
│   │   ├── gpt4v_ade20k_detailed_caption_point.json
│   │   ├── gpt4v_cocostuff_10k_detailed_caption_point.json
│   │   ├── gpt4v_cocostuff_164k_detailed_caption_point.json
│   │   ├── gpt4v_lvis_detailed_caption_point.json
│   │   ├── gpt4v_pascal_context_detailed_caption_point.json
│   │   ├── gpt4v_vg_detailed_caption_point.json
├── mtrag_instruct
│   ├── mtrag_multiturn.json
│   ├── mtrag_relationalreasoning.json
├── flikcr_30k
│   ├── train
│   │   ├── 1000092795.jpg
│   │   ├── 10002456.jpg
├── Reason
│   ├── RIO
│   │   ├── RIO_common_all_thing.json
│   │   ├── RIO_train_all_thing.json
│   │   ├── RIO_uncommon_all_thing.json
```

### 4) Multi-Object Grounding Datasets
For multi-object grounding, we first extend the SOG-style question-answer templates to accommodate multiple target objects within a single instruction, thereby converting open-source semantic segmentation and referring expression comprehension datasets into MOG-style instruction data. In addition, we incorporate segmentation datasets containing multiple objects, including gRefCOCO, RIO, and GranDf.

Download links and structure:
- Grandf: [Annotations](https://grounding-anything.com/GranD-f), [GranDf_HA_images](https://drive.google.com/file/d/1abdxVhrbNQhjJQ8eAcuPrOUBzhGaFsF_/view?usp=drive_link)
  - Other necessary datasets: 
    - Open-PSG GCG: `coco_2017` - COCO-2017 ([train2017](http://images.cocodataset.org/zips/train2017.zip))
    - RefCOCO-g GCG: `coco_2014` - COCO-2014 ([train2014](http://images.cocodataset.org/zips/train2014.zip))
    - Flickr-30k GCG: `flikcr_30k` - flikcr_30k (train) - Download the train images from the [Flickr30K webpage](https://shannon.cs.illinois.edu/DenotationGraph/) or use download from the following [link](https://drive.google.com/file/d/1iomUn-Ht0OBfieMuyoVqEFj5PEmXfQ0U/view?usp=drive_link).
- RIO: [Annotations](https://drive.google.com/drive/folders/1IAvh8tBGS3WWgV4SbVoqhwCkmyoSFffh)
- gRefCOCO: [gRefCOCO](https://entuedu-my.sharepoint.com/:f:/g/personal/liuc0058_e_ntu_edu_sg/Ep74cipLWvRPpkxF9q2M8-gBINURc6YmwwG2fq1nqg-j5Q)
- COCO images: coco_2014 - COCO-2014 (train2014)([train2014](http://images.cocodataset.org/zips/train2014.zip))

Download and arrange as shown in the directory structure below.
```
├── ade20k
│   ├── annotations
│   │   ├── training
│   │   │   ├── ADE_train_00000001.png
│   │   │   ├── ADE_train_00000002.png
│   ├── images
│   │   ├── training
│   │   │   ├── ADE_train_00000001.jpg
│   │   │   ├── ADE_train_00000002.jpg
├── coco_stuff
│   │   ├── train2017
│   │   │   ├── 000000000009.png
│   │   │   ├── 000000000025.png
├── mapillary
│   ├── config_v2.0.json
│   ├── training
│   │   ├── v2.0
│   │   │   ├── labels
│   │   │   │   ├── 0035fkbjWljhaftpVM37-g.png
│   │   │   │   ├── 00qclUcInksIYnm19b1Xfw.png
│   │   ├── images
│   │   │   ├── 0035fkbjWljhaftpVM37-g.jpg
│   │   │   ├── 00qclUcInksIYnm19b1Xfw.jpg
├── paco_lvis
│   ├── annotations
│   │   ├── paco_lvis_v1_train.json
├── pascal_part
│   ├── train.json
│   ├── VOCdevkit
│   │   │   ├── VOC2010
│   │   │   │   ├── JPEGImages
│   │   │   │   │   ├── 2007_000027.jpg
│   │   │   │   │   ├── 2007_000032.jpg
├── GranDf
│   ├── annotations
│   │   ├── train
│   │   │   ├── GranDf_HA_GCG_train.json
│   │   │   ├── OpenPsgGCG_train.json
│   │   │   ├── OpenPsgGCG_val.json
│   │   │   ├── RefCOCOg_GCG_train.json
│   │   │   ├── RefCOCOg_GCG_val.json
│   │   │   ├── flickr_mergedGT_GCG_train.json
│   │   │   ├── flickr_mergedGT_GCG_val.json
│   │   ├── val_test
│   │   │   ├── test_gcg_coco_caption_gt.json
│   │   │   ├── test_gcg_coco_mask_gt.json
│   │   │   ├── val_gcg_coco_caption_gt.json
│   │   │   ├── val_gcg_coco_mask_gt.json
├── GranDf_HA_images
│   ├── train
│   │   ├── sa_10010541.jpg
│   │   ├── sa_10014079.jpg
│   ├── val_test
│   │   ├── sa_10010541.jpg
│   │   ├── sa_10014079.jpg
├── Reason
│   ├── RIO
│   │   ├── RIO_common_all_thing.json
│   │   ├── RIO_train_all_thing.json
│   │   ├── RIO_uncommon_all_thing.json
│
├── referexp_segm
│   ├── grefcoco
├── coco_2014
│   ├── train2014
│   │   ├── COCO_train2014_000000000009.jpg
│   │   ├── COCO_train2014_000000000025.jpg
```

### 5) Image-Level Understanding Datasets
For image-level understanding, we also incorporate image-level datasets, including image captioning data converted from COCO Captions and visual question answering data from LLaVA-Instruct-150K.

Download links and structure:
- COCO Captions: [COCO - 2017 annotations](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)
- LLaVA-instruct-150k: [LLaVA-instruct-150k](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_instruct_150k.json)
- Images: `coco_2017` - COCO-2017 ([train2017](http://images.cocodataset.org/zips/train2017.zip))

Download the data from the source links, and arrange as follows:

```
├── llava_dataset
│   ├── llava_instruct_150k.json
├── coco
│   ├── train2017
│   │   ├── 000000000009.jpg
│   │   ├── 000000000025.jpg
│   ├── annotations
│   │   ├── captions_train2017.json
│   │   ├── captions_val2017.json
```






