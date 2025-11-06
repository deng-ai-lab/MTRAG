import argparse
import torch
import os
import json
from transformers import AutoTokenizer, CLIPImageProcessor
from mtrag.llava.constants import IMAGE_TOKEN_INDEX
from mtrag.llava.conversation import SeparatorStyle
from mtrag.llava import conversation as conversation_lib
from mtrag.llava.utils import disable_torch_init
from mtrag.llava.mm_utils import tokenizer_image_token
from dataset.utils.process import preprocess_multimodal
from mtrag.SAM.utils.transforms import ResizeLongestSide
from mtrag.MTRAG import MTRAGForCausalLM
from pycocotools import mask as maskUtils
import torch.nn.functional as F
import numpy as np
from PIL import Image
import re

def annToMask(ann, h, w):
    rles = maskUtils.frPyObjects(ann, h, w)
    rle = maskUtils.merge(rles)
    m = maskUtils.decode(rle)
    return m

class GPT_EVAL():
    IMG_MEAN = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    IMG_STD = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    IMG_SIZE = 1024
    IGNORE_LABEL = 255
    def __init__(self, model_path, args):
        super().__init__()
        self.args = args
        disable_torch_init()
        model_path = os.path.expanduser(model_path)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            model_max_length=2048,
            padding_side="right",
            use_fast=True
        )
        self.model = MTRAGForCausalLM.from_pretrained(
                                                model_path,
                                                torch_dtype=torch.bfloat16,
                                                ).cuda()
        self.tokenizer.pad_token = self.tokenizer.unk_token

        self.image_processor = CLIPImageProcessor.from_pretrained(self.model.config.mm_vision_tower)
        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        self.model.config.bos_token_id = self.tokenizer.bos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.get_model().initialize_vision_modules(self.model.get_model().config, add_region_feature=True)
        self.model.config.use_cache = True

        vision_tower = self.model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(dtype=torch.bfloat16, device='cuda')
        self.transform = ResizeLongestSide(self.IMG_SIZE)
        self.grounding = self.model.config.mm_projector_type == "SAMCLIP"
    
    def grounding_enc_processor(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.IMG_MEAN) / self.IMG_STD
        h, w = x.shape[-2:]
        x = F.pad(x, (0, self.IMG_SIZE - w, 0, self.IMG_SIZE - h))
        return x
    
    def eval(self, root_path, ann_file):
        final = []
        anns = json.load(open(ann_file))
        
        for i, ann in enumerate(anns):
            print(i)
            model_answer = {}
            model_answer["question_id"] = ann["question_id"]
            model_answer["image"] = ann["image"]
            model_answer["category"] = ann["category"]
            img_path = os.path.join(root_path,"train2017", ann['image'])
            question = ann["text"]
            annotations = ann['annotation']
            init_inputs = get_init_inputs(img_path,
                                        self.image_processor,
                                        annotation=annotations,
                                        question=question,
                                        round_ids=0,
                                        last_round_source={},
                                        )
            image = init_inputs['image']

            if self.grounding:
                image_np = np.array(image)
                original_size_list = [image_np.shape[:2]]
                image_np = self.transform.apply_image(image_np)
                image_resize = [image_np.shape[:2]]
                grounding_enc_image = self.grounding_enc_processor(torch.from_numpy(image_np).permute(2, 0, 1).contiguous()).unsqueeze(0).bfloat16().cuda()
            else:
                image_resize = [None]
                grounding_enc_image = torch.zeros(3, 1024, 1024)

            conv = conversation_lib.default_conversation.copy()
            qs = init_inputs['sources'][0][0]['value']

            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2



            with torch.inference_mode():

                output_ids, _ = self.model.evaluate(init_inputs["global_enc_image"], grounding_enc_image, input_ids, image_resize, original_size_list, max_tokens_new=2048,   region_masks=[torch.tensor(init_inputs["region_masks"])], do_sample=True if self.args.temperature > 0 else False, temperature=self.args.temperature, use_cache=True)

            outputs = self.tokenizer.batch_decode(output_ids,
                                                  skip_special_tokens=True)[0]

            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()
            if ':' in outputs:
                outputs = outputs.split(':')[1]

            model_answer['text'] = outputs
            
            print(outputs)
            final.append(model_answer)

        final_ = json.dumps(final)
        os.makedirs(os.path.dirname(self.args.result_json), exist_ok=True)
        with open(self.args.result_json,'w') as fw:
            fw.write(final_)
            fw.close()
     
def generate_alpha_for_feature(processor, image, masks):
    image_masks = np.where(masks > 0, 255.0, 0.0)
    rgb_masks = np.repeat(image_masks[:, :, :, np.newaxis], 3, axis=-1).astype(np.uint8)  # (batch, H, W, 3)
    global_enc_image_mask = processor.preprocess(list(rgb_masks)+[image], return_tensors="pt")["pixel_values"]
    global_enc_image = global_enc_image_mask[-1]
    global_enc_image_mask = global_enc_image_mask[:-1].sum(dim=1, keepdim=True)
    global_enc_image_mask = (global_enc_image_mask > 6) * 1.9231 + (global_enc_image_mask <= 6) * (-1.9231)
    return global_enc_image, global_enc_image_mask.squeeze(1)

def get_init_inputs(img_path,
                    processor,
                    annotation,
                    question=None,
                    round_ids=0,
                    last_round_source=None):

    if round_ids == 0:
        image = Image.open(img_path).convert('RGB')
    else:
        image = last_round_source['image']

    begin_str = """<image>.\nThis provides an overview of the picture.\n"""
    region_num = len(annotation)
    region_string = ''
    height, width = np.array(image).shape[:2]
    masks = []
    for i in range(region_num):
        mask = annotation[i]['segmentation']
        masks.append(annToMask(mask, height, width))
        region_string += f'region{i + 1} <region_fea>'
        if i < region_num -2:
            region_string += ', '
        elif i == region_num - 2:
            region_string += ' and '
    mid_str = "There are {} part regions in the picture: ".format(str(region_num))+region_string+'. '
    sources = dict()
    sources['conversations'] = []
    question = re.sub(r'<region(\d+)>', lambda match: f'region{match.group(1)}', question)

    sources['conversations'].append({'from': 'human', 'value': begin_str + mid_str + question})
    
    sources = preprocess_multimodal([sources['conversations']], use_im_start_end=False)



    global_enc_image, global_enc_image_mask = generate_alpha_for_feature(processor, np.array(image), np.array(masks))
    global_enc_image = torch.cat([global_enc_image, global_enc_image_mask], dim=0)

    data_dict = {}
    data_dict['sources'] = sources
    data_dict['image'] = image
    data_dict['masks'] = masks
    data_dict['region_masks'] = np.array(masks)
    data_dict['global_enc_image'] = global_enc_image.unsqueeze(0).bfloat16().cuda()

    return data_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='mtrag generate gpt answer', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--model', help='path to mtrag model', default='/path/to/mtrag-7b')
    parser.add_argument('--coco-img', help='path to coco imgs', default='./data/coco')
    parser.add_argument('--json', help='path to question json file', default='./description/questions.json')
    parser.add_argument('--result_json', default='./eval_results/mtr-bench/mtr_relations.json')
    parser.add_argument("--temperature", default=0.2, type=float)
    args = parser.parse_args()

    gpt_eval = GPT_EVAL(args.model,args)
    conversation_lib.default_conversation = conversation_lib.conv_templates["v1"]
    gpt_eval.eval(args.coco_img, args.json)
