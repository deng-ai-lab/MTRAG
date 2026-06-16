import re
import cv2
import json
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPImageProcessor
from torch.utils.data import DataLoader, DistributedSampler

from eval.utils import *
from eval.ddp import *
from mtrag.MTRAG import MTRAGForCausalLM
from mtrag.llava import conversation as conversation_lib
from mtrag.llava.mm_utils import tokenizer_image_token
from mtrag.SAM.utils.transforms import ResizeLongestSide
from tools.utils import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX


def parse_args():
    parser = argparse.ArgumentParser(description="MTRAG Inference - Region Captioning")

    parser.add_argument("--hf_model_path", required=True, help="The model path in huggingface format.")
    parser.add_argument("--lora_model_path", default=None, help="Path to the pretrained model for evaluation.")
    parser.add_argument("--annotation_file",
                        default="./data/region_cap/finetune_refcocog_val_captions.json", type=str,
                        help="Replace with 'data/visual_genome/test_caption.json' for VG.")
    parser.add_argument("--image_dir", default="./data/coco_2014/train2014", type=str,
                        help="Replace with 'data/visual_genome/images' for VG")
    parser.add_argument("--dataset", default="refcocog", type=str, help="Options are 'refcocog', 'vg'")
    parser.add_argument("--results_dir", default="results", type=str, help="The path to save the results.")
    parser.add_argument("--temperature", default=0.2, type=float)


    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--use_mm_start_end", action="store_true", default=False)
    parser.add_argument("--conv_type", default="llava_v1", type=str, choices=["llava_v1", "llava_llama_2"], )

    # DDP Related parameters
    parser.add_argument("--batch_size_per_gpu", required=False, default=1)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser.parse_args()

def generate_alpha_for_feature(image, masks, global_enc_processor):
    image_masks = np.where(masks > 0, 255.0, 0.0)
    rgb_masks = np.repeat(image_masks[:, :, :, np.newaxis], 3, axis=-1).astype(np.uint8)  # (batch, H, W, 3)
    global_enc_image_mask = global_enc_processor.preprocess(list(rgb_masks)+[image], return_tensors="pt")["pixel_values"]
    global_enc_image = global_enc_image_mask[-1]
    global_enc_image_mask = global_enc_image_mask[:-1].sum(dim=1, keepdim=True)
    global_enc_image_mask = (global_enc_image_mask > 6) * 1.9231 + (global_enc_image_mask <= 6) * (-1.9231)
    return global_enc_image, global_enc_image_mask.squeeze(1)
def inference(instructions, inputs, global_enc_processor,transform):
    # Extract the inputs
    bbox_img = inputs['region']
    image_path = inputs['image']
    instructions = instructions.replace('&lt;', '<').replace('&gt;', '>')
    assert len(bbox_img) == 1, "Only one region is supported"
    instructions = instructions.replace(f'<region>', f'region1 <region_fea>')

    # Prepare prompt for model Inference
    conv = conversation_lib.conv_templates[args.conv_type].copy()
    conv.messages = []
    begin_str = f"""The {DEFAULT_IMAGE_TOKEN} provides an overview of the picture.\n"""
    prompt = begin_str + instructions
    if args.use_mm_start_end:
        replace_token = (DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN)
        prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], "")
    prompt = conv.get_prompt()

    # Read and preprocess the image (Global image encoder - CLIP)
    image_np = cv2.imread(image_path)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    original_size_list = [image_np.shape[:2]]

    # Preprocess the image (Grounding image encoder)
    image = transform.apply_image(image_np)
    resize_list = [image.shape[:2]]
    grounding_enc_image = (
        grounding_image_ecoder_preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous()).unsqueeze(0).to(device))
    grounding_enc_image = grounding_enc_image.bfloat16()  # Precision is bf16 by default

    # Prepare inputs for inference
    input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
    input_ids = input_ids.unsqueeze(0).cuda()
    region_masks = None
    if len(bbox_img) > 0:
        masks = bbox_img
        region_masks = np.array(masks)
        global_enc_image, global_enc_image_mask = generate_alpha_for_feature(image_np, region_masks,global_enc_processor)
        global_enc_image = torch.cat([global_enc_image, global_enc_image_mask], dim=0).unsqueeze(0).bfloat16().to(device)
    else:
        global_enc_image = (global_enc_processor.preprocess(image_np, return_tensors="pt")["pixel_values"][0].unsqueeze(0).to(device))
        global_enc_image = global_enc_image.bfloat16()  # Precision is bf16 by default

    # Generate output
    output_ids, pred_masks = model.evaluate(global_enc_image, grounding_enc_image, input_ids, resize_list, original_size_list,max_tokens_new=1024, region_masks=[torch.tensor(region_masks)], do_sample=True if args.temperature > 0 else False, temperature=args.temperature,use_cache=True)
    output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]

    # Post-processing
    text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
    text_output = text_output.replace("\n", "").replace("  ", " ")
    text_output = text_output.split("ASSISTANT: ")[-1]

    cleaned_str = re.sub(r'<.*?>', '', text_output)

    # Remove the [SEG] token
    cleaned_str = cleaned_str.replace('[SEG]', '')

    # Strip unnecessary spaces
    cleaned_str = ' '.join(cleaned_str.split()).strip("'")
    cleaned_str = cleaned_str.strip()

    return cleaned_str


def custom_collate_fn(batch):
    image_id = [item[0] for item in batch]
    filename = [item[1] for item in batch]
    bbox = [item[2] for item in batch]
    gt = [item[3] for item in batch]

    return image_id, filename, bbox, gt


if __name__ == "__main__":
    args = parse_args()
    init_distributed_mode(args)

    # Initialize tokenizer and model
    device = torch.device(f"cuda:{args.rank}")
    if args.conv_type in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_type]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_path, cache_dir=None,
                                              model_max_length=args.model_max_length, padding_side="right",
                                              use_fast=False)
    tokenizer.pad_token = tokenizer.unk_token
    seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    torch_dtype = torch.bfloat16  # By default, using bf16
    kwargs = {"torch_dtype": torch_dtype}
    model = MTRAGForCausalLM.from_pretrained(args.hf_model_path, low_cpu_mem_usage=True, seg_token_idx=seg_token_idx, **kwargs)
    # Update model config
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.get_model().initialize_vision_modules(model.get_model().config, add_region_feature=True)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)

    # Transfer the model to GPU
    model = model.bfloat16().to(device) # Replace with model = model.float().cuda() for 32 bit inference
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(device)

    # Initialize Image Processor for GLobal Image Encoder (CLIP)
    clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    transform = ResizeLongestSide(args.image_size)

    model.eval()  # Model should be in evaluation mode for inference

    # Prompt model to perfor region captioning task
    instruction = "Please give me a short description of <region>?"

    # Intermediate results path is hard-coded (you may change it as per your needs)
    os.makedirs(args.results_dir, exist_ok=True)
    results_path = f"{args.results_dir}/{os.path.basename(args.hf_model_path)}_{args.dataset}_{args.rank}.json"

    # Create DDP Dataset
    dataset = RegionCapDDP(args.annotation_file)
    distributed_sampler = DistributedSampler(dataset, rank=args.rank, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size_per_gpu, num_workers=0,
                            sampler=distributed_sampler, collate_fn=custom_collate_fn)

    # Iterate over all the samples, perform inference and save results
    results = []
    for idx, (image_id, filename, bbox, gt) in enumerate(tqdm(dataloader)):
        image_id, filename, bbox, gt = image_id[0], filename[0], bbox[0], gt[0]
        image_path = os.path.join(args.image_dir, filename)
        inputs = {'image': image_path, 'region': [bbox]}

        result_caption = inference(instruction, inputs, clip_image_processor, transform)  # Perform inference

        result_dict = {}
        result_dict["image_id"] = image_id
        result_dict["caption"] = result_caption
        results.append(result_dict)

    with open(results_path, 'w') as json_file:
        json.dump(results, json_file, indent=2)
