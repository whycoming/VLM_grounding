import argparse
import json
import random
import re
from pathlib import Path

import torch
from peft import PeftModel
from PIL import Image
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def extract_bbox_answer(content: str):
    bbox_pattern = r"\[(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*)\]"
    bbox_match = re.search(bbox_pattern, content)
    if bbox_match:
        return [float(bbox_match.group(i)) for i in range(1, 5)]
    return [0, 0, 0, 0]


def resize_bbox(bbox, input_height, input_width, image_height, image_width):
    return [
        bbox[0] / input_width * image_width,
        bbox[1] / input_height * image_height,
        bbox[2] / input_width * image_width,
        bbox[3] / input_height * image_height,
    ]


def iou(box1, box2):
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2] - 1, box2[2] - 1)
    inter_y2 = min(box1[3] - 1, box2[3] - 1)
    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        inter = (inter_x2 - inter_x1 + 1) * (inter_y2 - inter_y1 + 1)
    else:
        inter = 0
    union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter
    return float(inter) / union if union > 0 else 0.0


def load_model_and_processor(model_path: Path, attn_implementation: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    adapter_config_path = model_path / "adapter_config.json"
    if adapter_config_path.exists():
        adapter_config = json.loads(adapter_config_path.read_text())
        base_model_path = adapter_config["base_model_name_or_path"]
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model_path,
            torch_dtype=dtype,
            attn_implementation=attn_implementation,
            device_map="auto" if device == "cuda" else "cpu",
        )
        model = PeftModel.from_pretrained(base_model, str(model_path))
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_path),
            torch_dtype=dtype,
            attn_implementation=attn_implementation,
            device_map="auto" if device == "cuda" else "cpu",
        )
    processor = AutoProcessor.from_pretrained(str(model_path))
    return model, processor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-stats", required=True)
    parser.add_argument("--max-samples", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-return-sequences", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--min-iou", type=float, default=0.7)
    parser.add_argument("--top-k-per-prompt", type=int, default=2)
    parser.add_argument("--attn-implementation", default="sdpa")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path = Path(args.output_stats)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open() as f:
        records = [json.loads(line) for line in f]
    random.shuffle(records)
    if args.max_samples > 0:
        records = records[: args.max_samples]

    model, processor = load_model_and_processor(Path(args.model_path), args.attn_implementation)
    model.eval()

    accepted = []
    sampled = 0
    accepted_count = 0

    for record in tqdm(records, desc="rft-generate"):
        human_turn = next(turn for turn in record["conversations"] if turn["from"] == "human")
        gt_box = next(turn for turn in record["conversations"] if turn["from"] == "gpt")["value"]
        image_path = Path(args.image_root) / record["image"]
        user_text = human_turn["value"].replace("<image>", "").strip()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{image_path}"},
                    {"type": "text", "text": user_text},
                ],
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        device = next(model.parameters()).device
        inputs = inputs.to(device)
        generated_ids = model.generate(
            **inputs,
            use_cache=True,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            num_return_sequences=args.num_return_sequences,
        )

        prompt_len = inputs.input_ids.shape[1]
        with Image.open(image_path) as image:
            image_width, image_height = image.size
        input_height = int(inputs["image_grid_thw"][0][1] * 14)
        input_width = int(inputs["image_grid_thw"][0][2] * 14)
        seen = set()
        accepted_for_prompt = []

        for out_ids in generated_ids:
            sampled += 1
            output_text = processor.decode(out_ids[prompt_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            model_answer = extract_bbox_answer(output_text)
            resized_answer = resize_bbox(model_answer, input_height, input_width, image_height, image_width)
            score = iou(resized_answer, gt_box)
            rounded_bbox = [round(v, 2) for v in resized_answer]
            bbox_key = tuple(int(round(v)) for v in rounded_bbox)
            if score < args.min_iou or bbox_key in seen:
                continue
            seen.add(bbox_key)
            accepted_for_prompt.append(
                {
                    "image": record["image"],
                    "conversations": [
                        {"from": "human", "value": human_turn["value"]},
                        {"from": "gpt", "value": rounded_bbox},
                    ],
                    "rft_source_iou": round(score, 6),
                }
            )

        accepted_for_prompt.sort(key=lambda row: row["rft_source_iou"], reverse=True)
        kept = accepted_for_prompt[: args.top_k_per_prompt] if args.top_k_per_prompt > 0 else accepted_for_prompt
        accepted_count += len(kept)
        accepted.extend(kept)

    with output_path.open("w") as f:
        for row in accepted:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats = {
        "input_jsonl": str(input_path),
        "model_path": args.model_path,
        "num_input_examples": len(records),
        "num_candidates_sampled": sampled,
        "num_accepted_examples": accepted_count,
        "accept_rate_over_candidates": accepted_count / sampled if sampled else 0.0,
        "min_iou": args.min_iou,
        "top_k_per_prompt": args.top_k_per_prompt,
        "num_return_sequences": args.num_return_sequences,
    }
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
