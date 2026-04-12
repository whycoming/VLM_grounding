import argparse
import json
import random
import re
from pathlib import Path

import torch
import torch.nn.functional as F
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
    return None


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


def build_messages(image_path: Path, prompt: str, bbox=None):
    user_text = prompt.replace("<image>", "").strip()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text", "text": user_text},
            ],
        }
    ]
    if bbox is not None:
        messages.append(
            {
                "role": "assistant",
                "content": f"```json\n[{int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}]\n```",
            }
        )
    return messages


def processor_call(processor, texts, images, videos):
    kwargs = {
        "text": texts,
        "padding": True,
        "return_tensors": "pt",
    }
    if any(item is not None and len(item) > 0 for item in images):
        kwargs["images"] = images
    if any(item is not None and len(item) > 0 for item in videos):
        kwargs["videos"] = videos
    return processor(**kwargs)


def sequence_logps(logits, input_ids, completion_mask):
    shifted_logits = logits[:, :-1, :]
    shifted_input_ids = input_ids[:, 1:]
    shifted_mask = completion_mask[:, 1:].to(logits.dtype)
    token_logps = F.log_softmax(shifted_logits, dim=-1).gather(
        dim=-1, index=shifted_input_ids.unsqueeze(-1)
    ).squeeze(-1)
    return (token_logps * shifted_mask).sum(dim=-1)


def compute_ref_pair_logps(model, processor, image_path: Path, prompt: str, chosen_bbox, rejected_bbox):
    prompt_messages = [build_messages(image_path, prompt), build_messages(image_path, prompt)]
    full_messages = [
        build_messages(image_path, prompt, chosen_bbox),
        build_messages(image_path, prompt, rejected_bbox),
    ]

    prompt_texts = [processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) for msgs in prompt_messages]
    full_texts = [processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False) for msgs in full_messages]
    prompt_vision = [process_vision_info(msgs) for msgs in prompt_messages]
    full_vision = [process_vision_info(msgs) for msgs in full_messages]

    prompt_batch = processor_call(
        processor,
        prompt_texts,
        [imgs for imgs, _ in prompt_vision],
        [vids for _, vids in prompt_vision],
    )
    full_batch = processor_call(
        processor,
        full_texts,
        [imgs for imgs, _ in full_vision],
        [vids for _, vids in full_vision],
    )

    completion_mask = torch.zeros_like(full_batch["attention_mask"])
    prompt_lengths = prompt_batch["attention_mask"].sum(dim=1)
    for idx, prompt_len in enumerate(prompt_lengths.tolist()):
        seq_len = int(full_batch["attention_mask"][idx].sum().item())
        completion_mask[idx, prompt_len:seq_len] = 1

    device = next(model.parameters()).device
    model_inputs = {k: v.to(device) for k, v in full_batch.items()}
    with torch.no_grad():
        outputs = model(**model_inputs)
        logps = sequence_logps(outputs.logits, model_inputs["input_ids"], completion_mask.to(device))
    return float(logps[0].item()), float(logps[1].item())


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
    parser.add_argument("--min-iou-gap", type=float, default=0.1)
    parser.add_argument("--min-chosen-iou", type=float, default=0.0)
    parser.add_argument("--max-rejected-iou", type=float, default=1.0)
    parser.add_argument("--pairs-per-prompt", type=int, default=1)
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

    pairs = []
    sampled = 0
    prompts_with_pairs = 0
    candidate_ious = []

    for record in tqdm(records, desc="online-dpo-generate"):
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

        scored = []
        seen = set()
        for out_ids in generated_ids:
            sampled += 1
            output_text = processor.decode(out_ids[prompt_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            model_answer = extract_bbox_answer(output_text)
            if model_answer is None:
                continue
            resized_answer = resize_bbox(model_answer, input_height, input_width, image_height, image_width)
            rounded_bbox = [round(v, 2) for v in resized_answer]
            bbox_key = tuple(int(round(v)) for v in rounded_bbox)
            if bbox_key in seen:
                continue
            seen.add(bbox_key)
            score = iou(resized_answer, gt_box)
            candidate_ious.append(score)
            scored.append({"bbox": rounded_bbox, "iou": round(score, 6)})

        if len(scored) < 2:
            continue
        scored.sort(key=lambda row: row["iou"], reverse=True)

        created_pairs = 0
        for pair_idx in range(min(args.pairs_per_prompt, len(scored) // 2)):
            chosen = scored[pair_idx]
            rejected = scored[-(pair_idx + 1)]
            gap = chosen["iou"] - rejected["iou"]
            if chosen["iou"] < args.min_chosen_iou:
                continue
            if rejected["iou"] > args.max_rejected_iou:
                continue
            if gap < args.min_iou_gap:
                continue
            pairs.append(
                {
                    "image": record["image"],
                    "prompt": human_turn["value"],
                    "chosen": chosen["bbox"],
                    "rejected": rejected["bbox"],
                    "chosen_iou": chosen["iou"],
                    "rejected_iou": rejected["iou"],
                    "iou_gap": round(gap, 6),
                    "gt_bbox": gt_box,
                }
            )
            ref_chosen_logps, ref_rejected_logps = compute_ref_pair_logps(
                model,
                processor,
                image_path,
                human_turn["value"],
                chosen["bbox"],
                rejected["bbox"],
            )
            pairs[-1]["ref_chosen_logps"] = round(ref_chosen_logps, 6)
            pairs[-1]["ref_rejected_logps"] = round(ref_rejected_logps, 6)
            created_pairs += 1

        if created_pairs > 0:
            prompts_with_pairs += 1

    with output_path.open("w") as f:
        for row in pairs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats = {
        "input_jsonl": str(input_path),
        "model_path": args.model_path,
        "num_input_examples": len(records),
        "num_candidates_sampled": sampled,
        "num_pairs": len(pairs),
        "prompts_with_pairs": prompts_with_pairs,
        "pair_rate_over_prompts": prompts_with_pairs / len(records) if records else 0.0,
        "pair_rate_over_candidates": len(pairs) / sampled if sampled else 0.0,
        "avg_candidate_iou": sum(candidate_ious) / len(candidate_ious) if candidate_ious else 0.0,
        "min_iou_gap": args.min_iou_gap,
        "min_chosen_iou": args.min_chosen_iou,
        "max_rejected_iou": args.max_rejected_iou,
        "pairs_per_prompt": args.pairs_per_prompt,
        "num_return_sequences": args.num_return_sequences,
    }
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
