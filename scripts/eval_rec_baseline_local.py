import argparse
import json
import os
import random
import re
from pathlib import Path

import torch
from PIL import Image
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def extract_bbox_answer(content: str):
    bbox_pattern = r"\[(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*)\]"
    bbox_match = re.search(bbox_pattern, content)
    if bbox_match:
        return [float(bbox_match.group(i)) for i in range(1, 5)]
    return [0, 0, 0, 0]


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


def resize_bbox(bbox, input_height, input_width, image_height, image_width):
    return [
        bbox[0] / input_width * image_width,
        bbox[1] / input_height * image_height,
        bbox[2] / input_width * image_width,
        bbox[3] / input_height * image_height,
    ]


def load_dataset(path, num_samples, seed):
    with open(path, "r") as f:
        data = json.load(f)
    random.seed(seed)
    random.shuffle(data)
    return data[:num_samples] if num_samples > 0 else data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attn-implementation", default="sdpa")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    model_path = Path(args.model_path)
    adapter_config_path = model_path / "adapter_config.json"
    if adapter_config_path.exists():
        adapter_config = json.loads(adapter_config_path.read_text())
        base_model_path = adapter_config["base_model_name_or_path"]
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model_path,
            torch_dtype=dtype,
            attn_implementation=args.attn_implementation,
            device_map="auto" if device == "cuda" else "cpu",
        )
        model = PeftModel.from_pretrained(base_model, str(model_path))
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_path),
            torch_dtype=dtype,
            attn_implementation=args.attn_implementation,
            device_map="auto" if device == "cuda" else "cpu",
        )
    processor = AutoProcessor.from_pretrained(str(model_path))

    question_template = "{Question} Please provide the bounding box coordinate in JSON format."
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for ds in args.datasets:
        ds_path = Path(args.data_root) / f"{ds}.json"
        data = load_dataset(ds_path, args.num_samples, args.seed)
        messages = []
        for item in data:
            image_path = Path(args.image_root) / item["image"]
            messages.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": f"file://{image_path}"},
                            {"type": "text", "text": question_template.format(Question=item["problem"])},
                        ],
                    }
                ]
            )

        outputs = []
        for i in tqdm(range(0, len(messages), args.batch_size), desc=ds):
            batch_messages = messages[i : i + args.batch_size]
            text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
            image_inputs, video_inputs = process_vision_info(batch_messages)
            inputs = processor(
                text=text,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                padding_side="left",
                return_tensors="pt",
            )
            inputs = inputs.to(device)
            generated_ids = model.generate(
                **inputs,
                use_cache=True,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            batch_output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            for j, output_text in enumerate(batch_output_text):
                input_height = int(inputs["image_grid_thw"][j][1] * 14)
                input_width = int(inputs["image_grid_thw"][j][2] * 14)
                image = Image.open(batch_messages[j][0]["content"][0]["image"].split("file://")[1])
                image_width, image_height = image.size
                outputs.append((output_text, input_height, input_width, image_height, image_width))

        final_output = []
        correct_number = 0
        for input_example, model_output in zip(data, outputs):
            original_output, input_height, input_width, image_height, image_width = model_output
            ground_truth = input_example["solution"]
            model_answer = extract_bbox_answer(original_output)
            resized_model_answer = resize_bbox(model_answer, input_height, input_width, image_height, image_width)
            correct = int(iou(resized_model_answer, ground_truth) > 0.5)
            correct_number += correct
            final_output.append(
                {
                    "image": input_example["image"],
                    "question": input_example["problem"],
                    "ground_truth": ground_truth,
                    "model_output": original_output,
                    "extracted_answer": resized_model_answer,
                    "correct": correct,
                }
            )

        accuracy = correct_number / len(data) * 100 if data else 0.0
        model_name = model_path.name
        if adapter_config_path.exists():
            model_name = f"{model_path.parent.name}_{model_name}"
        output_path = output_dir / f"rec_results_{ds}_{model_name}.json"
        with open(output_path, "w") as f:
            json.dump({"accuracy": accuracy, "results": final_output}, f, indent=2)
        print(f"{ds}: {accuracy:.2f}% -> {output_path}")


if __name__ == "__main__":
    main()
