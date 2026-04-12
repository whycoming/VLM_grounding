import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


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


def round_box(box, ndigits=2):
    return tuple(round(float(x), ndigits) for x in box)


def load_records(path: Path):
    with path.open() as f:
        if path.suffix == ".json":
            records = json.load(f)
        elif path.suffix == ".jsonl":
            records = [json.loads(line) for line in f]
        else:
            raise ValueError(f"Unsupported input format: {path}")
    return records


def get_prompt(record):
    if "problem" in record:
        return f"<image>{record['problem']}".strip()
    if "conversations" in record:
        human_turn = next(turn for turn in record["conversations"] if turn["from"] == "human")
        return human_turn["value"]
    raise KeyError("Record is missing both `problem` and `conversations` fields.")


def get_solution(record):
    if "solution" in record:
        return [float(x) for x in record["solution"]]
    if "conversations" in record:
        gpt_turn = next(turn for turn in record["conversations"] if turn["from"] == "gpt")
        return [float(x) for x in gpt_turn["value"]]
    raise KeyError("Record is missing both `solution` and `conversations` target fields.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-stats", required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-negatives-per-prompt", type=int, default=1)
    parser.add_argument("--max-negative-iou", type=float, default=0.3)
    parser.add_argument("--min-negative-iou", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    input_path = Path(args.input_json)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path = Path(args.output_stats)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    records = load_records(input_path)

    grouped_boxes = defaultdict(dict)
    for record in records:
        if "category_id" not in record:
            raise KeyError("Hard Negative DPO requires `category_id` in the source dataset.")
        gt_box = get_solution(record)
        group_key = (record["image"], int(record["category_id"]))
        grouped_boxes[group_key][round_box(gt_box)] = gt_box

    output_rows = []
    prompts_with_pairs = 0
    negative_iou_values = []
    skipped_no_group_match = 0
    skipped_no_valid_negative = 0
    seen_pairs = set()

    indices = list(range(len(records)))
    random.shuffle(indices)
    if args.max_samples > 0:
        indices = indices[: args.max_samples]

    for index in indices:
        record = records[index]
        gt_box = get_solution(record)
        group_key = (record["image"], int(record["category_id"]))
        candidate_boxes = grouped_boxes[group_key]
        if len(candidate_boxes) <= 1:
            skipped_no_group_match += 1
            continue

        negatives = []
        gt_key = round_box(gt_box)
        for neg_key, neg_box in candidate_boxes.items():
            if neg_key == gt_key:
                continue
            neg_iou = iou(gt_box, neg_box)
            if neg_iou < args.min_negative_iou or neg_iou >= args.max_negative_iou:
                continue
            negatives.append((neg_iou, neg_box))

        if not negatives:
            skipped_no_valid_negative += 1
            continue

        negatives.sort(key=lambda item: item[0], reverse=True)
        prompt = get_prompt(record)
        added = 0
        for neg_iou, neg_box in negatives:
            pair_key = (record["image"], prompt, gt_key, round_box(neg_box))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            output_rows.append(
                {
                    "image": record["image"],
                    "prompt": prompt,
                    "chosen": gt_box,
                    "rejected": neg_box,
                    "chosen_iou": 1.0,
                    "rejected_iou": neg_iou,
                    "iou_gap": 1.0 - neg_iou,
                    "image_id": record.get("image_id"),
                    "category_id": record.get("category_id"),
                    "source_id": record.get("id"),
                    "source_dataset": record.get("dataset_name"),
                }
            )
            negative_iou_values.append(neg_iou)
            added += 1
            if added >= args.max_negatives_per_prompt:
                break

        if added > 0:
            prompts_with_pairs += 1

    with output_path.open("w") as f:
        for row in output_rows:
            f.write(json.dumps(row) + "\n")

    stats = {
        "input_json": str(input_path),
        "output_jsonl": str(output_path),
        "source_records": len(records),
        "sampled_records": len(indices),
        "pairs": len(output_rows),
        "prompts_with_pairs": prompts_with_pairs,
        "unique_image_category_groups": len(grouped_boxes),
        "max_negatives_per_prompt": args.max_negatives_per_prompt,
        "min_negative_iou": args.min_negative_iou,
        "max_negative_iou": args.max_negative_iou,
        "avg_negative_iou": sum(negative_iou_values) / len(negative_iou_values) if negative_iou_values else 0.0,
        "max_negative_iou_observed": max(negative_iou_values) if negative_iou_values else 0.0,
        "min_negative_iou_observed": min(negative_iou_values) if negative_iou_values else 0.0,
        "skipped_no_group_match": skipped_no_group_match,
        "skipped_no_valid_negative": skipped_no_valid_negative,
        "seed": args.seed,
    }
    stats_path.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
