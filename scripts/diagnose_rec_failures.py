import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


SPATIAL_WORDS = {"left", "right", "above", "below", "near", "between", "behind", "front"}


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


def center(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def diagonal(box):
    return math.sqrt(max(box[2] - box[0], 0) ** 2 + max(box[3] - box[1], 0) ** 2)


def distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def parse_valid_bbox(box):
    return (
        isinstance(box, list)
        and len(box) == 4
        and all(isinstance(v, (int, float)) for v in box)
        and box[2] > box[0]
        and box[3] > box[1]
    )


def classify_failure(result):
    pred_box = result["extracted_answer"]
    gt_box = result["ground_truth"]
    description = result["question"]
    pred_valid = parse_valid_bbox(pred_box)

    if not pred_valid or pred_box == [0, 0, 0, 0]:
        return "FORMAT_ERROR"

    sample_iou = iou(pred_box, gt_box)
    if sample_iou >= 0.5:
        return "CORRECT"

    gt_diag = diagonal(gt_box)
    ctr_dist = distance(center(pred_box), center(gt_box))
    if gt_diag > 0 and ctr_dist < 0.3 * gt_diag and sample_iou < 0.5:
        return "COORDINATE_IMPRECISION"

    if any(w in description.lower() for w in SPATIAL_WORDS):
        return "SPATIAL_REASONING_ERROR"

    # Conservative bucket: without full object annotations, we cannot reliably
    # separate wrong-target from attribute misunderstanding.
    if sample_iou < 0.1:
        return "LOW_OVERLAP_SEMANTIC_OR_WRONG_TARGET"

    return "ATTRIBUTE_OR_OTHER"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    per_dataset = {}
    overall_counter = Counter()
    examples_by_type = defaultdict(list)

    for path in sorted(Path(args.input_dir).glob("rec_results_*.json")):
        payload = json.loads(path.read_text())
        dataset_counter = Counter()
        for result in payload["results"]:
            failure_type = classify_failure(result)
            result["failure_type"] = failure_type
            dataset_counter[failure_type] += 1
            overall_counter[failure_type] += 1
            if failure_type != "CORRECT" and len(examples_by_type[failure_type]) < 10:
                examples_by_type[failure_type].append(
                    {
                        "dataset": path.name,
                        "image": result["image"],
                        "question": result["question"],
                        "ground_truth": result["ground_truth"],
                        "prediction": result["extracted_answer"],
                        "model_output": result["model_output"],
                    }
                )
        per_dataset[path.name] = dict(dataset_counter)

    report = {
        "overall_counts": dict(overall_counter),
        "per_dataset_counts": per_dataset,
        "example_failures": dict(examples_by_type),
        "notes": [
            "WRONG_TARGET and ATTRIBUTE_ERROR cannot be cleanly separated from prediction-vs-GT alone.",
            "LOW_OVERLAP_SEMANTIC_OR_WRONG_TARGET is a conservative proxy bucket.",
        ],
    }
    Path(args.output_json).write_text(json.dumps(report, indent=2, ensure_ascii=False))

    total_failures = sum(v for k, v in overall_counter.items() if k != "CORRECT")
    lines = [
        "# Diagnostic Report",
        "",
        "## Reliability Notes",
        "",
        "- This diagnosis is heuristic and reproducible, but not omniscient.",
        "- `LOW_OVERLAP_SEMANTIC_OR_WRONG_TARGET` is a conservative bucket because the current files do not include all candidate object annotations.",
        "",
        "## Overall Failure Counts",
        "",
        "| type | count | ratio_over_failures |",
        "|---|---:|---:|",
    ]
    for k, v in overall_counter.most_common():
        if k == "CORRECT":
            continue
        ratio = v / total_failures if total_failures else 0.0
        lines.append(f"| {k} | {v} | {ratio:.4f} |")

    lines.extend(["", "## Per Dataset Counts", ""])
    for dataset, counts in per_dataset.items():
        lines.append(f"### {dataset}")
        lines.append("")
        lines.append("| type | count |")
        lines.append("|---|---:|")
        for k, v in sorted(counts.items()):
            lines.append(f"| {k} | {v} |")
        lines.append("")

    Path(args.output_md).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
