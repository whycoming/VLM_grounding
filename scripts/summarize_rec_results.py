import argparse
import json
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


def summarize_file(path: Path):
    payload = json.loads(path.read_text())
    results = payload["results"]
    ious = [iou(r["extracted_answer"], r["ground_truth"]) for r in results]
    acc05 = sum(v >= 0.5 for v in ious) / len(ious) if ious else 0.0
    acc075 = sum(v >= 0.75 for v in ious) / len(ious) if ious else 0.0
    mean_iou = sum(ious) / len(ious) if ious else 0.0
    return {
        "file": path.name,
        "dataset": path.name.replace("rec_results_", "").rsplit("_", 1)[0],
        "num_samples": len(ious),
        "acc@0.5": acc05,
        "acc@0.75": acc075,
        "mean_iou": mean_iou,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    rows = [summarize_file(path) for path in sorted(input_dir.glob("rec_results_*.json"))]

    Path(args.output_json).write_text(json.dumps(rows, indent=2))

    lines = [
        "# Baseline Summary",
        "",
        "| dataset | samples | acc@0.5 | acc@0.75 | mean_iou |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['num_samples']} | {row['acc@0.5']:.4f} | {row['acc@0.75']:.4f} | {row['mean_iou']:.4f} |"
        )
    Path(args.output_md).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
