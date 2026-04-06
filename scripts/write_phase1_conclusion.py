import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--diagnostic-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    summary = json.loads(Path(args.summary_json).read_text())
    diagnostic = json.loads(Path(args.diagnostic_json).read_text())

    overall = diagnostic.get("overall_counts", {})
    total_failures = sum(v for k, v in overall.items() if k != "CORRECT")

    def ratio(key):
        return overall.get(key, 0) / total_failures if total_failures else 0.0

    mean_iou = sum(row["mean_iou"] for row in summary) / len(summary) if summary else 0.0
    acc05 = sum(row["acc@0.5"] for row in summary) / len(summary) if summary else 0.0

    top_failure = None
    top_count = -1
    for k, v in overall.items():
        if k == "CORRECT":
            continue
        if v > top_count:
            top_failure = k
            top_count = v

    lines = [
        "# Phase 1 Conclusion",
        "",
        "## Reliability Boundary",
        "",
        "- This conclusion is based on reproducible result files and deterministic post-processing scripts.",
        "- It is still a heuristic diagnosis, not an oracle. In particular, `LOW_OVERLAP_SEMANTIC_OR_WRONG_TARGET` is a conservative proxy bucket.",
        "",
        "## Baseline",
        "",
        f"- Mean `acc@0.5` across evaluated splits: `{acc05:.4f}`",
        f"- Mean `mean_iou` across evaluated splits: `{mean_iou:.4f}`",
        "",
        "## Main Finding",
        "",
        f"- The dominant residual failure bucket is `{top_failure}` with count `{top_count}`.",
        f"- `FORMAT_ERROR` failure ratio: `{ratio('FORMAT_ERROR'):.4f}`",
        f"- `COORDINATE_IMPRECISION` failure ratio: `{ratio('COORDINATE_IMPRECISION'):.4f}`",
        f"- `SPATIAL_REASONING_ERROR` failure ratio: `{ratio('SPATIAL_REASONING_ERROR'):.4f}`",
        f"- `LOW_OVERLAP_SEMANTIC_OR_WRONG_TARGET` failure ratio: `{ratio('LOW_OVERLAP_SEMANTIC_OR_WRONG_TARGET'):.4f}`",
        "",
        "## Decision",
        "",
        "- Proceed to Phase 2 SFT baseline. This is justified regardless of the final Phase 3 method choice, because SFT is the controlled next-stage baseline in the plan.",
        "- Do not over-claim which RL method should win yet. That decision should wait until the Phase 2 residual-error profile is available.",
        "",
        "## Next Step",
        "",
        "- Start Phase 2 local SFT and preserve logs/checkpoints for a second-round diagnosis.",
    ]

    Path(args.output_md).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
