import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--attn-implementation", default="sdpa")
    args = parser.parse_args()

    adapter_path = Path(args.adapter_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter_config = json.loads((adapter_path / "adapter_config.json").read_text())
    base_model_path = adapter_config["base_model_name_or_path"]

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model = model.merge_and_unload()
    model.save_pretrained(str(output_dir))

    processor = AutoProcessor.from_pretrained(str(adapter_path))
    processor.save_pretrained(str(output_dir))

    print(f"Merged model written to {output_dir}")


if __name__ == "__main__":
    main()
