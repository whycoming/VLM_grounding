import json
import logging
import math
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Any, Optional

import datasets
import torch
import torch.nn.functional as F
import transformers
import yaml
from PIL import Image
from peft import get_peft_model, prepare_model_for_kbit_training
from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset
from transformers import AutoProcessor, Trainer, set_seed
from transformers.trainer_utils import get_last_checkpoint

from open_r1.configs import OnlineDPOConfig
from open_r1.utils.callbacks import get_callbacks
from trl import ModelConfig, ScriptArguments, TrlParser, get_kbit_device_map, get_peft_config, get_quantization_config

logger = logging.getLogger(__name__)
processor = None


@dataclass
class OnlineDPOScriptArguments(ScriptArguments):
    image_root: str = field(default=None, metadata={"help": "The root directory of the image."})


@dataclass
class OnlineDPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = False


class LazyPreferenceDataset(Dataset):
    def __init__(self, data_path: str, script_args: OnlineDPOScriptArguments):
        super().__init__()
        self.script_args = script_args
        self.list_data_dict = []

        if not data_path.endswith(".yaml"):
            raise ValueError(f"Unsupported file type: {data_path}")

        with open(data_path, "r") as file:
            yaml_data = yaml.safe_load(file)
        for data in yaml_data.get("datasets", []):
            json_path = data.get("json_path")
            sampling_strategy = data.get("sampling_strategy", "all")
            sampling_number = None

            if json_path.endswith(".jsonl"):
                cur_data_dict = []
                with open(json_path, "r") as json_file:
                    for line in json_file:
                        cur_data_dict.append(json.loads(line.strip()))
            elif json_path.endswith(".json"):
                with open(json_path, "r") as json_file:
                    cur_data_dict = json.load(json_file)
            else:
                raise ValueError(f"Unsupported file type: {json_path}")

            if ":" in sampling_strategy:
                sampling_strategy, sampling_number = sampling_strategy.split(":")
                if "%" in sampling_number:
                    sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
                else:
                    sampling_number = int(sampling_number)

            if sampling_strategy == "first" and sampling_number is not None:
                cur_data_dict = cur_data_dict[:sampling_number]
            elif sampling_strategy == "end" and sampling_number is not None:
                cur_data_dict = cur_data_dict[-sampling_number:]
            elif sampling_strategy == "random" and sampling_number is not None:
                random.shuffle(cur_data_dict)
                cur_data_dict = cur_data_dict[:sampling_number]

            print(f"Loaded {len(cur_data_dict)} samples from {json_path}")
            self.list_data_dict.extend(cur_data_dict)

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        example = self.list_data_dict[i]
        image_path = os.path.join(self.script_args.image_root, example["image"])
        row = {
            "image_path": image_path,
            "prompt": example["prompt"],
            "chosen": example["chosen"],
            "rejected": example["rejected"],
            "chosen_iou": example.get("chosen_iou"),
            "rejected_iou": example.get("rejected_iou"),
            "iou_gap": example.get("iou_gap"),
        }
        if "ref_chosen_logps" in example:
            row["ref_chosen_logps"] = float(example["ref_chosen_logps"])
        if "ref_rejected_logps" in example:
            row["ref_rejected_logps"] = float(example["ref_rejected_logps"])
        return row


def build_messages(image_path: str, prompt: str, bbox: Optional[list[float]] = None):
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
        bbox_text = f"```json\n[{int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}]\n```"
        messages.append({"role": "assistant", "content": bbox_text})
    return messages


class VisionPreferenceCollator:
    def __init__(self, processing_class, max_prompt_length: int, max_completion_length: int):
        self.processor = processing_class
        self.max_prompt_length = max_prompt_length
        self.max_completion_length = max_completion_length

    def _prepare_sequence(self, example: dict[str, Any], bbox: list[float]):
        prompt_messages = build_messages(example["image_path"], example["prompt"])
        full_messages = build_messages(example["image_path"], example["prompt"], bbox=bbox)

        prompt_text = self.processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        full_text = self.processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)

        prompt_images, prompt_videos = process_vision_info(prompt_messages)
        full_images, full_videos = process_vision_info(full_messages)

        return {
            "prompt_text": prompt_text,
            "full_text": full_text,
            "prompt_images": prompt_images,
            "prompt_videos": prompt_videos,
            "full_images": full_images,
            "full_videos": full_videos,
        }

    @staticmethod
    def _processor_call(processor, texts, images, videos):
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

    def __call__(self, examples: list[dict[str, Any]]):
        chosen_sequences = [self._prepare_sequence(example, example["chosen"]) for example in examples]
        rejected_sequences = [self._prepare_sequence(example, example["rejected"]) for example in examples]
        sequences = chosen_sequences + rejected_sequences

        prompt_batch = self._processor_call(
            self.processor,
            [row["prompt_text"] for row in sequences],
            [row["prompt_images"] for row in sequences],
            [row["prompt_videos"] for row in sequences],
        )
        full_batch = self._processor_call(
            self.processor,
            [row["full_text"] for row in sequences],
            [row["full_images"] for row in sequences],
            [row["full_videos"] for row in sequences],
        )

        prompt_lengths = prompt_batch["attention_mask"].sum(dim=1)
        prompt_lengths = torch.clamp(prompt_lengths, max=self.max_prompt_length)

        input_ids = full_batch["input_ids"]
        attention_mask = full_batch["attention_mask"]
        max_total_length = min(input_ids.shape[1], self.max_prompt_length + self.max_completion_length)
        input_ids = input_ids[:, :max_total_length]
        attention_mask = attention_mask[:, :max_total_length]

        completion_mask = torch.zeros_like(attention_mask)
        for idx, prompt_len in enumerate(prompt_lengths.tolist()):
            prompt_len = min(prompt_len, max_total_length)
            seq_len = int(attention_mask[idx].sum().item())
            completion_mask[idx, prompt_len:seq_len] = 1

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "completion_mask": completion_mask,
        }
        if "ref_chosen_logps" in examples[0]:
            batch["ref_chosen_logps"] = torch.tensor([example["ref_chosen_logps"] for example in examples], dtype=torch.float32)
        if "ref_rejected_logps" in examples[0]:
            batch["ref_rejected_logps"] = torch.tensor([example["ref_rejected_logps"] for example in examples], dtype=torch.float32)
        sequence_keys = {"input_ids", "attention_mask", "mm_token_type_ids"}
        for key, value in full_batch.items():
            if key not in batch:
                if key in sequence_keys and value.ndim == 2:
                    batch[key] = value[:, :max_total_length]
                else:
                    batch[key] = value
        return batch


class OnlineDPOTrainer(Trainer):
    def __init__(self, *args, ref_model=None, beta: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.ref_model = ref_model
        self.beta = beta
        if self.ref_model is not None and not hasattr(self.ref_model, "hf_device_map"):
            self._move_model_to_device(self.ref_model, self.args.device)
        if self.ref_model is not None:
            self.ref_model.eval()

    @staticmethod
    def _sequence_logps(logits, input_ids, completion_mask):
        shifted_logits = logits[:, :-1, :]
        shifted_input_ids = input_ids[:, 1:]
        shifted_mask = completion_mask[:, 1:].to(logits.dtype)
        token_logps = F.log_softmax(shifted_logits, dim=-1).gather(
            dim=-1, index=shifted_input_ids.unsqueeze(-1)
        ).squeeze(-1)
        return (token_logps * shifted_mask).sum(dim=-1)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        model_inputs = {
            k: v for k, v in inputs.items() if k not in {"completion_mask", "ref_chosen_logps", "ref_rejected_logps"}
        }
        completion_mask = inputs["completion_mask"]

        outputs = model(**model_inputs)
        policy_logps = self._sequence_logps(outputs.logits, inputs["input_ids"], completion_mask)

        pair_count = inputs["input_ids"].shape[0] // 2
        policy_chosen_logps = policy_logps[:pair_count]
        policy_rejected_logps = policy_logps[pair_count:]
        if "ref_chosen_logps" in inputs and "ref_rejected_logps" in inputs:
            ref_chosen_logps = inputs["ref_chosen_logps"].to(policy_logps.device)
            ref_rejected_logps = inputs["ref_rejected_logps"].to(policy_logps.device)
        else:
            with torch.no_grad():
                ref_outputs = self.ref_model(**model_inputs)
                ref_logps = self._sequence_logps(ref_outputs.logits, inputs["input_ids"], completion_mask)
            ref_chosen_logps = ref_logps[:pair_count]
            ref_rejected_logps = ref_logps[pair_count:]

        logits = (policy_chosen_logps - policy_rejected_logps) - (ref_chosen_logps - ref_rejected_logps)
        losses = -F.logsigmoid(self.beta * logits)
        loss = losses.mean()

        metrics = {
            "rewards/chosen": float((self.beta * (policy_chosen_logps - ref_chosen_logps)).mean().detach().cpu().item()),
            "rewards/rejected": float((self.beta * (policy_rejected_logps - ref_rejected_logps)).mean().detach().cpu().item()),
            "rewards/accuracy": float((logits > 0).float().mean().detach().cpu().item()),
            "logps/chosen": float(policy_chosen_logps.mean().detach().cpu().item()),
            "logps/rejected": float(policy_rejected_logps.mean().detach().cpu().item()),
        }
        self.log(metrics)

        if return_outputs:
            outputs.loss = loss
            return loss, outputs
        return loss


def disable_dropout_in_model(model):
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0


def load_model(model_args, training_args, is_ref_model: bool):
    dtype_name = getattr(model_args, "torch_dtype", None)
    if dtype_name is None:
        dtype_name = getattr(model_args, "dtype", None)
    torch_dtype = dtype_name if dtype_name in ["auto", None] else getattr(torch, dtype_name)
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration

    model_name_lower = model_args.model_name_or_path.lower()
    if "qwen2-vl" in model_name_lower:
        model_cls = Qwen2VLForConditionalGeneration
    elif "qwen2.5-vl" in model_name_lower or "qwen25vl" in model_name_lower:
        model_cls = Qwen2_5_VLForConditionalGeneration
    else:
        raise ValueError(f"Unsupported model: {model_args.model_name_or_path}")

    model = model_cls.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    model.config.use_cache = False if training_args.gradient_checkpointing else True

    if training_args.disable_dropout:
        disable_dropout_in_model(model)

    if is_ref_model:
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        return model

    peft_config = get_peft_config(model_args)
    vision_modules_keywords = ["visual"]
    if peft_config is not None:
        if quantization_config is not None:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)
        if not getattr(peft_config, "target_modules", None):
            target_modules = set()
            for name, module in model.named_modules():
                if any(keyword in name for keyword in vision_modules_keywords):
                    continue
                if isinstance(module, torch.nn.Linear) and "embed_tokens" not in name:
                    target_modules.add(name)
            peft_config.target_modules = list(target_modules)
        model = get_peft_model(model, peft_config)

    if model_args.freeze_vision_modules:
        for name, param in model.named_parameters():
            if any(keyword in name for keyword in vision_modules_keywords):
                param.requires_grad = False

    return model


def main(script_args, training_args, model_args):
    set_seed(training_args.seed)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    dataset = LazyPreferenceDataset(script_args.dataset_name, script_args)

    global processor
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )
    if hasattr(processor, "pad_token") and processor.pad_token is None:
        processor.pad_token = processor.eos_token
    elif hasattr(processor, "tokenizer") and processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    policy_model = load_model(model_args, training_args, is_ref_model=False)
    ref_model = None
    if len(dataset) > 0:
        first_row = dataset[0]
        has_ref_logps = "ref_chosen_logps" in first_row and "ref_rejected_logps" in first_row
    else:
        has_ref_logps = False
    if not has_ref_logps:
        ref_model = load_model(model_args, training_args, is_ref_model=True)

    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    trainer = OnlineDPOTrainer(
        model=policy_model,
        ref_model=ref_model,
        beta=training_args.beta,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        processing_class=processor,
        data_collator=VisionPreferenceCollator(
            processing_class=processor,
            max_prompt_length=training_args.max_prompt_length,
            max_completion_length=training_args.max_completion_length,
        ),
        callbacks=get_callbacks(training_args, model_args),
    )

    logger.info("*** Train ***")
    checkpoint = training_args.resume_from_checkpoint if training_args.resume_from_checkpoint is not None else last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    parser = TrlParser((OnlineDPOScriptArguments, OnlineDPOConfig, OnlineDPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
