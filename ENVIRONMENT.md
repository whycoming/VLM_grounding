# Environment Status

## Virtual Environment

- Path: `/mnt/VLM_grounding/.venv`
- Python: `3.10.12`

Activate with:

```bash
cd /mnt/VLM_grounding
source .venv/bin/activate
```

## Installed Core Packages

- `torch==2.5.1+cu124`
- `torchvision==0.20.1+cu124`
- `torchaudio==2.5.1+cu124`
- `transformers==5.5.0`
- `accelerate==1.13.0`
- `peft==0.18.1`
- `bitsandbytes==0.49.2`
- `deepspeed==0.18.9`
- `trl==1.1.0.dev0`
- `qwen-vl-utils==0.0.14`
- `wandb==0.25.1`
- `matplotlib==3.10.8`
- `seaborn==0.13.2`

## Verified

- `torch.cuda.is_available()` returns `True`
- CUDA device detected: `NVIDIA GeForce RTX 4090`
- CUDA tensor matmul runs successfully
- Python imports succeed for:
  - `transformers`
  - `accelerate`
  - `peft`
  - `bitsandbytes`
  - `deepspeed`
  - `trl`
  - `qwen_vl_utils`
  - `wandb`
- Hugging Face CLI available via:

```bash
hf
```

## Repositories

- `VLM-R1` cloned to `/mnt/VLM_grounding/VLM-R1`
- Current commit: `67bc01f`

## Pending

- `flash-attn` was not completed. The build stalled during wheel compilation and was stopped.
- Model weights and datasets from `plan.md` are not downloaded yet.
- `wandb login` is not configured yet.
