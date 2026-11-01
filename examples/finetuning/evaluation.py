from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from tqdm.auto import tqdm

from pilot.training.metrics import field_metrics as field_metrics, parse_fields as parse_fields

from .common import read_jsonl


def load_best(config_path):
    from pilot.loading import load_pilot_model

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return load_pilot_model(config_path, device)


@torch.inference_mode()
def predict_records(model, tokenizer, model_config, manifest, *, max_length, limit=32):
    manifest = Path(manifest)
    records = read_jsonl(manifest)
    if limit is not None:
        records = records[:limit]
    device = next(model.parameters()).device
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    mean = np.asarray(model_config["preprocessing"]["mean"], dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(model_config["preprocessing"]["std"], dtype=np.float32).reshape(1, 1, 3)
    model.eval()
    predictions = []
    for row in tqdm(records, desc="Predict held-out examples"):
        with Image.open(manifest.parent / row["image"]) as image:
            pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
        images = torch.from_numpy((pixels - mean) / std).permute(2, 0, 1).unsqueeze(0).to(device)
        prefix = [tokenizer.bos_token_id, *tokenizer.encode(row["prompt"], add_special_tokens=False)]
        capacity = min(max_length, model.decoder.model.config.max_position_embeddings) - len(prefix)
        if capacity < 1:
            raise ValueError("The generation limit leaves no room after the prompt.")
        with torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda"):
            generated = model.generate(
                images, torch.tensor([prefix], device=device), max_new_tokens=capacity, use_cache=True,
                num_beams=1, do_sample=False, repetition_penalty=1.0,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            )[0, len(prefix):].tolist()
        terminated = tokenizer.eos_token_id in generated
        if terminated:
            generated = generated[:generated.index(tokenizer.eos_token_id)]
        text = tokenizer.decode(generated, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        predictions.append({**row, "prediction": text, "terminated": terminated})
    return predictions


def history_table(run_dir):
    import pandas as pd

    return pd.json_normalize(read_jsonl(Path(run_dir) / "metrics.jsonl"))
