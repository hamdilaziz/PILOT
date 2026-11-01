from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm.auto import tqdm

IGNORE_INDEX = -100


class SequenceDataset(Dataset):
    """JSONL records with image, prompt and target; image paths are manifest-relative."""

    def __init__(self, manifest, tokenizer, mean, std, max_length):
        self.manifest = Path(manifest).resolve()
        self.mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
        if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or np.any(self.std <= 0):
            raise ValueError("Preprocessing mean/std must be finite RGB values with std > 0.")
        if any(getattr(tokenizer, name) is None for name in ("bos_token_id", "eos_token_id", "pad_token_id")):
            raise ValueError("The tokenizer must define BOS, EOS and PAD.")
        self.samples = []
        self.image_paths = set()
        lines = self.manifest.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(tqdm(lines, desc=f"Check {self.manifest.name}"), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or any(not isinstance(row.get(k), str) for k in ("image", "prompt", "target")):
                raise ValueError(f"{self.manifest}:{line_number}: expected string image, prompt and target.")
            image = (self.manifest.parent / row["image"]).resolve()
            if not image.is_file():
                raise FileNotFoundError(image)
            prompt_ids = tokenizer.encode(row["prompt"], add_special_tokens=False)
            target_ids = tokenizer.encode(row["target"], add_special_tokens=False)
            reserved = {tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id}
            if reserved.intersection(prompt_ids + target_ids):
                raise ValueError(f"{self.manifest}:{line_number}: do not include BOS, EOS or PAD in prompt/target.")
            if not prompt_ids:
                raise ValueError(f"{self.manifest}:{line_number}: prompt is empty.")
            if tokenizer.unk_token_id is not None and tokenizer.unk_token_id in prompt_ids + target_ids:
                raise ValueError(f"{self.manifest}:{line_number}: unknown token; check the tokenizer and extra_tokens.")
            ids = [tokenizer.bos_token_id, *prompt_ids, *target_ids, tokenizer.eos_token_id]
            if len(ids) > max_length:
                raise ValueError(f"{self.manifest}:{line_number}: {len(ids)} tokens exceed max_length={max_length}.")
            self.samples.append({**row, "image": image, "ids": ids, "prompt_length": len(prompt_ids)})
            self.image_paths.add(image)
        if not self.samples:
            raise ValueError(f"{self.manifest}: no training examples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        with Image.open(sample["image"]) as image:
            pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
        pixels = (pixels - self.mean) / self.std
        return {**sample, "pixels": torch.from_numpy(pixels).permute(2, 0, 1).contiguous()}


class SequenceCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, samples):
        shapes = {tuple(sample["pixels"].shape) for sample in samples}
        if len(shapes) != 1:
            raise ValueError("Images in one batch must have equal dimensions. Use batch_size=1 and accumulation_steps.")
        length = max(len(sample["ids"]) for sample in samples)
        full = torch.full((len(samples), length), self.pad_token_id, dtype=torch.long)
        for i, sample in enumerate(samples):
            full[i, :len(sample["ids"])] = torch.tensor(sample["ids"])
        inputs = full[:, :-1].clone()
        labels = full[:, 1:].clone()
        for i, sample in enumerate(samples):
            labels[i, :sample["prompt_length"]] = IGNORE_INDEX
            labels[i, len(sample["ids"]) - 1:] = IGNORE_INDEX
        return {
            "images": torch.stack([sample["pixels"] for sample in samples]),
            "input_ids": inputs,
            "attention_mask": inputs.ne(self.pad_token_id).long(),
            "labels": labels,
            "samples": samples,
        }


def corrupt_inputs(input_ids, attention_mask, samples, tokenizer, rate, strategy="random_vocab"):
    """Vectorized lexical input noise; prompt, structural tokens and labels stay clean."""
    if rate == 0:
        return input_ids
    if strategy not in {"sequence", "random_vocab"}:
        raise ValueError("noise_strategy must be sequence or random_vocab.")
    positions = torch.arange(input_ids.shape[1], device=input_ids.device)[None, :]
    starts = torch.tensor([s["prompt_length"] + 1 for s in samples], device=input_ids.device)[:, None]
    protected = torch.tensor(tokenizer.all_special_ids, device=input_ids.device)
    eligible = attention_mask.bool() & (positions >= starts) & ~torch.isin(input_ids, protected)
    replace = eligible & (torch.rand(input_ids.shape, device=input_ids.device) < rate)
    if strategy == "sequence":
        weights = eligible.float()
        weights[:, 0] += weights.sum(dim=1).eq(0).float()
        sampled = torch.multinomial(weights, input_ids.shape[1], replacement=True)
        noise = input_ids.gather(1, sampled)
    else:
        allowed = torch.arange(len(tokenizer), device=input_ids.device)
        allowed = allowed[~torch.isin(allowed, protected)]
        if not len(allowed):
            return input_ids
        noise = allowed[torch.randint(len(allowed), input_ids.shape, device=input_ids.device)]
    return torch.where(replace, noise, input_ids)
