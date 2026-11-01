from __future__ import annotations

import os
from pathlib import Path
import platform
from statistics import median
import time

import numpy as np
from PIL import Image
import torch
from tqdm.auto import tqdm

from pilot.optimization import optimize_for_cpu

from .common import read_jsonl


def available_cpus():
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1


@torch.no_grad()
def benchmark_cpu(model, tokenizer, model_config, manifest, *, max_length, num_threads=4,
                  compile_encoder=False, examples=1, repeats=3):
    """Compare warmed model latency on identical cached inputs, excluding image I/O."""
    if examples < 1 or repeats < 1 or num_threads < 1:
        raise ValueError("examples, repeats and num_threads must be positive.")

    manifest = Path(manifest)
    records = read_jsonl(manifest)[:examples]
    if not records:
        raise ValueError("The CPU benchmark needs at least one example.")
    mean = np.asarray(model_config["preprocessing"]["mean"], dtype=np.float32)
    std = np.asarray(model_config["preprocessing"]["std"], dtype=np.float32)
    inputs = []
    for row in records:
        with Image.open(manifest.parent / row["image"]) as image:
            pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
        image = torch.from_numpy((pixels - mean) / std).permute(2, 0, 1).unsqueeze(0)
        prefix = torch.tensor([[tokenizer.bos_token_id, *tokenizer.encode(row["prompt"], add_special_tokens=False)]])
        capacity = min(max_length, model.decoder.model.config.max_position_embeddings) - prefix.shape[1]
        if capacity < 1:
            raise ValueError("The generation limit leaves no room after the prompt.")
        inputs.append((image, prefix, capacity))

    model.to(device="cpu", dtype=torch.float32).eval()
    model.encoder = getattr(model.encoder, "_orig_mod", model.encoder)
    model.encoder.to(memory_format=torch.contiguous_format)
    model.cpu_channels_last = False
    model.positional_encoding_cache_size = 0
    model._positional_encoding_cache.clear()
    original_threads = torch.get_num_threads()
    baseline_threads = min(num_threads, available_cpus())
    torch.set_num_threads(baseline_threads)
    eager_encoder = model.encoder

    @torch.inference_mode()
    def measure(label):
        predictions, latencies = [], []
        for image, prefix, capacity in tqdm(inputs, desc=label):
            def generate():
                return model.generate(image, prefix, max_new_tokens=capacity, num_beams=1, do_sample=False,
                                      use_cache=True, repetition_penalty=1.0, pad_token_id=tokenizer.pad_token_id,
                                      eos_token_id=tokenizer.eos_token_id)

            generate()
            durations = []
            for _ in range(repeats):
                start = time.perf_counter()
                generated = generate()
                durations.append(time.perf_counter() - start)
            predictions.append(generated.tolist())
            latencies.append(median(durations))
        return predictions, latencies

    try:
        baseline, baseline_times = measure("CPU eager baseline")
        optimize_for_cpu(model, num_threads=baseline_threads, compile_encoder=compile_encoder)
        optimized, optimized_times = measure("CPU optimized")
    except Exception:
        model.encoder = eager_encoder
        model.encoder.to(memory_format=torch.contiguous_format)
        model.cpu_channels_last = False
        model.positional_encoding_cache_size = 0
        model._positional_encoding_cache.clear()
        raise
    finally:
        torch.set_num_threads(original_threads)
    rows = [{"image": row["image"], "height": image.shape[-2], "width": image.shape[-1],
             "baseline_seconds": base, "optimized_seconds": fast, "speedup": base / fast,
             "baseline_new_tokens": len(ref[0]) - prefix.shape[1],
             "optimized_new_tokens": len(pred[0]) - prefix.shape[1], "same_token_ids": ref == pred}
            for row, (image, prefix, _), base, fast, ref, pred
            in zip(records, inputs, baseline_times, optimized_times, baseline, optimized)]
    return {"threads": baseline_threads, "repeats": repeats, "compile_encoder": compile_encoder,
            "examples": len(rows), "speedup": sum(baseline_times) / sum(optimized_times),
            "same_token_ids": baseline == optimized, "cpu": platform.processor() or platform.machine(),
            "available_cpus": available_cpus(), "torch": str(torch.__version__), "per_example": rows}
