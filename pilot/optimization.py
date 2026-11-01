from __future__ import annotations

import torch


@torch.inference_mode(False)
def optimize_for_cpu(model, *, num_threads=None, compile_encoder=False, positional_cache_size=3):
    """Prepare a loaded model in place; benchmark and check task metrics before deployment."""
    if num_threads is not None:
        if num_threads < 1:
            raise ValueError("num_threads must be positive.")
        torch.set_num_threads(num_threads)
    if positional_cache_size < 0:
        raise ValueError("positional_cache_size must be nonnegative.")
    model.to(device="cpu", dtype=torch.float32).eval()
    model.encoder.to(memory_format=torch.channels_last)
    model.cpu_channels_last = True
    model.positional_encoding_cache_size = positional_cache_size
    model._positional_encoding_cache.clear()
    if compile_encoder:
        model.encoder = torch.compile(model.encoder, dynamic=True)
    return model
