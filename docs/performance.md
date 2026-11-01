# Performance and compatibility

## Tested environment

The CPU regression suite uses PyTorch `2.6.0+cpu`, Transformers `4.57.6` and Python `3.12`.
It combines a reduced convolutional encoder with a small real mBART decoder to test training,
generation, tokenizer extension, `.pt` export/reload and interruption recovery. Separate block tests
check GroupNorm output and gradient equivalence against the former normalization.
These tests do not establish accuracy or throughput for released full-size checkpoints.

Dependencies constrain Transformers to `>=4.57,<5`. Generation uses the library's native input
preparation and an explicit `EncoderDecoderCache`, with separate self-attention and image
cross-attention KV caches. Tests compare cached/uncached multi-token-prompt decoding and verify
that visual keys/values are projected once per generation. Version upgrades require regression
tests against actual pretrained checkpoints; this is not a claim of compatibility with version 5.
See the [Transformers cache documentation](https://huggingface.co/docs/transformers/v4.57.1/kv_cache).

## Encoder normalization and training

Encoder blocks now use `GroupNorm(C, C, eps=1e-3, affine=False)`. There is one group per channel,
so mean and biased variance are computed over the same spatial values as the former
`InstanceNorm2d(C, eps=1e-3, affine=False, track_running_stats=False)` in both training and evaluation.
This preserves the mathematical normalization and adds no parameters or running statistics:
existing `.pt` state dictionaries retain their keys and shapes. Floating-point results need not
be bitwise identical across kernels, precisions or devices. Changing the number of groups or using
affine parameters would change this equivalence and is not done here.
See [PyTorch GroupNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html).

Other included changes:

- CUDA AMP uses BF16 where supported and FP16 with GradScaler on other CUDA GPUs.
- Sequence padding follows the longest target in the microbatch; tokenization is cached by the dataset.
- Lexical teacher-forcing noise uses tensor operations, with same-sequence or vocabulary sampling.
- Gradient accumulation is normalized by supervised token count, including a partial final window.
- Decoder activation checkpointing is optional.
- Encoder finite checks are opt-in through `encoder.check_finite=true`, avoiding repeated device
  synchronization. Training still checks loss and gradients.
- Evaluation avoids random draws for disabled dropout and uses generated CER/F1 without validation loss.

Existing encoder FP32 regions remain for stability. The historical `encoder.use_checkpointing`
flag is inactive; the training option enables checkpointing in the decoder only.
The objective is ordinary masked token cross-entropy; category-balanced variants require a separate
validation experiment. See the [training guide](training.md) for data and resolution considerations.

## Simple CPU optimization

Both notebooks benchmark the following runtime options after training:

```python
import torch
from pilot.loading import load_pilot_model
from pilot.optimization import optimize_for_cpu

model, tokenizer, config = load_pilot_model("runs/example/training/best/config.json", torch.device("cpu"))
model = optimize_for_cpu(model, num_threads=4, compile_encoder=False)
with torch.inference_mode():
    output_ids = model.generate(images, prompt_ids, max_new_tokens=128, use_cache=True)
```

`images` must use the trained model's preparation/normalization policy and `prompt_ids` its tokenizer.
The helper changes the loaded model in place: CPU FP32, evaluation mode, channels-last encoder
weights/inputs, and an LRU positional-encoding cache with at most three feature-map shapes.
Cached positional tensors are used only in evaluation without gradients and are absent from
checkpoints. Clear/disable the cache when memory is more important than latency.
The `num_threads` option sets PyTorch's process-wide intra-op thread count. Tune it to available
CPU resources and concurrent requests; more threads can be slower. Avoid using more workers and
threads than the deployment can support.

Set `compile_encoder=True` to optionally use `torch.compile(..., dynamic=True)` on the encoder.
It requires a supported toolchain, incurs startup cost and can compile new graphs for new shapes.
The notebook benchmark warms both variants, excludes warm-up and image I/O, uses identical inputs,
decoding settings and thread count, then reports per-image medians and token equality.
If compilation/measurement fails, that benchmark restores the eager encoder and raises the error.
The standalone helper does not hide compilation errors. The benchmark restores the previous thread
count; set the desired count explicitly when deploying the optimized model.
Runtime preparation should be repeated after loading the ordinary `.pt` export, rather than saving
compiled-module state as a new training checkpoint.

Combining normalization, layout, caching, thread tuning and compilation can sometimes yield an
inference speedup around **3×** on suitable CPU workloads. This is an indicative possibility, not
a measured result for these examples or a hardware-independent guarantee. Some options may regress
latency. Quantization, batching, shape bucketing and alternative engines are further experiments.
Validate CER/F1 again, especially for approximations such as quantization.

For deployment, compare representative image sizes and generated lengths on the actual CPU.
Report cold-start cost, warmed latency distribution, throughput under concurrency, peak memory,
versions and thread counts. Token equality on one sample is a sanity check, not a quality study.
Useful upstream references are the [channels-last tutorial](https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html),
[compilation tutorial](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) and
[CPU threading API](https://docs.pytorch.org/docs/stable/generated/torch.set_num_threads.html).

## Full-checkpoint and GPU validation

Record checkpoint, tokenizer, image sizes, sequence lengths, precision, batch size, hardware,
versions and warm-up policy. Compare training/generation time, peak memory and held-out task metrics.
BF16/FP16 training and released-checkpoint OCR/extraction quality require a separate GPU run.
The accumulation/scaling policy follows the [PyTorch AMP examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html).
