# Fine-tune PILOT on a new dataset

This recipe adapts a released checkpoint to new documents or a new extraction schema.
It keeps the checkpoint's encoder, decoder positional capacity and tokenizer.
The paper describes the original multi-stage pretraining curriculum; reproducing it requires
its data preparation.

## 1. Prepare the base model

Follow the [README](../README.md) to install dependencies and download weights.
Start from `pilot_generic` for a new domain. Check that its inference command works first.
Match the `.pt` filename in `configs/pilot_generic.json` to your download.

## 2. Prepare training and validation JSONL files

```text
data/
  train.jsonl
  valid.jsonl
  images/
    train_001.png
    valid_001.png
```

Each line contains one example:

```json
{"image": "images/train_001.png", "prompt": "<ocr>", "target": "Invoice number 123\nTotal 25.00"}
```

Paths are relative to the JSONL file. Different prompts may share an image within a split.
Split by document before creating task examples. The trainer rejects image paths shared between
training and validation; copies stored under other paths and near-duplicate images need separate checks.
Keep a held-out test set for evaluation after model selection.

The trainer reads existing images as RGB without resizing or rotating them.
Use the same preparation for inference. For layout tasks, upstream image transformations must
also transform the box coordinates.

| Task | Example prompt | Target |
| --- | --- | --- |
| OCR | `<ocr>` | Text in reading order |
| OCR with layout | `<ocr_with_boxes>` | Text interleaved with box tokens |
| Spotting | `<find_it> Total` | Coordinates of matching occurrences |
| Region OCR | `<ocr_on_box> <x_1><y_2><x_20><y_8>` | Text inside the region |
| Custom extraction | `<extract>` | A consistent tagged field sequence |

A two-region `<ocr_with_boxes>` target has this form:

```text
<x_1><y_2> Invoice<x_15><y_5><sep/><x_1><y_6> Total 25.00<x_20><y_9>
```

Each region contains top-left coordinates, text, then bottom-right coordinates. `<sep/>`
separates regions. Follow the checkpoint's reading order and coordinate convention.
With `coord_bin_size=10`, convert pixels to bins using integer division by 10.
All coordinate tokens must exist in the tokenizer; check the range rather than silently clipping boxes.
For spotting, concatenate box quadruples in target order, following the released examples.

Do not add BOS, EOS or PAD to records. The trainer builds `BOS + prompt + target + EOS`,
shifts once and masks prompt/padding positions from the loss. Unknown tokens and overlong targets
fail before training. `max_length` includes BOS, prompt, target and EOS, and cannot exceed
the pretrained decoder capacity. Targets are never silently truncated.

## 3. Add an extraction task

Example using synthetic values:

```json
{"image": "images/train_001.png", "prompt": "<extract>", "target": "<vendor>Example Shop</vendor><total>25.00</total>"}
```

Set these options in the training config:

```json
"extra_tokens": ["<extract>", "<vendor>", "</vendor>", "<total>", "</total>"],
"selection_metric": "field_micro_f1",
"metric_fields": ["vendor", "total"]
```

Weights are restored **before** vocabulary extension. Existing IDs and special-token registrations
are preserved. New tokens follow the configured order, and embeddings/output head are resized together.
New rows use the mean of their old lexical subword pieces when available; otherwise the resize
initializer is retained. Always use the exported tokenizer with its fine-tuned checkpoint.

Adding tags alone does not teach extraction: train with matching prompt/target examples.
Define a fixed field order and a consistent representation of absent or repeated values.

## 4. Configure and run

Paths in the training JSON are relative to that JSON file. The supplied config points to
`data/train.jsonl` and `data/valid.jsonl`.

```bash
python train_pilot.py --config configs/train_custom.json
```

Defaults use one image per microbatch and four accumulation steps. This supports different
page sizes without spatial padding. Larger batches require equal image dimensions within each batch.
Accumulation is normalized by supervised token count, including the final partial window.

`precision="auto"` selects BF16 on supported CUDA GPUs, FP16 on other CUDA GPUs, and FP32 on CPU.
`gradient_checkpointing=true` enables decoder activation recomputation to reduce memory.
`noise_rate` corrupts lexical decoder inputs while protecting prompts/structural tokens;
labels stay clean. Start at zero and compare noise settings on validation.
The pretrained image mean/std are retained, on the 0–255 pixel scale.

To generate synthetic smoke data:

```bash
python examples/create_toy_dataset.py
python train_pilot.py --config configs/train_toy.json
```

This run requires the downloaded base weights. The tiny dataset checks that a job runs;
it is not an OCR quality benchmark.

## 5. Validation and recovery

Each epoch runs greedy generation on all validation examples. Validation does not run
teacher forcing, compute cross-entropy, or add noise to the prompt.
Progress bars cover dataset checks, training and validation.

| File in output_dir | Contents |
| --- | --- |
| `metrics.jsonl` | Training loss and generated validation metrics |
| `validation_predictions.json` | Latest targets, predictions, prompts and EOS status |
| `training_config.json` | Resolved settings |
| `environment.json` | Runtime version, device and precision |
| `last.pt` | Full recovery state after a completed epoch |
| `best/config.json` | Inference config for the selected model |
| `best/model.pt` | Weights without optimizer state |
| `best/tokenizer/` | Exact fine-tuned tokenizer |

`train_loss` is token-weighted cross-entropy over target tokens including EOS.
`sequence_exact_match` compares complete generated strings.
For OCR prompts, corpus CER/WER remove coordinate/separator tokens and compare the remaining text
without case or whitespace normalization. These values are fractions and can exceed 1.
For paired-tag extraction targets, `metric_fields` specifies the field names in target order.
`field_micro_f1` uses one-to-one multiset matching after NFKC, whitespace and case normalization;
repeated values count separately. `parse_valid_rate` checks the ordered grammar and
`document_exact_match` requires matching field multisets and a valid parse. Field F1 here evaluates
extracted values; it is not a span-level NER score or box-localization metric.

`selection_metric` defaults to `cer`. Choose `cer`/`wer` to minimize generated OCR error or
`field_micro_f1` to maximize generated extraction F1; `sequence_exact_match` is also maximized.
F1 requires a nonempty `metric_fields` list and correctly tagged targets.
CER/WER require OCR validation examples. The same matching policy is used for checkpoint selection
and final evaluation. Generation is capped by `max_length`; `eos_rate` exposes predictions that
reach the cap without EOS. Keep a representative validation set: generation costs more than a
teacher-forced pass but measures the behavior used at inference.

```bash
python train_pilot.py --config configs/train_custom.json --resume
```

Resume restores model, optimizer, scheduler, scaler and random-generator states.
It requires unchanged training settings, epoch budget, model config, tokenizer and dataset signature.
The signature includes manifests and image paths/sizes/modification times; keep image bytes unchanged.
An interrupted training or validation epoch is repeated from the last completed epoch.
Reproducibility also depends on the same runtime and hardware.
To change data or extend the epoch budget, start a new run from `best/config.json`.

## 6. Run the fine-tuned model

```bash
python run_pilot.py --config runs/custom/best/config.json --image data/images/valid_001.png --task ocr
python run_pilot.py --config runs/custom/best/config.json --image data/images/valid_001.png --prompt "<extract>"
```

Use the prompt trained for your model. Raw text and the JSON summary retain extraction tags.
The exported config lists built-in tasks present in the training manifest.
Measure performance on other prompts if you need to retain those capabilities.

## 7. Choose data, resolution and loss deliberately

All provided settings are examples, not tuned recommendations. A few hundred training examples
are generally insufficient for broad robustness and can cause memorization or unstable predictions.
Several thousand diverse examples are often a useful starting point, depending on the domain,
task and annotations; use learning curves and representative validation data to determine sufficiency.
Synthetic data can broaden coverage but should not replace held-out real-world evaluation.
Split originals before generating crops, augmented views or synthetic variants derived from them.

Resolution changes the evidence available to the model. Downscaling can erase small character
strokes and make recognition impossible; later upscaling does not recover the lost information.
Inspect prepared images at native size and tune resolution/crop height together with memory,
visual-sequence cost and target length. Keep transformations consistent at inference.
The notebook adapters expose `MAX_SIDE` and OCR `LINE_HEIGHT` and save preprocessing provenance.

This implementation uses simple target-token cross-entropy, not a category-balanced objective.
Lexical, coordinate, field-tag, separator and EOS tokens can have different frequencies and roles.
Per-category weighting or separately normalized terms can balance their contributions, but the
choice is empirical and may hurt as well as help. Compare with this baseline using generated
validation CER or field F1. Changes to an objective require a new run and documented settings.

See the [performance guide](performance.md) and the notebooks for CPU inference optimization
and a baseline comparison. Always measure latency and task quality on the intended hardware.
