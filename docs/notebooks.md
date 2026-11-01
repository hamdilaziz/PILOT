# Fine-tuning notebooks

The examples use the public `pilot.training` package. Prepared data, settings, metrics and
checkpoints are stored as local files.

## Setup

From the repository root, in the environment used by your Jupyter kernel:

~~~bash
pip install -e ".[notebooks]"
jupyter lab
~~~

An equivalent uv setup is:

~~~bash
uv venv
uv pip install -e ".[notebooks]"
uv run jupyter lab
~~~

Install an appropriate CUDA-enabled PyTorch build before the editable installation if training on GPU.
Download the released generic checkpoint/tokenizer as described in the main README.
The copied experiment config uses the configured `.pt` checkpoint and retains the pretrained
decoder positional capacity. The best weights are exported as `best/model.pt`.

## Organization

There are two notebooks rather than four separate preparation/training notebooks:

1. [FATURA field extraction](../examples/finetune_fatura.ipynb).
2. [HierText line OCR](../examples/finetune_hiertext_ocr.ipynb).

Each has setup, preparation, audit/training, and prediction/evaluation sections.
The preparation writes reusable PNG images, JSONL manifests, provenance and training-only RGB
statistics. Rerunning the notebook reuses them; after a kernel restart, the training section can
load `prepared.json` directly. Changing preparation parameters requires a new prepared directory.
The shared source adapters are in `examples/finetuning/` so the notebooks remain readable.

Training restores pretrained weights before
adding tokens, preserve existing IDs, use paired tags for extraction, mask prompt/padding targets,
keep labels clean when corrupting decoder inputs, audit lengths, and accumulate gradients.
`noise_strategy="sequence"` samples lexical replacements from the same sequence.
Configuration, epoch history and recovery state are local files.

## FATURA

Source: [FATURA2 on Zenodo](https://zenodo.org/records/10371464), CC BY 4.0.
The ~691 MB official ZIP is downloaded once and checked against its published MD5.
Preparation reads selected members directly rather than extracting all 50,000 archive members.

The default tutorial subset has 400 train, 80 validation and 80 test documents.
Sampling covers the templates in each official strategy-1 CSV split.
Only white-background images are used; the colored duplicate of an invoice is not added as
another example. Template layouts can occur in multiple splits under this protocol.
It evaluates new content on familiar layouts, not unseen-layout generalization.

Targets use six keys from `Original_Format`: NUMBER, DATE, DUE_DATE, SUB_TOTAL, TAX and TOTAL.
They preserve complete annotated region text, including labels/currencies; absent keys produce
empty paired tags. They are not automatically converted into clean business values.
The source is a synthetic dataset with annotation imperfections, so inspect image/target samples.

The best checkpoint maximizes generated validation field micro F1, using the same matching policy
as final evaluation. No validation loss is computed. Final evaluation also shows
per-field precision/recall/F1, valid-parse rate and full-document exact match.
Repeated values use one-to-one multiset matching. Amount/date punctuation is retained.

## HierText

Source: [official HierText repository](https://github.com/google-research-datasets/hiertext),
CC BY-SA 4.0. Annotation URLs are pinned to revision
`70b6620b2b112597d8219e11eee9773a1403827c`.
Compressed ground truths total about 83 MB.

For the default small example, the helper streams the beginning of each official image archive
and stops after caching 50 train, 10 validation and 10 test source images.
This avoids downloading the full ~2.8 GB train image archive.
Selected source IDs are saved. This is archive-order sampling, not uniform corpus sampling.
Set a source-image limit to `None` to process a full split.

Each source contributes up to eight legible nonvertical line crops of 2–160 characters.
Actual counts depend on eligible lines in the selected images.
All crops retain their original official split and source image ID.
Images are checked against annotation dimensions before cropping.
Line images are RGB and scaled toward a height of 64 pixels while respecting the maximum side length,
then padded to at least 64 pixels per dimension. New-line inference uses the same transformation.

The experiment learns `<ocr>` without adding field tags.
The best checkpoint minimizes generated validation CER, and final evaluation reports CER/WER,
sequence exact match and EOS termination.
These are recognition scores on ground-truth crops, not official HierText end-to-end scores.
The source does not guarantee line order within a paragraph, so full-page targets need a
separate documented ordering policy.

## Experimental settings and CPU inference

Neither notebook supplies tuned hyperparameters or a quality benchmark. Defaults use small subsets
so the complete workflow can be tried. A few hundred examples generally provide limited coverage
and can lead to memorization and unstable generation. Several thousand diverse examples are often
a practical starting point; dataset quality, domain and task difficulty determine actual requirements.
Measure learning curves rather than treating any sample count as sufficient. Synthetic data can
help cover rare text or layouts; retain a representative real held-out evaluation set.

Image resolution, crop height, sequence length, learning rate, batch size and training duration
need validation experiments. Inspect prepared small text: downscaling may remove distinguishing
strokes, and upscaling cannot restore them. `MAX_SIDE` and `LINE_HEIGHT` expose the main resize
settings. New settings require new preparation caches and the same inference preprocessing.

The training objective is ordinary cross-entropy with masked prompts/padding. Structural and lexical
tokens contribute to the same average, so frequent categories can dominate. Category weights or
separate normalized loss terms are possible extensions; choose them empirically with generated
validation metrics. A more complex objective is not automatically better.

Both notebooks finish with a CPU benchmark of channels-last layout and a bounded positional-encoding
cache, on top of GroupNorm and separate attention caches. Optional encoder compilation is disabled
by default. The same inputs, decoding settings and thread count are used for both variants; warm-up
and image I/O are excluded. Token equality is checked, and latencies are saved to `cpu_benchmark.json`.
A single example is a quick check, not a production benchmark. A gain around 3× is a possible outcome
of combined optimizations on suitable workloads, not a measured or guaranteed result here.
See [performance and compatibility](performance.md) before choosing deployment settings.

## Validation performed

Source-adapter checks use actual FATURA and HierText samples.
The regression suite executes all Python code cells of both notebooks in fresh subprocesses,
using synthetic source-format fixtures and a reduced real mBART on CPU. It exercises training,
export/reload, evaluation, tables and plotting code without downloading data during CI.
Notebook schemas and code-cell syntax are checked separately.

The tests do not establish full-checkpoint OCR/extraction quality or CUDA behavior.
Run the notebooks with the released PILOT weights on your GPU to validate those.
