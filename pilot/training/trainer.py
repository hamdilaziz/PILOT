from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import re

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import IGNORE_INDEX, SequenceCollator, SequenceDataset, corrupt_inputs
from .io import atomic_checkpoint, export_model, save_json
from .metrics import field_metrics, is_ocr_prompt, parse_fields, sequence_metrics


@dataclass
class TrainingConfig:
    model_config: str
    train_manifest: str
    validation_manifest: str
    output_dir: str
    checkpoint: str | None = None
    tokenizer: str | None = None
    extra_tokens: tuple[str, ...] = ()
    epochs: int = 10
    learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    batch_size: int = 1
    accumulation_steps: int = 4
    max_length: int = 512
    num_workers: int = 0
    precision: str = "auto"
    device: str = "auto"
    warmup_ratio: float = 0.05
    grad_clip_norm: float = 1.0
    noise_rate: float = 0.0
    noise_strategy: str = "random_vocab"
    gradient_checkpointing: bool = False
    selection_metric: str = "cer"
    metric_fields: tuple[str, ...] = ()
    seed: int = 42

    @classmethod
    def from_file(cls, filename):
        filename = Path(filename).resolve()
        values = json.loads(filename.read_text(encoding="utf-8"))
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown training settings: {sorted(unknown)}")
        for key in ("model_config", "train_manifest", "validation_manifest", "output_dir", "checkpoint", "tokenizer"):
            if values.get(key) is not None:
                values[key] = str((filename.parent / values[key]).resolve())
        values["extra_tokens"] = tuple(values.get("extra_tokens", ()))
        values["metric_fields"] = tuple(values.get("metric_fields", ()))
        config = cls(**values)
        for name in ("epochs", "batch_size", "accumulation_steps", "max_length"):
            if getattr(config, name) < 1:
                raise ValueError(f"{name} must be positive.")
        if config.num_workers < 0 or config.learning_rate <= 0 or config.weight_decay < 0 or config.grad_clip_norm <= 0:
            raise ValueError("Invalid optimizer or worker settings.")
        if not 0 <= config.noise_rate <= 1 or not 0 <= config.warmup_ratio <= 1:
            raise ValueError("noise_rate and warmup_ratio must be in [0, 1].")
        if config.precision not in {"auto", "fp32", "bf16", "fp16"}:
            raise ValueError("precision must be auto, fp32, bf16 or fp16.")
        if config.selection_metric not in {"cer", "wer", "field_micro_f1", "sequence_exact_match"}:
            raise ValueError("selection_metric must be cer, wer, field_micro_f1 or sequence_exact_match.")
        if len(set(config.metric_fields)) != len(config.metric_fields) or any(
            not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name)
            for name in config.metric_fields
        ):
            raise ValueError("metric_fields must contain unique field names matching the paired target tags.")
        if config.selection_metric == "field_micro_f1" and not config.metric_fields:
            raise ValueError("field_micro_f1 requires an ordered list of metric_fields.")
        if config.noise_strategy not in {"sequence", "random_vocab"}:
            raise ValueError("noise_strategy must be sequence or random_vocab.")
        if any(not isinstance(token, str) or not token for token in config.extra_tokens):
            raise ValueError("extra_tokens must contain non-empty strings.")
        return config


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def precision_dtype(config, device):
    precision = config.precision
    if precision == "auto":
        precision = "bf16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else (
            "fp16" if device.type == "cuda" else "fp32"
        )
    if precision != "fp32" and device.type != "cuda":
        raise ValueError("Mixed precision in this trainer requires CUDA. Use precision=fp32 on CPU.")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This GPU does not support BF16; select fp16 or fp32.")
    return {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[precision]


def data_signature(*datasets):
    digest = hashlib.sha256()
    for dataset in datasets:
        digest.update(dataset.manifest.read_bytes())
        for path in sorted(dataset.image_paths):
            stat = path.stat()
            digest.update(f"{path}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()


def summed_loss(logits, labels):
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1),
                           ignore_index=IGNORE_INDEX, reduction="sum")


@torch.inference_mode()
def validate(model, tokenizer, loader, device, dtype, max_length=None, metric_fields=()):
    model.eval()
    predictions = []
    for batch in tqdm(loader, desc="Validate"):
        images = batch["images"].to(device, non_blocking=True)
        inputs = batch["input_ids"].to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
            features = model._encode_for_decoder(images)
            for i, sample in enumerate(batch["samples"]):
                prefix_length = sample["prompt_length"] + 1
                prefix = inputs[i:i + 1, :prefix_length]
                capacity = min(max_length or model.decoder.model.config.max_position_embeddings,
                               model.decoder.model.config.max_position_embeddings) - prefix_length
                if capacity < 1:
                    raise ValueError("The generation limit leaves no room after the prompt.")
                generated = model.decoder.generate(
                    input_ids=prefix, encoder_hidden_states=features[i:i + 1], max_new_tokens=capacity,
                    num_beams=1, do_sample=False, use_cache=True, repetition_penalty=1.0,
                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                )[0, prefix_length:].tolist()
                terminated = tokenizer.eos_token_id in generated
                if terminated:
                    generated = generated[:generated.index(tokenizer.eos_token_id)]
                text = tokenizer.decode(generated, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                predictions.append({
                    "image": str(sample["image"]), "prompt": sample["prompt"], "target": sample["target"],
                    "prediction": text, "terminated": terminated,
                })
    metrics = sequence_metrics(predictions)
    if metric_fields:
        summary, _ = field_metrics(predictions, metric_fields)
        metrics.update(summary)
    metrics["eos_rate"] = sum(row["terminated"] for row in predictions) / len(predictions)
    return metrics, predictions


def train_epoch(model, tokenizer, loader, optimizer, scheduler, scaler, config, device, dtype):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = total_tokens = window_tokens = 0
    progress = tqdm(loader, desc="Train")
    for step, batch in enumerate(progress, 1):
        images = batch["images"].to(device, non_blocking=True)
        inputs = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        attention = batch["attention_mask"].to(device, non_blocking=True)
        inputs = corrupt_inputs(inputs, attention, batch["samples"], tokenizer, config.noise_rate,
                                getattr(config, "noise_strategy", "random_vocab"))
        with torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
            logits = model(images, input_ids=inputs, attention_mask=attention, use_cache=False).logits
            loss = summed_loss(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss. Check images, learning rate and precision.")
        tokens = labels.ne(IGNORE_INDEX).sum().item()
        scaler.scale(loss).backward()
        total_loss += loss.item()
        total_tokens += tokens
        window_tokens += tokens
        if step % config.accumulation_steps == 0 or step == len(loader):
            scaler.unscale_(optimizer)
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(window_tokens)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm,
                                           error_if_nonfinite=not scaler.is_enabled())
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= previous_scale:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            window_tokens = 0
        progress.set_postfix(loss=f"{total_loss / total_tokens:.4f}")
    return total_loss / total_tokens


def rng_state(generator):
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
        "torch": torch.get_rng_state(), "loader": generator.get_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state, generator):
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.array(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    torch.set_rng_state(state["torch"])
    generator.set_state(state["loader"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def train(config_path, resume=False):
    # Imported lazily so the trainer remains usable as a Python API after installation.
    from pilot.loading import load_pilot_model

    config = TrainingConfig.from_file(config_path)
    output = Path(config.output_dir)
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError(f"{output} is not empty. Use --resume or a new output_dir.")
    state = torch.load(output / "last.pt", map_location="cpu", weights_only=True) if resume else None
    settings = asdict(config)
    saved_settings = dict(state["settings"]) if state else {}
    if state:
        saved_settings.setdefault("noise_strategy", "random_vocab")
        saved_settings.setdefault("metric_fields", ())
    if state and saved_settings != settings:
        raise ValueError("Resume requires the saved training settings, including the epoch budget.")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if config.device == "auto" else (
        torch.device(config.device)
    )
    dtype = precision_dtype(config, device)
    # Restore pretrained shapes first; extend the tokenizer only after strict loading.
    model, tokenizer, public_config = load_pilot_model(config.model_config, torch.device("cpu"),
                                                      config.checkpoint, config.tokenizer)
    added = model.decoder.add_special_tokens(config.extra_tokens)
    model.to(device)
    capacity = model.decoder.model.config.max_position_embeddings
    if config.max_length > capacity:
        raise ValueError(f"max_length={config.max_length} exceeds the pretrained decoder capacity {capacity}.")
    if config.gradient_checkpointing:
        model.decoder.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    preprocessing = public_config["preprocessing"]
    train_data = SequenceDataset(config.train_manifest, tokenizer, preprocessing["mean"], preprocessing["std"],
                                 config.max_length)
    valid_data = SequenceDataset(config.validation_manifest, tokenizer, preprocessing["mean"], preprocessing["std"],
                                 config.max_length)
    if config.selection_metric in {"cer", "wer"} and not any(
        is_ocr_prompt(sample["prompt"]) for sample in valid_data.samples
    ):
        raise ValueError("CER/WER selection requires OCR validation examples. For extraction, use field_micro_f1.")
    if config.metric_fields:
        for sample in (*train_data.samples, *valid_data.samples):
            if not parse_fields(sample["target"], config.metric_fields)[1]:
                raise ValueError("Every extraction target must match the ordered metric_fields tag schema.")
    overlap = train_data.image_paths & valid_data.image_paths
    if overlap:
        raise ValueError(f"Training and validation share {len(overlap)} image paths; split by document.")
    signature = data_signature(train_data, valid_data)
    model_signature = hashlib.sha256(Path(config.model_config).read_bytes()).hexdigest()
    tokenizer_signature = hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest()
    if state and (state["data_signature"] != signature or state["vocabulary"] != tokenizer.get_vocab() or
                  state["model_signature"] != model_signature or state["tokenizer_signature"] != tokenizer_signature):
        raise ValueError("The dataset or tokenizer vocabulary changed since this checkpoint.")
    public_config["name"] = public_config.get("name", "pilot") + "_finetuned"
    public_config["training_prompts"] = sorted({sample["prompt"] for sample in train_data.samples})
    task_names = {"<ocr>": "ocr", "<ocr_with_boxes>": "ocr_with_boxes", "<find_it>": "find_it",
                  "<ocr_on_box>": "ocr_on_box"}
    public_config["supported_tasks"] = sorted({task_names[sample["prompt"].split()[0]]
                                               for sample in train_data.samples
                                               if sample["prompt"].split()[0] in task_names})
    generator = torch.Generator().manual_seed(config.seed)
    collator = SequenceCollator(tokenizer.pad_token_id)
    common = {"batch_size": config.batch_size, "num_workers": config.num_workers, "collate_fn": collator,
              "pin_memory": device.type == "cuda", "worker_init_fn": seed_worker}
    train_loader = DataLoader(train_data, shuffle=True, generator=generator, **common)
    valid_loader = DataLoader(valid_data, shuffle=False, generator=torch.Generator().manual_seed(config.seed), **common)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    steps = math.ceil(len(train_loader) / config.accumulation_steps) * config.epochs
    warmup = int(steps * config.warmup_ratio)

    def learning_rate(step):
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * min(1, (step - warmup) / max(steps - warmup, 1))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    start_epoch, best = 0, None
    if state:
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, best = state["epoch"], state["best"]
        restore_rng(state["rng"], generator)
    output.mkdir(parents=True, exist_ok=True)
    history = state["history"] if state else []
    history_path = output / "metrics.jsonl"
    history_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in history), encoding="utf-8")
    save_json(output / "training_config.json", settings)
    save_json(output / "environment.json", {
        "python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
        "numpy": np.__version__, "cuda": torch.version.cuda, "device": str(device),
        "precision": str(dtype), "new_tokens": added,
    })
    for epoch in range(start_epoch, config.epochs):
        print(f"Epoch {epoch + 1}/{config.epochs}")
        training_loss = train_epoch(model, tokenizer, train_loader, optimizer, scheduler, scaler,
                                    config, device, dtype)
        validation, predictions = validate(model, tokenizer, valid_loader, device, dtype,
                                           config.max_length, config.metric_fields)
        if config.selection_metric not in validation:
            raise ValueError(f"No validation metric named {config.selection_metric} was produced.")
        score = validation[config.selection_metric]
        maximize = config.selection_metric in {"sequence_exact_match", "field_micro_f1"}
        improved = best is None or (score > best if maximize else score < best)
        if improved:
            best = score
            export_model(model, tokenizer, public_config, output / "best")
        save_json(output / "validation_predictions.json", predictions)
        metrics = {"epoch": epoch + 1, "train_loss": training_loss, "validation": validation, "best": best}
        history.append(metrics)
        atomic_checkpoint(output / "last.pt", {
            "epoch": epoch + 1, "best": best, "settings": settings, "data_signature": signature,
            "vocabulary": tokenizer.get_vocab(), "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "model_signature": model_signature, "tokenizer_signature": tokenizer_signature,
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "rng": rng_state(generator),
            "history": history,
        })
        history_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in history), encoding="utf-8")
        print(json.dumps(metrics, ensure_ascii=False))
    return {"output_dir": str(output), "best": best}
