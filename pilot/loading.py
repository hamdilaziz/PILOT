from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from transformers import XLMRobertaTokenizerFast

from .modeling import PILOTModel
from .tokenization import load_tokenizer


def read_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_text(path: Path, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def save_json(path: Path, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def resolve_repo_path(config_path: Path, value: str | Path) -> Path:
    """
    Resolve a path coming from the config or CLI.

    Resolution order:
    1. absolute path
    2. current working directory
    3. repository root inferred from config path
    4. config parent directory
    """
    value = Path(value)

    if value.is_absolute():
        return value

    if value.exists():
        return value.resolve()

    config_path = config_path.resolve()
    config_dir = config_path.parent
    repo_root = config_dir.parent if config_dir.name == "configs" else config_dir

    candidate_repo = repo_root / value
    if candidate_repo.exists():
        return candidate_repo.resolve()

    candidate_cfg = config_dir / value
    if candidate_cfg.exists():
        return candidate_cfg.resolve()

    return candidate_repo.resolve()


def normalize_runtime_config(
    raw_config: Dict[str, Any],
    tokenizer: XLMRobertaTokenizerFast,
) -> Dict[str, Any]:
    """
    Adapt user-facing JSON config to the config expected by PILOTModel.
    """
    config = copy.deepcopy(raw_config)

    config.setdefault("use_2d_positional_encoding", True)

    encoder_cfg = config.setdefault("encoder", {})
    encoder_cfg.setdefault("input_channels", 3)

    decoder_cfg = config.setdefault("decoder", {})
    decoder_cfg["tokenizer"] = tokenizer

    if "num_layers" in decoder_cfg and "bart_layers" not in decoder_cfg:
        decoder_cfg["bart_layers"] = decoder_cfg["num_layers"]

    if "max_length" in decoder_cfg and "max_position_embeddings" not in decoder_cfg:
        decoder_cfg["max_position_embeddings"] = decoder_cfg["max_length"]

    return config


def load_pilot_model(
    config_path: str | Path,
    device: torch.device,
    checkpoint_path: Optional[str | Path] = None,
    tokenizer_path: Optional[str | Path] = None,
) -> tuple[PILOTModel, XLMRobertaTokenizerFast, Dict[str, Any]]:
    config_path = Path(config_path)
    public_config = read_json(config_path)

    if tokenizer_path is None:
        tokenizer_path = public_config.get("tokenizer_path")
    if tokenizer_path is None:
        raise ValueError(
            "No tokenizer path provided. Add 'tokenizer_path' in the config "
            "or pass --tokenizer."
        )

    if checkpoint_path is None:
        checkpoint_path = public_config.get("checkpoint")
    if checkpoint_path is None:
        raise ValueError(
            "No checkpoint path provided. Add 'checkpoint' in the config "
            "or pass --checkpoint."
        )

    tokenizer_path = resolve_repo_path(config_path, tokenizer_path)
    checkpoint_path = resolve_repo_path(config_path, checkpoint_path)
    if checkpoint_path.suffix != ".pt":
        raise ValueError("PILOT checkpoints must use the .pt format.")

    tokenizer = load_tokenizer(tokenizer_path)
    runtime_config = normalize_runtime_config(public_config, tokenizer)

    model = PILOTModel(runtime_config)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=True)

    model.to(device)
    model.eval()

    return model, tokenizer, public_config
