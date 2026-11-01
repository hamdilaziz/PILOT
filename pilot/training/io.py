from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import torch


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def export_model(model, tokenizer, public_config, directory):
    """Write a self-contained inference bundle without optimizer state."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config = copy.deepcopy(public_config)
    config["checkpoint"] = "model.pt"
    config["tokenizer_path"] = "tokenizer"
    tokenizer.save_pretrained(directory / "tokenizer")
    atomic_checkpoint(directory / "model.pt", {key: value.detach().cpu() for key, value in model.state_dict().items()})
    save_json(directory / "config.json", config)


def atomic_checkpoint(path, state):
    temporary = Path(str(path) + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)
