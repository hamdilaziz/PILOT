from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import requests
from requests.adapters import HTTPAdapter
from tqdm.auto import tqdm
from urllib3.util.retry import Retry


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def http_session():
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=1, status_forcelist=(429, 500, 502, 503, 504))
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def download_file(url, path, *, expected_size=None, md5=None):
    """Cache complete downloads; keep partial bytes for HTTP Range recovery."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    marker = path.with_suffix(path.suffix + ".download.json")
    if path.is_file() and marker.is_file():
        info = json.loads(marker.read_text())
        stat = path.stat()
        if (info.get("url"), info.get("size"), info.get("mtime_ns"), info.get("md5")) == (
            url, stat.st_size, stat.st_mtime_ns, md5
        ):
            return path
    if not path.is_file():
        partial = path.with_suffix(path.suffix + ".part")
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        with http_session().get(url, headers=headers, stream=True, timeout=(15, 120)) as response:
            response.raise_for_status()
            resumed = response.status_code == 206 and offset > 0
            if response.status_code == 206:
                content_range = response.headers.get("Content-Range", "")
                if not content_range.startswith(f"bytes {offset}-"):
                    raise RuntimeError(f"Unexpected partial response: {content_range}")
            initial = offset if resumed else 0
            total = expected_size or initial + int(response.headers.get("Content-Length", 0)) or None
            with partial.open("ab" if resumed else "wb") as handle, tqdm(
                total=total, initial=initial, unit="B", unit_scale=True, desc=f"Download {path.name}"
            ) as progress:
                for chunk in response.iter_content(1024 * 1024):
                    handle.write(chunk)
                    progress.update(len(chunk))
        if expected_size and partial.stat().st_size != expected_size:
            raise RuntimeError(f"Incomplete {path.name}; rerun to resume the download.")
        os.replace(partial, path)
    if expected_size and path.stat().st_size != expected_size:
        raise ValueError(f"{path} has an unexpected size. Remove it and download again.")
    if md5:
        digest = hashlib.md5()
        with path.open("rb") as handle, tqdm(total=path.stat().st_size, unit="B", unit_scale=True,
                                           desc=f"Verify {path.name}") as progress:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                progress.update(len(chunk))
        if digest.hexdigest() != md5:
            raise ValueError(f"Checksum mismatch for {path}. Remove it before downloading again.")
    stat = path.stat()
    save_json(marker, {"url": url, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "md5": md5})
    return path


def save_rgb(image, path, max_side=1280, min_size=64, target_height=None):
    """Prepare each image once; use the saved pixels for stats, training and prediction."""
    path = Path(path)
    if max_side < min_size:
        raise ValueError(f"max_side must be at least {min_size}.")
    if path.is_file():
        return
    image = image.convert("RGB")
    if target_height is not None:
        scale = min(target_height / image.height, max_side / image.width)
        image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                             Image.Resampling.LANCZOS)
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    if min(image.size) < min_size:
        canvas = Image.new("RGB", (max(min_size, image.width), max(min_size, image.height)), "white")
        canvas.paste(image, ((canvas.width - image.width) // 2, (canvas.height - image.height) // 2))
        image = canvas
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    image.save(temporary, format="PNG")
    os.replace(temporary, path)


def prepared_cache(directory, settings):
    directory = Path(directory)
    pending = directory / "preparation_settings.json"
    if pending.is_file() and json.loads(pending.read_text()) != settings:
        raise ValueError(f"Preparation settings changed. Choose another prepared_dir: {directory}")
    save_json(pending, settings)
    metadata = directory / "prepared.json"
    if metadata.is_file():
        info = json.loads(metadata.read_text())
        if info["settings"] != settings:
            raise ValueError(f"Preparation settings changed. Choose another prepared_dir instead of overwriting {directory}.")
        for split in ("train", "validation", "test"):
            for row in read_jsonl(directory / f"{split}.jsonl"):
                if not (directory / row["image"]).is_file():
                    raise FileNotFoundError(directory / row["image"])
        print(f"Reuse prepared data: {directory}")
        return info
    return None


def finish_preparation(directory, records, settings, details):
    directory = Path(directory)
    source_sets = {}
    for split, rows in records.items():
        if not rows:
            raise ValueError(f"No eligible {split} examples. Increase the source-image or document limit.")
        source_sets[split] = {row["source_id"] for row in rows}
    for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if source_sets[first] & source_sets[second]:
            raise ValueError(f"Source-document leakage between {first} and {second}.")
    total = np.zeros(3, dtype=np.float64)
    squares = np.zeros(3, dtype=np.float64)
    count = 0
    for row in tqdm(records["train"], desc="Training-image statistics"):
        with Image.open(directory / row["image"]) as image:
            pixels = np.asarray(image.convert("RGB"), dtype=np.float64).reshape(-1, 3)
        total += pixels.sum(axis=0)
        squares += np.square(pixels).sum(axis=0)
        count += len(pixels)
    mean = total / count
    std = np.sqrt(np.maximum(squares / count - mean**2, 0))
    if np.any(std <= 0):
        raise ValueError("Training images have zero channel variance; inspect the prepared data.")
    info = {
        "settings": settings, "counts": {split: len(rows) for split, rows in records.items()},
        "source_counts": {split: len(ids) for split, ids in source_sets.items()},
        "image_stats": {"mean": mean.tolist(), "std": std.tolist(), "pixel_count": count},
        **details,
    }
    for split, rows in records.items():
        destination = directory / f"{split}.jsonl"
        temporary = destination.with_suffix(".tmp")
        temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        os.replace(temporary, destination)
    save_json(directory / "prepared.json", info)
    return info


def audit_targets(directory, model_config, extra_tokens=()):
    """Choose a dataset sequence limit without changing pretrained positional embeddings."""
    from pilot.loading import read_json, resolve_repo_path
    from pilot.tokenization import load_tokenizer

    model_config = Path(model_config)
    config = read_json(model_config)
    tokenizer = load_tokenizer(resolve_repo_path(model_config, config["tokenizer_path"]))
    tokenizer.add_special_tokens({"additional_special_tokens": list(extra_tokens)},
                                 replace_additional_special_tokens=False)
    capacity = config["decoder"]["max_length"]
    summary = []
    for split in ("train", "validation"):
        lengths = []
        rows = read_jsonl(Path(directory) / f"{split}.jsonl")
        for row in tqdm(rows, desc=f"Audit {split} targets"):
            prompt = tokenizer.encode(row["prompt"], add_special_tokens=False)
            target = tokenizer.encode(row["target"], add_special_tokens=False)
            if tokenizer.unk_token_id is not None and tokenizer.unk_token_id in prompt + target:
                raise ValueError(f"Unknown token in {split} example {row['source_id']}; inspect text/tokenizer.")
            lengths.append(len(prompt) + len(target) + 2)
        summary.append({"split": split, "examples": len(lengths), "max_tokens": max(lengths),
                        "median_tokens": float(np.median(lengths))})
    longest = max(row["max_tokens"] for row in summary)
    if longest > capacity:
        raise ValueError(f"A target needs {longest} tokens, beyond decoder capacity {capacity}. Reduce the task output.")
    recommended = min(capacity, int(np.ceil((longest + 16) / 32) * 32))
    return summary, recommended


def experiment_model_config(model_config, destination, image_stats):
    """Keep checkpoint architecture and resolve artifact paths before copying the config."""
    from pilot.loading import read_json, resolve_repo_path

    model_config = Path(model_config)
    config = read_json(model_config)
    checkpoint = resolve_repo_path(model_config, config["checkpoint"])
    if checkpoint.suffix != ".pt":
        raise ValueError("Set checkpoint to the released .pt weights in the model config.")
    tokenizer = resolve_repo_path(model_config, config["tokenizer_path"])
    if not checkpoint.is_file() or not tokenizer.is_dir():
        raise FileNotFoundError("Download the released PILOT weights/tokenizer first; see the repository README.")
    config["checkpoint"], config["tokenizer_path"] = str(checkpoint), str(tokenizer)
    config["preprocessing"].update(mean=image_stats["mean"], std=image_stats["std"])
    save_json(destination, config)
    return Path(destination)
