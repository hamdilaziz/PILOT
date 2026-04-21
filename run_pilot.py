from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from safetensors.torch import load_model as load_safetensors_model
from transformers import XLMRobertaTokenizerFast

from pilot import PILOTModel


ALL_TASKS = ["ocr", "ocr_with_boxes", "find_it", "ocr_on_box"]


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

    tokenizer = XLMRobertaTokenizerFast.from_pretrained(str(tokenizer_path))
    runtime_config = normalize_runtime_config(public_config, tokenizer)

    model = PILOTModel(runtime_config)

    if checkpoint_path.suffix == ".safetensors":
        load_safetensors_model(model, str(checkpoint_path))
    else:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict, strict=True)

    model.to(device)
    model.eval()

    return model, tokenizer, public_config


def prepare_image(
    image: Image.Image,
    mean: List[float],
    std: List[float],
) -> torch.Tensor:
    image_np = np.array(image.convert("RGB"), dtype=np.float32)
    mean_np = np.array(mean, dtype=np.float32)
    std_np = np.array(std, dtype=np.float32)

    image_np = (image_np - mean_np) / std_np
    return torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)


def build_task_prompt(
    task: str,
    query_text: Optional[str] = None,
    box: Optional[Tuple[int, int, int, int]] = None,
    coord_bin_size: int = 10,
) -> str:
    if task == "ocr":
        return "<ocr>"

    if task == "ocr_with_boxes":
        return "<ocr_with_boxes>"

    if task == "find_it":
        if not query_text:
            raise ValueError("`--query` must be provided for task 'find_it'.")
        return f"<find_it> {query_text}"

    if task == "ocr_on_box":
        if box is None:
            raise ValueError(
                "`--box x1 y1 x2 y2` must be provided for task 'ocr_on_box'."
            )
        x1, y1, x2, y2 = box
        x1 //= coord_bin_size
        y1 //= coord_bin_size
        x2 //= coord_bin_size
        y2 //= coord_bin_size
        return f"<ocr_on_box> <x_{x1}><y_{y1}><x_{x2}><y_{y2}>"

    raise ValueError(f"Unknown task: {task}")


def prepare_batch_for_inference(
    image: Image.Image,
    tokenizer: XLMRobertaTokenizerFast,
    prompt: str,
    mean: List[float],
    std: List[float],
) -> Dict[str, Any]:
    img_tensor = prepare_image(image=image, mean=mean, std=std)
    token_prompt = torch.tensor(
        [tokenizer.encode(prompt, add_special_tokens=False)],
        dtype=torch.long,
    )

    return {
        "imgs": img_tensor,
        "token_prompt": token_prompt,
    }


def clean_prediction_text(prediction: str) -> str:
    text = re.sub(r"<x_\d+>|<y_\d+>", "", prediction)
    text = text.replace("<sep/>", "\n")
    text = re.sub(r"</?s>|<ocr_with_boxes>|<ocr>|<find_it>|<ocr_on_box>", "", text)

    lines = [line.strip() for line in text.split("\n")]
    lines = [line for line in lines if line]

    return "\n".join(lines)


def extract_boxes_from_prediction(
    prediction: str,
    coord_bin_size: int = 10,
) -> np.ndarray:
    matches = re.findall(r"<x_(\d+)>|<y_(\d+)>", prediction)
    coords = [int("".join(match)) for match in matches]

    if len(coords) < 4:
        return np.zeros((0, 4), dtype=np.int32)

    usable = (len(coords) // 4) * 4
    coords = coords[:usable]

    boxes = np.array(coords, dtype=np.int32).reshape(-1, 4)
    boxes = boxes * coord_bin_size
    # filter out the malformed boxes where x2 <= x1 or y2 <= y1
    boxes = boxes[(boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])]
    return boxes


def draw_boxes(
    image: Image.Image,
    boxes: np.ndarray,
    color: str = "red",
    thickness: int = 2,
    show_id: bool = False,
) -> Image.Image:
    image = image.convert("RGB").copy()
    draw = ImageDraw.Draw(image)

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(v) for v in box]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)
        if show_id:
            draw.text((x1, max(0, y1 - 12)), str(i), fill=color)

    return image


def draw_single_box(
    image: Image.Image,
    box: Tuple[int, int, int, int],
    color: str = "blue",
    thickness: int = 2,
) -> Image.Image:
    image = image.convert("RGB").copy()
    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = [int(v) for v in box]
    draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)
    return image


def parse_box(box_values: Optional[List[int]]) -> Optional[Tuple[int, int, int, int]]:
    if box_values is None:
        return None
    if len(box_values) != 4:
        raise ValueError("`--box` must contain exactly 4 integers: x1 y1 x2 y2")
    return tuple(int(v) for v in box_values)


def validate_task_support(config: Dict[str, Any], task: str) -> None:
    supported_tasks = config.get("supported_tasks", ALL_TASKS)
    if task not in supported_tasks:
        raise ValueError(
            f"Task '{task}' is not supported by model '{config.get('name', 'unknown')}'. "
            f"Supported tasks: {supported_tasks}"
        )


def run_task(
    model: PILOTModel,
    tokenizer: XLMRobertaTokenizerFast,
    model_config: Dict[str, Any],
    image_path: str | Path,
    task: str,
    output_dir: str | Path,
    device: torch.device,
    query_text: Optional[str] = None,
    box: Optional[Tuple[int, int, int, int]] = None,
    coord_bin_size: Optional[int] = None,
    use_amp: bool = True,
    num_beams: Optional[int] = None,
    max_length: Optional[int] = None,
) -> Dict[str, Any]:
    validate_task_support(model_config, task)

    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    preprocessing_cfg = model_config.get("preprocessing", {})
    generation_cfg = model_config.get("decoder", {})

    mean = preprocessing_cfg.get("mean", [123.675, 116.28, 103.53])
    std = preprocessing_cfg.get("std", [58.395, 57.12, 57.375])

    if coord_bin_size is None:
        coord_bin_size = int(preprocessing_cfg.get("coord_bin_size", 10))

    if max_length is None:
        max_length = int(generation_cfg.get("max_length", 1024))

    if num_beams is None:
        num_beams = 1

    image = Image.open(image_path)

    prompt = build_task_prompt(
        task=task,
        query_text=query_text,
        box=box,
        coord_bin_size=coord_bin_size,
    )

    batch = prepare_batch_for_inference(
        image=image,
        tokenizer=tokenizer,
        prompt=prompt,
        mean=mean,
        std=std,
    )
    batch["imgs"] = batch["imgs"].to(device)
    batch["token_prompt"] = batch["token_prompt"].to(device)

    results = model.predict(
        batch,
        use_amp=use_amp and device.type == "cuda",
        num_beams=num_beams,
        max_length=max_length,
    )

    raw_prediction = results["str_pred"][0]
    cleaned_text = clean_prediction_text(raw_prediction)
    predicted_boxes = extract_boxes_from_prediction(
        raw_prediction,
        coord_bin_size=coord_bin_size,
    )

    stem = image_path.stem
    prefix = f"{stem}_{task}"

    save_text(output_dir / f"{prefix}_raw.txt", raw_prediction)

    summary: Dict[str, Any] = {
        "model_name": model_config.get("name"),
        "image": str(image_path),
        "task": task,
        "prompt": prompt,
        "prediction_time": results["time"],
        "raw_prediction": raw_prediction,
        "cleaned_text": cleaned_text,
        "num_boxes": int(len(predicted_boxes)),
        "boxes": predicted_boxes.tolist(),
        "coord_bin_size": coord_bin_size,
        "max_length": max_length,
        "mean": mean,
        "std": std,
    }

    if task in {"ocr", "ocr_on_box", "ocr_with_boxes"}:
        save_text(output_dir / f"{prefix}.txt", cleaned_text)

    if task in {"ocr_with_boxes", "find_it"} and len(predicted_boxes) > 0:
        vis = draw_boxes(image, predicted_boxes, color="red", thickness=2, show_id=False)
        vis.save(output_dir / f"{prefix}_boxes.png")

    if task == "ocr_on_box" and box is not None:
        vis = draw_single_box(image, box, color="blue", thickness=2)
        vis.save(output_dir / f"{prefix}_input_box.png")

    save_json(output_dir / f"{prefix}.json", summary)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PILOT inference.")

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the input image.",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=ALL_TASKS,
        help="Task prompt to use.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Query text for the 'find_it' task.",
    )
    parser.add_argument(
        "--box",
        type=int,
        nargs=4,
        default=None,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Input box in pixel coordinates for the 'ocr_on_box' task.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the model config JSON file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional override for the checkpoint path.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Optional override for the tokenizer directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory where outputs will be saved.",
    )
    parser.add_argument(
        "--coord-bin-size",
        type=int,
        default=None,
        help="Optional override for coordinate bin size.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Optional override for generation max length.",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=None,
        help="Optional override for beam size.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    model, tokenizer, model_config = load_pilot_model(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        device=device,
    )

    summary = run_task(
        model=model,
        tokenizer=tokenizer,
        model_config=model_config,
        image_path=args.image,
        task=args.task,
        output_dir=args.output_dir,
        device=device,
        query_text=args.query,
        box=parse_box(args.box),
        coord_bin_size=args.coord_bin_size,
        use_amp=True,
        num_beams=args.num_beams,
        max_length=args.max_length,
    )

    print(f"Model: {summary['model_name']}")
    print(f"Task: {summary['task']}")
    print(f"Prompt: {summary['prompt']}")
    print(f"Prediction time: {summary['prediction_time']:.4f}s")
    print("Raw prediction:")
    print(summary["raw_prediction"])
    print()

    if summary["cleaned_text"]:
        print("Cleaned text:")
        print(summary["cleaned_text"])
        print()

    if summary["boxes"]:
        print("Predicted boxes:")
        for i, box in enumerate(summary["boxes"]):
            print(f"  {i}: {box}")


if __name__ == "__main__":
    main()