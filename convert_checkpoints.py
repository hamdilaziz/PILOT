from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable

import torch


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Remove a leading 'module.' prefix coming from DataParallel / DDP checkpoints.
    """
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        cleaned[key] = value
    return cleaned


def normalize_encoder_state_dict(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Encoder weights were already close to the final structure.
    We only remove possible 'module.' prefixes.
    """
    return strip_module_prefix(state_dict)


def normalize_decoder_state_dict(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Convert legacy decoder weights to the final PILOTDecoder format.

    Legacy training code used a wrapper:
        GlobalBARTDecoder(
            decoder = BARTDecoder(...)
            features_updater = ...
        )

    In the final public code, the wrapper is removed, so:
        old: decoder.model....
        new: model....

    Also remove legacy parts that are no longer used:
        features_updater.*
    """
    state_dict = strip_module_prefix(state_dict)

    cleaned: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("features_updater."):
            continue

        if key.startswith("decoder."):
            key = key[len("decoder.") :]

        cleaned[key] = value

    return cleaned


def build_final_state_dict(training_checkpoint: Dict) -> Dict[str, torch.Tensor]:
    """
    Build a final state_dict compatible with:

        PILOTModel(
            encoder=...,
            decoder=...,
        )

    Supported inputs:
    1) Training checkpoint with:
         - encoder_state_dict
         - decoder_state_dict

    2) Already converted checkpoint with:
         - model_state_dict
         - or a flat state dict
    """
    if "encoder_state_dict" in training_checkpoint and "decoder_state_dict" in training_checkpoint:
        encoder_sd = normalize_encoder_state_dict(training_checkpoint["encoder_state_dict"])
        decoder_sd = normalize_decoder_state_dict(training_checkpoint["decoder_state_dict"])

        final_state_dict: Dict[str, torch.Tensor] = {}

        for key, value in encoder_sd.items():
            final_state_dict[f"encoder.{key}"] = value.detach().cpu()

        for key, value in decoder_sd.items():
            final_state_dict[f"decoder.{key}"] = value.detach().cpu()

        return final_state_dict

    if "model_state_dict" in training_checkpoint:
        state_dict = strip_module_prefix(training_checkpoint["model_state_dict"])
        return {k: v.detach().cpu() for k, v in state_dict.items()}

    # Maybe the file itself is already a plain state_dict
    if all(isinstance(v, torch.Tensor) for v in training_checkpoint.values()):
        state_dict = strip_module_prefix(training_checkpoint)
        return {k: v.detach().cpu() for k, v in state_dict.items()}

    raise ValueError(
        "Unsupported checkpoint format. Expected either:\n"
        "  - encoder_state_dict + decoder_state_dict\n"
        "  - model_state_dict\n"
        "  - or a plain state_dict"
    )


def make_tensors_serializable_for_safetensors(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Clone every tensor so shared storage does not trigger safetensors errors.
    """
    serializable = {}
    for key, value in state_dict.items():
        serializable[key] = value.detach().cpu().contiguous().clone()
    return serializable


def save_pt(state_dict: Dict[str, torch.Tensor], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, output_path)


def save_safetensors(state_dict: Dict[str, torch.Tensor], output_path: Path) -> None:
    from safetensors.torch import save_file

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(make_tensors_serializable_for_safetensors(state_dict), str(output_path))


def convert_one_checkpoint(
    input_path: Path,
    output_path: Path,
    output_format: str,
) -> None:
    checkpoint = torch.load(input_path, map_location="cpu")
    final_state_dict = build_final_state_dict(checkpoint)

    if output_format == "pt":
        save_pt(final_state_dict, output_path)
    elif output_format == "safetensors":
        save_safetensors(final_state_dict, output_path)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")

    print(f"[OK] Converted: {input_path} -> {output_path}")


def iter_checkpoint_files(input_paths: Iterable[Path]) -> Iterable[Path]:
    for path in input_paths:
        if path.is_file():
            yield path
        elif path.is_dir():
            for ext in ("*.pt", "*.pth", "*.bin"):
                yield from sorted(path.glob(ext))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert legacy PILOT training checkpoints to final public checkpoints."
    )

    parser.add_argument(
        "--input",
        type=str,
        nargs="+",
        required=True,
        help="Input checkpoint file(s) or directory(ies).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where converted checkpoints will be written.",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="pt",
        choices=["pt", "safetensors"],
        help="Output checkpoint format.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_public",
        help="Suffix added to the converted checkpoint filename stem.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_paths = [Path(p) for p in args.input]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = list(iter_checkpoint_files(input_paths))
    if not files:
        raise FileNotFoundError("No checkpoint files were found.")

    ext = ".safetensors" if args.format == "safetensors" else ".pt"

    for input_path in files:
        output_name = f"{input_path.stem}{args.suffix}{ext}"
        output_path = output_dir / output_name
        convert_one_checkpoint(
            input_path=input_path,
            output_path=output_path,
            output_format=args.format,
        )


if __name__ == "__main__":
    convert_one_checkpoint(
        input_path=Path("checkpoints/rimes.pt"),
        output_path=Path("checkpoints/pilot_rimes.pt"),
        output_format="pt",
    )