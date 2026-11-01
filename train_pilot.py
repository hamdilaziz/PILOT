"""Fine-tune a released PILOT checkpoint: python train_pilot.py --config configs/train_custom.json."""
import argparse

from pilot.training.trainer import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Training JSON config; paths are relative to this file.")
    parser.add_argument("--resume", action="store_true", help="Resume output_dir/last.pt after a completed epoch.")
    args = parser.parse_args()
    train(args.config, resume=args.resume)


if __name__ == "__main__":
    main()
