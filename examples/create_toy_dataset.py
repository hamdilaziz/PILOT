"""Create synthetic images and JSONL records for a training smoke run."""
import json
from pathlib import Path

from PIL import Image, ImageDraw
from tqdm.auto import tqdm


def main():
    root = Path(__file__).resolve().parents[1] / "data" / "toy"
    (root / "images").mkdir(parents=True, exist_ok=True)
    texts = {
        "train": ["PILOT reads documents.", "Hello world.", "Invoice number 12.", "Total amount 25 euros."],
        "valid": ["Invoice number 31.", "Total amount 42 euros."],
    }
    for split, sentences in texts.items():
        records = []
        for i, text in enumerate(tqdm(sentences, desc=f"Create {split}")):
            filename = f"images/{split}_{i}.png"
            image = Image.new("RGB", (640, 128), "white")
            ImageDraw.Draw(image).text((24, 40), text, fill="black", font_size=24)
            image.save(root / filename)
            records.append({"image": filename, "prompt": "<ocr>", "target": text})
        (root / f"{split}.jsonl").write_text(
            "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n", encoding="utf-8"
        )
    print("Created data/toy. Run: python train_pilot.py --config configs/train_toy.json")


if __name__ == "__main__":
    main()
