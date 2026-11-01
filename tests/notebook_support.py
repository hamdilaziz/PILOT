"""Small offline datasets and a real, reduced mBART used only by notebook smoke tests."""
from pathlib import Path
import csv
import gzip
import io
import json
import zipfile

import numpy as np
from PIL import Image


def fixture_image():
    pixels = np.arange(96, dtype=np.uint8)[None, :, None]
    return Image.fromarray(np.broadcast_to(pixels, (72, 96, 3)).copy())


def fatura_fixture(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / "fixture.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for split, instance in (("train", 0), ("dev", 150), ("test", 175)):
            rows = []
            for template in (1, 2):
                stem = f"Template{template}_Instance{instance}"
                rows.append({"img_path": stem + ".jpg", "annot_path": stem + ".json"})
                buffer = io.BytesIO()
                fixture_image().save(buffer, format="JPEG")
                z.writestr(f"invoices_dataset_final/images/{stem}.jpg", buffer.getvalue())
                annotation = {
                    "NUMBER": {"text": f"Invoice #{template}-{instance}"},
                    "DATE": {"text": "Date: 01-Jan-2020"},
                    "DUE_DATE": None, "SUB_TOTAL": {"text": "Subtotal: 10.00 USD"},
                    "TAX": {"text": "Tax: 2.00 USD"}, "TOTAL": {"text": "TOTAL: 12.00 USD"},
                }
                z.writestr(f"invoices_dataset_final/Annotations/Original_Format/{stem}.json", json.dumps(annotation))
            table = io.StringIO()
            writer = csv.DictWriter(table, fieldnames=("img_path", "annot_path"))
            writer.writeheader()
            writer.writerows(rows)
            z.writestr(f"invoices_dataset_final/strat1_{split}.csv", table.getvalue())
    return archive


def hiertext_fixture(directory):
    directory = Path(directory)
    (directory / "gt").mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        image_id = f"fixture_{split}"
        (directory / split).mkdir(exist_ok=True)
        fixture_image().save(directory / split / f"{image_id}.jpg")
        (directory / split / "selection.json").write_text(json.dumps({"images": [image_id + ".jpg"], "complete": True}))
        lines = [
            {"text": "Hello world", "legible": True, "vertical": False, "vertices": [[2, 2], [90, 2], [90, 20], [2, 20]]},
            {"text": "Invoice 12", "legible": True, "vertical": False, "vertices": [[2, 25], [90, 25], [90, 45], [2, 45]]},
            {"text": "Ignored", "legible": False, "vertical": False, "vertices": [[0, 0], [10, 0], [10, 10], [0, 10]]},
        ]
        annotation = {"image_id": image_id, "image_width": 96, "image_height": 72,
                      "paragraphs": [{"lines": lines}]}
        with gzip.open(directory / "gt" / f"{split}.jsonl.gz", "wt") as handle:
            json.dump({"annotations": [annotation]}, handle)
    return directory


def smoke_model(directory):
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import MBartConfig, XLMRobertaTokenizerFast
    from unittest.mock import patch
    from pilot.modeling import PILOTModel
    from pilot.training.io import export_model

    torch.set_num_threads(1)
    torch.manual_seed(7)
    directory = Path(directory)
    vocabulary = {token: i for i, token in enumerate(("<s>", "<pad>", "</s>", "<unk>", "<ocr>"))}
    for token in sorted(pre_tokenizers.ByteLevel.alphabet()):
        vocabulary.setdefault(token, len(vocabulary))
    backend = Tokenizer(models.BPE(vocab=vocabulary, merges=[], unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = XLMRobertaTokenizerFast(tokenizer_object=backend, bos_token="<s>", pad_token="<pad>",
                                       eos_token="</s>", unk_token="<unk>", additional_special_tokens=["<ocr>"])

    class TinyEncoder(torch.nn.Sequential):
        def __init__(self, _):
            super().__init__(torch.nn.Conv2d(3, 8, kernel_size=1), torch.nn.AdaptiveAvgPool2d((2, 2)))

    def tiny_config(**kwargs):
        return MBartConfig(**kwargs, d_model=8, decoder_attention_heads=2, decoder_ffn_dim=16,
                           dropout=0.0, attention_dropout=0.0, activation_dropout=0.0)

    patches = [patch("pilot.modeling.PILOTEncoder", TinyEncoder), patch("pilot.models.decoder.MBartConfig", tiny_config)]
    for item in patches:
        item.start()
    config = {"name": "smoke", "encoder": {"dropout": 0.0}, "decoder": {"num_layers": 1, "max_length": 512},
              "preprocessing": {"mean": [127] * 3, "std": [50] * 3}}
    model = PILOTModel({**config, "decoder": {**config["decoder"], "tokenizer": tokenizer}})
    export_model(model, tokenizer, config, directory)
    # Keep patches alive for the lifetime of this test kernel.
    return directory / "config.json", patches
