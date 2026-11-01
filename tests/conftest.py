import json

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import MBartConfig, XLMRobertaTokenizerFast


@pytest.fixture
def tokenizer():
    vocab = {word: i for i, word in enumerate(
        ("<s>", "<pad>", "</s>", "<unk>", "<ocr>", "<x_0>", "hello", "world", "field", "<", ">", "/")
    )}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return XLMRobertaTokenizerFast(tokenizer_object=backend, bos_token="<s>", eos_token="</s>", pad_token="<pad>",
                                  unk_token="<unk>", additional_special_tokens=["<ocr>", "<x_0>"])


@pytest.fixture
def tiny_model(monkeypatch, tokenizer):
    from pilot import modeling
    from pilot.models import decoder

    torch.set_num_threads(1)

    class TinyEncoder(torch.nn.Conv2d):
        def __init__(self, _):
            super().__init__(3, 8, kernel_size=3, stride=2, padding=1)

    def tiny_config(**kwargs):
        return MBartConfig(**kwargs, d_model=8, decoder_attention_heads=2, decoder_ffn_dim=16,
                           dropout=0.0, attention_dropout=0.0, activation_dropout=0.0)

    monkeypatch.setattr(modeling, "PILOTEncoder", TinyEncoder)
    monkeypatch.setattr(decoder, "MBartConfig", tiny_config)

    def factory():
        return modeling.PILOTModel({"encoder": {"dropout": 0.0},
                                    "decoder": {"tokenizer": tokenizer, "num_layers": 1, "max_length": 16}})
    return factory


@pytest.fixture
def small_run(tmp_path, tokenizer, tiny_model):
    from PIL import Image
    from pilot.training.io import export_model

    public_config = {"encoder": {"dropout": 0.0}, "decoder": {"num_layers": 1, "max_length": 16},
                     "preprocessing": {"mean": [127, 127, 127], "std": [50, 50, 50]}}
    export_model(tiny_model(), tokenizer, public_config, tmp_path / "pretrained")
    for split in ("train", "valid"):
        Image.new("RGB", (4, 4), color=(240, 240, 240)).save(tmp_path / f"{split}.png")
        records = [{"image": f"{split}.png", "prompt": "<ocr>", "target": text}
                   for text in ("hello", "hello world", "world")]
        (tmp_path / f"{split}.jsonl").write_text("\n".join(json.dumps(row) for row in records))
    config = {"model_config": "pretrained/config.json", "train_manifest": "train.jsonl",
              "validation_manifest": "valid.jsonl", "output_dir": "run", "epochs": 2,
              "max_length": 8, "accumulation_steps": 2, "learning_rate": 0.001, "precision": "fp32", "seed": 12}
    filename = tmp_path / "training.json"
    filename.write_text(json.dumps(config))
    return filename
