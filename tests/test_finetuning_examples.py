import importlib.util
from pathlib import Path
import runpy

import numpy as np
import pytest
from PIL import Image
import torch

from examples.finetuning.common import read_jsonl
from examples.finetuning.evaluation import field_metrics, parse_fields
from examples.finetuning.fatura import FIELDS, annotation_fields, balanced_subset, prepare_fatura, serialize_fields
from examples.finetuning.hiertext import line_samples, prepare_hiertext
from pilot.training.data import corrupt_inputs

SUPPORT = runpy.run_path(str(Path(__file__).with_name("notebook_support.py")))


def test_fatura_preserves_labeled_source_text_and_repeated_values():
    values = annotation_fields({"NUMBER": {"text": " Invoice  #12\n "},
                                "TOTAL": [{"text": "TOTAL: 12 & 3 USD"}, {"text": "TOTAL: 12 & 3 USD"}]})
    target = serialize_fields(values)
    assert "&amp;" in target
    parsed, valid = parse_fields(target, list(FIELDS))
    assert valid and parsed["invoice_number"] == ["Invoice #12"]
    assert parsed["total"] == ["TOTAL: 12 & 3 USD", "TOTAL: 12 & 3 USD"]
    assert parsed["due_date"] == []
    prediction = target.replace("<sep/>TOTAL: 12 &amp; 3 USD", "")
    summary, table = field_metrics([{"target": target, "prediction": prediction}], list(FIELDS))
    total = next(row for row in table if row["field"] == "total")
    assert total["tp"] == 1 and total["fn"] == 1 and total["fp"] == 0
    assert summary["document_exact_match"] == 0
    assert not parse_fields(target + "<unknown>extra</unknown>", list(FIELDS))[1]


def test_fatura_subsample_covers_templates():
    rows = [{"img_path": f"Template{template}_Instance{instance}.jpg"} for template in range(1, 6)
            for instance in range(10)]
    selected = balanced_subset(rows, 5, 42)
    assert len({row["img_path"].split("_")[0] for row in selected}) == 5
    assert selected == balanced_subset(rows, 5, 42)


def test_fatura_cache_train_stats_and_split_provenance(tmp_path):
    archive = SUPPORT["fatura_fixture"](tmp_path / "raw")
    output = tmp_path / "prepared"
    limits = {"train": 2, "validation": 2, "test": 2}
    result = prepare_fatura(archive, output, limits=limits)
    assert result["counts"] == limits
    sets = [{row["source_id"] for row in read_jsonl(output / f"{split}.jsonl")} for split in limits]
    assert all(not sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3))
    training_pixels = []
    for row in read_jsonl(output / "train.jsonl"):
        with Image.open(output / row["image"]) as image:
            training_pixels.append(np.asarray(image).reshape(-1, 3))
    expected = np.concatenate(training_pixels)
    np.testing.assert_allclose(result["image_stats"]["mean"], expected.mean(axis=0))
    assert prepare_fatura(archive, output, limits=limits) == result
    with pytest.raises(ValueError, match="settings changed"):
        prepare_fatura(archive, output, limits={**limits, "train": 1})


def test_hiertext_filters_lines_and_keeps_source_splits(tmp_path, monkeypatch):
    import examples.finetuning.hiertext as data

    raw = SUPPORT["hiertext_fixture"](tmp_path / "raw")
    monkeypatch.setattr(data, "HIER_GT_SIZES", {
        split: (raw / "gt" / f"{split}.jsonl.gz").stat().st_size for split in ("train", "validation", "test")
    })
    result = prepare_hiertext(raw, tmp_path / "prepared",
                              image_limits={"train": 1, "validation": 1, "test": 1}, max_lines=8)
    assert result["counts"] == {"train": 2, "validation": 2, "test": 2}
    assert result["source_counts"] == {"train": 1, "validation": 1, "test": 1}
    for row in read_jsonl(tmp_path / "prepared/train.jsonl"):
        assert row["source_id"] == "fixture_train"
        assert row["target"] != "Ignored"
        with Image.open(tmp_path / "prepared" / row["image"]) as image:
            assert min(image.size) >= 64
    annotation = {"paragraphs": [{"lines": [{"text": "Vertical", "legible": True, "vertical": True}]}]}
    assert line_samples(annotation) == []


def test_sequence_noise_draws_only_from_lexical_tokens_in_its_own_row(tokenizer):
    ids = torch.tensor([[0, 4, 6, 7, 2], [0, 4, 5, 2, 1]])
    for _ in range(4):
        result = corrupt_inputs(ids, ids.ne(1), [{"prompt_length": 1}] * 2, tokenizer, 1.0, "sequence")
        assert set(result[0, 2:4].tolist()) <= {6, 7}
        assert torch.equal(result[1], ids[1])
        assert torch.equal(result[:, :2], ids[:, :2])


@pytest.mark.parametrize("filename,dataset", [("finetune_fatura.ipynb", "fatura"),
                                             ("finetune_hiertext_ocr.ipynb", "hiertext")])
def test_notebook_code_cells_execute_end_to_end_on_cpu(tmp_path, filename, dataset):
    if not importlib.util.find_spec("nbformat"):
        pytest.skip("Install the dev and notebook extras for executable notebook tests.")
    import nbformat
    import os
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    notebook = nbformat.read(root / "examples" / filename, as_version=4)
    work = tmp_path / "work"
    setup = (
        f"\nimport runpy\nsupport = runpy.run_path({str(root / 'tests/notebook_support.py')!r})\n"
        f"MODEL_CONFIG, _patches = support['smoke_model']({str(work / 'base')!r})\n"
        f"DATA_DIR = Path({str(work / 'data')!r})\nPREPARED_DIR = DATA_DIR / 'prepared'\n"
        f"EXPERIMENT_DIR = Path({str(work / 'experiment')!r})\nMAX_SIDE = 128\nEPOCHS = 1\n"
    )
    if dataset == "fatura":
        archive = SUPPORT["fatura_fixture"](work / "source")
        setup += f"ARCHIVE_PATH = Path({str(archive)!r})\ndownload_fatura = lambda path: ARCHIVE_PATH\n"
        setup += "LIMITS = {'train': 2, 'validation': 2, 'test': 2}\n"
    else:
        raw = SUPPORT["hiertext_fixture"](work / "source")
        setup += (
            f"RAW_DIR = Path({str(raw)!r})\nIMAGE_LIMITS = {{'train': 1, 'validation': 1, 'test': 1}}\n"
            "import examples.finetuning.hiertext as _hier\n"
            "_hier.HIER_GT_SIZES = {s: (RAW_DIR / 'gt' / f'{s}.jsonl.gz').stat().st_size for s in IMAGE_LIMITS}\n"
        )
    parameters = next(cell for cell in notebook.cells if cell.get("id") == "parameters")
    parameters.source += setup
    script = "\n\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    completed = subprocess.run([sys.executable, "-c", script], cwd=root,
                               env={**os.environ, "MPLBACKEND": "Agg"}, capture_output=True, text=True, timeout=240)
    assert completed.returncode == 0, completed.stdout[-5000:] + completed.stderr[-5000:]
    assert (work / "experiment/training/best/model.pt").is_file()
    assert (work / "experiment/training/test_metrics.json").is_file()
