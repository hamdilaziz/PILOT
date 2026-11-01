from __future__ import annotations

from collections import defaultdict
import csv
import html
import io
import json
from pathlib import Path
import random
import re
import zipfile

from PIL import Image
from tqdm.auto import tqdm

from .common import download_file, finish_preparation, prepared_cache, save_rgb

FATURA_URL = "https://zenodo.org/records/10371464/files/FATURA2.zip?download=1"
FATURA_SIZE = 690717658
FATURA_MD5 = "4c9404462f22c5241eb1a290a02eb2a2"
TASK_PROMPT = "<invoice_extract>"
FIELDS = {
    "invoice_number": "NUMBER",
    "invoice_date": "DATE",
    "due_date": "DUE_DATE",
    "subtotal": "SUB_TOTAL",
    "tax": "TAX",
    "total": "TOTAL",
}
SPECIAL_TOKENS = [TASK_PROMPT, "<sep/>", *(token for name in FIELDS for token in (f"<{name}>", f"</{name}>"))]


def download_fatura(path):
    return download_file(FATURA_URL, path, expected_size=FATURA_SIZE, md5=FATURA_MD5)


def annotation_fields(annotation):
    """Keep the source region's text, including its printed label and currency."""
    result = {}
    for name, source in FIELDS.items():
        value = annotation.get(source)
        regions = value if isinstance(value, list) else [value]
        texts = []
        for region in regions:
            if region is None:
                continue
            if not isinstance(region, dict):
                raise ValueError(f"Unexpected annotation type for {source}: {type(region).__name__}")
            text = " ".join(str(region.get("text", "")).split())
            if text:
                texts.append(text)
        result[name] = texts
    return result


def serialize_fields(values):
    return "".join(f"<{name}>" + "<sep/>".join(html.escape(v, quote=False) for v in values[name]) +
                   f"</{name}>" for name in FIELDS)


def balanced_subset(rows, limit, seed):
    """Sample across templates instead of taking only the first template in a CSV."""
    groups = defaultdict(list)
    for row in rows:
        match = re.fullmatch(r"Template(\d+)_Instance(\d+)\.jpg", Path(row["img_path"]).name)
        if not match:
            raise ValueError(f"Unexpected FATURA image filename: {row['img_path']}")
        groups[int(match.group(1))].append(row)
    rng = random.Random(seed)
    keys = sorted(groups)
    rng.shuffle(keys)
    for group in groups.values():
        rng.shuffle(group)
    selected = []
    while keys and (limit is None or len(selected) < limit):
        active = []
        for key in keys:
            if limit is not None and len(selected) >= limit:
                break
            selected.append(groups[key].pop())
            if groups[key]:
                active.append(key)
        keys = active
    return selected


def prepare_fatura(archive, prepared_dir, *, limits=None, max_side=1280, seed=42):
    limits = limits or {"train": 400, "validation": 80, "test": 80}
    if set(limits) != {"train", "validation", "test"} or any(n is not None and n < 1 for n in limits.values()):
        raise ValueError("Provide positive train/validation/test limits, or None for a complete split.")
    settings = {"version": 1, "dataset": "FATURA2", "protocol": "official_strat1_white_images",
                "limits": limits, "max_side": max_side, "seed": seed, "fields": FIELDS, "field_order": list(FIELDS)}
    cached = prepared_cache(prepared_dir, settings)
    if cached:
        return cached
    prepared_dir = Path(prepared_dir)
    records = {}
    templates = {}
    with zipfile.ZipFile(archive) as z:
        # Ground paths in actual archive members; do not assume its outer directory.
        names = z.namelist()
        roots = [name.rsplit("/", 1)[0] for name in names if name.endswith("/strat1_train.csv")]
        if len(roots) != 1:
            raise ValueError("Expected one FATURA2 root with strat1_train.csv.")
        root = roots[0]
        white_members = {Path(name).name: name for name in names
                         if "/white_images/" in name and name.lower().endswith(".jpg")}
        if not white_members:
            # The official archive calls its white-background folder 'images'.
            white_members = {Path(name).name: name for name in names
                             if name.startswith(root + "/images/") and name.lower().endswith(".jpg")}
        if not white_members:
            raise ValueError("Could not locate the white-background invoice images.")
        for split, original in (("train", "train"), ("validation", "dev"), ("test", "test")):
            table = list(csv.DictReader(io.StringIO(z.read(f"{root}/strat1_{original}.csv").decode("utf-8-sig"))))
            selected = balanced_subset(table, limits[split], seed)
            records[split] = []
            for row in tqdm(selected, desc=f"Prepare FATURA {split}"):
                filename = Path(row["img_path"]).name
                annotation_name = Path(row["annot_path"]).name
                annotation = json.loads(z.read(f"{root}/Annotations/Original_Format/{annotation_name}"))
                fields = annotation_fields(annotation)
                image_name = f"images/{split}/{Path(filename).stem}.png"
                with Image.open(io.BytesIO(z.read(white_members[filename]))) as image:
                    save_rgb(image, prepared_dir / image_name, max_side)
                template = int(re.match(r"Template(\d+)", filename).group(1))
                records[split].append({"image": image_name, "prompt": TASK_PROMPT,
                                       "target": serialize_fields(fields), "fields": fields,
                                       "source_id": Path(filename).stem, "template_id": template})
            templates[split] = sorted({row["template_id"] for row in records[split]})
    return finish_preparation(prepared_dir, records, settings, {
        "source": "https://zenodo.org/records/10371464", "archive_md5": FATURA_MD5,
        "templates": templates, "task_prompt": TASK_PROMPT, "extra_tokens": SPECIAL_TOKENS,
        "field_presence": {split: {name: sum(bool(row["fields"][name]) for row in rows) for name in FIELDS}
                           for split, rows in records.items()},
        "target_definition": "Original_Format field-region text, including printed labels; whitespace normalized.",
    })
