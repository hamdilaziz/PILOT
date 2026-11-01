from __future__ import annotations

import gzip
import json
from pathlib import Path
import tarfile

import ijson
from PIL import Image
from tqdm.auto import tqdm

from .common import download_file, finish_preparation, http_session, prepared_cache, save_json, save_rgb

HIER_REVISION = "70b6620b2b112597d8219e11eee9773a1403827c"
HIER_GT_SIZES = {"train": 59214216, "validation": 12274736, "test": 11342161}
HIER_IMAGE_COUNTS = {"train": 8281, "validation": 1724, "test": 1634}


class ProgressReader:
    def __init__(self, stream, progress):
        self.stream, self.progress = stream, progress

    def read(self, size=-1):
        chunk = self.stream.read(size)
        self.progress.update(len(chunk))
        return chunk


def source_images(raw_dir, split, limit):
    """Cache an archive prefix; stop the HTTP stream once enough images are saved."""
    directory = Path(raw_dir) / split
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / "selection.json"
    if marker.is_file():
        saved = json.loads(marker.read_text())
        images = [directory / name for name in saved["images"]]
        complete = saved["complete"] if limit is None else len(images) >= limit
        if complete and all(path.is_file() for path in images):
            return images if limit is None else images[:limit]
    url = f"https://open-images-dataset.s3.amazonaws.com/ocr/{split}.tgz?download=1"
    images = []
    complete = True
    with http_session().get(url, stream=True, timeout=(15, 120)) as response:
        response.raise_for_status()
        if response.status_code != 200:
            raise RuntimeError("Expected a complete archive stream; received an unexpected partial HTTP response.")
        with tqdm(unit="B", unit_scale=True, desc=f"Stream HierText {split}") as byte_progress, (
            tqdm(total=limit, desc=f"Cache {split} source images")
        ) as image_progress:
            reader = ProgressReader(response.raw, byte_progress)
            with tarfile.open(fileobj=reader, mode="r|gz") as tar:
                for member in tar:
                    if not member.isfile() or not member.name.lower().endswith(".jpg"):
                        continue
                    destination = directory / Path(member.name).name
                    if not destination.is_file():
                        temporary = destination.with_suffix(".part")
                        temporary.write_bytes(tar.extractfile(member).read())
                        temporary.replace(destination)
                    images.append(destination)
                    image_progress.update()
                    if limit is not None and len(images) >= limit:
                        complete = False
                        break
    if not images:
        raise ValueError(f"No source images in HierText {split} archive.")
    save_json(marker, {"url": url, "images": [path.name for path in images], "complete": complete})
    return images


def line_samples(annotation, *, max_lines=8, max_characters=160):
    """Keep complete legible, nonvertical lines; retain source paragraph/line indices."""
    result = []
    for pi, paragraph in enumerate(annotation["paragraphs"]):
        for li, line in enumerate(paragraph["lines"]):
            text = " ".join(line.get("text", "").split())
            if not line.get("legible") or line.get("vertical") or not 2 <= len(text) <= max_characters:
                continue
            vertices = line["vertices"]
            xs, ys = zip(*vertices, strict=True)
            box = (float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys)))
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            result.append({"line_id": f"p{pi}_l{li}", "target": text, "box": box})
    # Reading order between lines is irrelevant: each line becomes an independent image.
    return result if max_lines is None else result[:max_lines]


def prepare_hiertext(raw_dir, prepared_dir, *, image_limits=None, max_lines=8, max_side=1280, line_height=64):
    if line_height < 1 or max_lines is not None and max_lines < 1:
        raise ValueError("line_height and max_lines must be positive; max_lines may also be None.")
    image_limits = image_limits or {"train": 50, "validation": 10, "test": 10}
    if set(image_limits) != {"train", "validation", "test"} or any(
        n is not None and n < 1 for n in image_limits.values()
    ):
        raise ValueError("Provide positive source-image limits for all splits, or None.")
    settings = {"version": 2, "dataset": "HierText", "revision": HIER_REVISION, "line_height": line_height,
                "image_limits": image_limits, "max_lines_per_image": max_lines, "max_side": max_side,
                "max_characters": 160, "task": "legible_nonvertical_line_ocr"}
    cached = prepared_cache(prepared_dir, settings)
    if cached:
        return cached
    raw_dir, prepared_dir = Path(raw_dir), Path(prepared_dir)
    records = {}
    selections = {}
    for split in ("train", "validation", "test"):
        url = f"https://raw.githubusercontent.com/google-research-datasets/hiertext/{HIER_REVISION}/gt/{split}.jsonl.gz"
        annotations = download_file(url, raw_dir / "gt" / f"{split}.jsonl.gz", expected_size=HIER_GT_SIZES[split])
        images = source_images(raw_dir, split, image_limits[split])
        selected = {path.stem: path for path in images}
        selections[split] = sorted(selected)
        records[split] = []
        found = set()
        with gzip.open(annotations, "rb") as handle, tqdm(
            total=HIER_IMAGE_COUNTS[split], desc=f"Scan HierText {split} annotations"
        ) as progress:
            for annotation in ijson.items(handle, "annotations.item", use_float=True):
                progress.update()
                image_id = annotation["image_id"]
                if image_id not in selected:
                    continue
                found.add(image_id)
                with Image.open(selected[image_id]) as image:
                    expected = (annotation["image_width"], annotation["image_height"])
                    if image.size != expected:
                        raise ValueError(f"{image_id}: image size {image.size} differs from official annotations {expected}.")
                    for sample in line_samples(annotation, max_lines=max_lines):
                        x1, y1, x2, y2 = sample["box"]
                        box = (max(0, int(x1) - 2), max(0, int(y1) - 2),
                               min(image.width, int(x2 + 1) + 2), min(image.height, int(y2 + 1) + 2))
                        image_name = f"images/{split}/{image_id}_{sample['line_id']}.png"
                        save_rgb(image.crop(box), prepared_dir / image_name, max_side, target_height=line_height)
                        records[split].append({"image": image_name, "prompt": "<ocr>", "target": sample["target"],
                                               "source_id": image_id, "line_id": sample["line_id"], "source_box": box})
                progress.set_postfix_str(f"selected={len(found)}/{len(selected)}", refresh=False)
                if len(found) == len(selected):
                    break
        missing = set(selected) - found
        if missing:
            raise ValueError(f"Missing official annotations for images: {sorted(missing)[:5]}")
    return finish_preparation(prepared_dir, records, settings, {
        "source": "https://github.com/google-research-datasets/hiertext", "source_images": selections,
        "task_prompt": "<ocr>", "extra_tokens": [], "selection": "First source images in each official archive.",
        "target_definition": "Legible nonvertical line text; whitespace normalized. This is recognition on GT crops.",
    })
