from collections import Counter
import html
import re
import unicodedata


def edit_distance(reference, prediction):
    """Levenshtein distance using memory linear in the shorter sequence."""
    if reference == prediction:
        return 0
    if len(reference) < len(prediction):
        reference, prediction = prediction, reference
    previous = list(range(len(prediction) + 1))
    for i, left in enumerate(reference, 1):
        current = [i]
        for j, right in enumerate(prediction, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def recognition_text(text):
    return re.sub(r"<(?:[xy]_\d+|sep/)>", "", text)


def is_ocr_prompt(prompt):
    return prompt in {"<ocr>", "<ocr_with_boxes>", "<ocr_on_box>"} or prompt.startswith("<ocr_on_box> ")


def sequence_metrics(records):
    if not records:
        return {}
    result = {"sequence_exact_match": sum(r["target"] == r["prediction"] for r in records) / len(records)}
    ocr = [r for r in records if is_ocr_prompt(r["prompt"])]
    if ocr:
        texts = [(recognition_text(r["target"]), recognition_text(r["prediction"])) for r in ocr]
        chars = sum(len(ref) for ref, _ in texts)
        words = sum(len(ref.split()) for ref, _ in texts)
        result["cer"] = sum(edit_distance(ref, pred) for ref, pred in texts) / max(chars, 1)
        result["wer"] = sum(edit_distance(ref.split(), pred.split()) for ref, pred in texts) / max(words, 1)
    return result


def parse_fields(sequence, field_names):
    sequence = str(sequence).strip()
    pattern = r"\s*".join(re.escape(f"<{name}>") + f"(?P<f{i}>.*?)" + re.escape(f"</{name}>")
                          for i, name in enumerate(field_names))
    strict = re.fullmatch(pattern, sequence, flags=re.DOTALL)
    result = {}
    valid = strict is not None
    for i, name in enumerate(field_names):
        match = re.search(re.escape(f"<{name}>") + r"(.*?)" + re.escape(f"</{name}>"),
                          sequence, flags=re.DOTALL)
        payload = strict.group(f"f{i}") if strict else match.group(1) if match else ""
        if re.search(r"<[^>]*>", payload.replace("<sep/>", "")):
            valid = False
        result[name] = [html.unescape(value.strip()) for value in payload.split("<sep/>") if value.strip()]
    return result, valid


def normalize_value(text):
    return " ".join(unicodedata.normalize("NFKC", text).split()).casefold()


def field_metrics(predictions, field_names):
    """Exact normalized multiset matching; repeated equal values count separately."""
    totals = {name: [0, 0, 0, 0] for name in field_names}
    valid = exact_documents = 0
    for row in predictions:
        predicted, parsed = parse_fields(row["prediction"], field_names)
        gold, _ = parse_fields(row["target"], field_names)
        valid += parsed
        document_exact = True
        for name in field_names:
            pred = Counter(normalize_value(value) for value in predicted[name])
            truth = Counter(normalize_value(value) for value in gold[name])
            tp = sum((pred & truth).values())
            fp, fn = sum((pred - truth).values()), sum((truth - pred).values())
            counts = totals[name]
            counts[0] += tp
            counts[1] += fp
            counts[2] += fn
            counts[3] += pred == truth
            document_exact &= pred == truth
        exact_documents += document_exact and parsed
    table = []
    for name, (tp, fp, fn, exact) in totals.items():
        table.append({"field": name, "tp": tp, "fp": fp, "fn": fn, "precision": tp / max(tp + fp, 1),
                      "recall": tp / max(tp + fn, 1), "f1": 2 * tp / max(2 * tp + fp + fn, 1),
                      "exact_match": exact / max(len(predictions), 1)})
    tp, fp, fn = (sum(values[i] for values in totals.values()) for i in range(3))
    summary = {"examples": len(predictions), "field_micro_f1": 2 * tp / max(2 * tp + fp + fn, 1),
               "parse_valid_rate": valid / max(len(predictions), 1),
               "document_exact_match": exact_documents / max(len(predictions), 1)}
    return summary, table
