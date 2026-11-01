import json

from tokenizers import Tokenizer
from transformers import XLMRobertaTokenizerFast


def fix_tokenizer(tokenizer):
    """Preserve newline token IDs while fixing legacy AddedToken normalization."""
    backend = json.loads(tokenizer.backend_tokenizer.to_str())
    changed = False
    for token in backend.get("added_tokens", []):
        if token["content"] == "\n" and token["normalized"]:
            token["normalized"] = False
            changed = True
    if changed:
        tokenizer._tokenizer = Tokenizer.from_str(json.dumps(backend, ensure_ascii=False))
    return tokenizer


def load_tokenizer(path):
    return fix_tokenizer(XLMRobertaTokenizerFast.from_pretrained(str(path)))
