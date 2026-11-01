# Contributing

```bash
pip install -e ".[dev,notebooks]"
ruff check pilot tests train_pilot.py run_pilot.py examples
pytest -q
```

Keep checkpoint compatibility when modifying parameter names, tokenizer IDs, preprocessing
or coordinates. Include focused regression tests and use synthetic images or public examples.
Bug reports should include the command, full error, versions, device, model config and checkpoint.
Accuracy/performance changes should include held-out evaluation and hardware details.
