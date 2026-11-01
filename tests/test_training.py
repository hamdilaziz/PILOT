import copy
import json
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader

from pilot.loading import load_pilot_model
from pilot.training.data import IGNORE_INDEX, SequenceCollator, SequenceDataset, corrupt_inputs
from pilot.training.metrics import sequence_metrics
from pilot.training.trainer import summed_loss, train, train_epoch


def test_tokens_preserve_ids_special_flags_and_subword_initialization(tokenizer, tiny_model):
    model = tiny_model()
    before = tokenizer.get_vocab().copy()
    old_weights = model.decoder.model.get_input_embeddings().weight.detach().clone()
    assert model.decoder.add_special_tokens(["<field>", "</field>", "<field>"]) == 2
    assert all(tokenizer.get_vocab()[token] == index for token, index in before.items())
    assert {"<ocr>", "<x_0>", "<field>", "</field>"} <= set(tokenizer.all_special_tokens)
    assert tokenizer.encode("<field>", add_special_tokens=False) == [len(before)]
    weights = model.decoder.model.get_input_embeddings().weight
    torch.testing.assert_close(weights[:len(before)], old_weights[:len(before)])
    pieces = old_weights[[before["<"], before["field"], before[">"]]].mean(dim=0)
    torch.testing.assert_close(weights[tokenizer.convert_tokens_to_ids("<field>")], pieces)
    assert model.decoder.add_special_tokens(["<field>"]) == 0


def test_teacher_forcing_masks_prompt_padding_but_learns_first_target_and_eos():
    collate = SequenceCollator(1)
    samples = [
        {"pixels": torch.zeros(3, 4, 4), "ids": [0, 4, 6, 7, 2], "prompt_length": 1},
        {"pixels": torch.zeros(3, 4, 4), "ids": [0, 4, 6, 2], "prompt_length": 1},
    ]
    batch = collate(samples)
    assert batch["input_ids"].tolist() == [[0, 4, 6, 7], [0, 4, 6, 2]]
    assert batch["labels"].tolist() == [[IGNORE_INDEX, 6, 7, 2], [IGNORE_INDEX, 6, 2, IGNORE_INDEX]]


def test_input_noise_protects_prompt_and_structural_tokens(tokenizer):
    ids = torch.tensor([[0, 4, 5, 6, 7, 2, 1]])
    before = ids.clone()
    for _ in range(5):
        noisy = corrupt_inputs(ids, ids.ne(1), [{"prompt_length": 1}], tokenizer, 1.0)
        assert noisy[0, [0, 1, 2, 5, 6]].tolist() == ids[0, [0, 1, 2, 5, 6]].tolist()
        assert not set(noisy[0, 3:5].tolist()) & set(tokenizer.all_special_ids)
    assert torch.equal(ids, before)


def test_overflow_fails_instead_of_truncating(small_run, tokenizer):
    with pytest.raises(ValueError, match="exceed max_length"):
        SequenceDataset(small_run.parent / "train.jsonl", tokenizer, [0] * 3, [1] * 3, 3)


def test_unequal_image_batches_are_rejected():
    with pytest.raises(ValueError, match="equal dimensions"):
        SequenceCollator(1)([{"pixels": torch.zeros(3, 4, 4)}, {"pixels": torch.zeros(3, 4, 8)}])


def test_accumulation_matches_token_weighted_updates_including_partial_window(tokenizer, tiny_model):
    model = tiny_model()
    reference = copy.deepcopy(model)
    samples = [{"pixels": torch.ones(3, 4, 4), "ids": ids, "prompt_length": 1}
               for ids in ([0, 4, 6, 2], [0, 4, 6, 7, 2], [0, 4, 7, 2])]
    collate = SequenceCollator(tokenizer.pad_token_id)
    loader = DataLoader(samples, batch_size=1, collate_fn=collate)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
    train_epoch(model, tokenizer, loader, optimizer, scheduler, torch.amp.GradScaler("cuda", enabled=False),
                SimpleNamespace(noise_rate=0, accumulation_steps=2, grad_clip_norm=1e6),
                torch.device("cpu"), torch.float32)
    reference.train()
    for group in (samples[:2], samples[2:]):
        batch = collate(group)
        reference_optimizer.zero_grad()
        logits = reference(batch["images"], input_ids=batch["input_ids"],
                           attention_mask=batch["attention_mask"], use_cache=False).logits
        (summed_loss(logits, batch["labels"]) / batch["labels"].ne(IGNORE_INDEX).sum()).backward()
        reference_optimizer.step()
    for parameter, expected in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(parameter, expected, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("custom_task", [False, True])
def test_export_reload_generate_and_resume_after_interruption(small_run, monkeypatch, custom_task):
    import pilot.training.trainer as trainer
    if custom_task:
        settings = json.loads(small_run.read_text())
        settings["extra_tokens"] = ["<extract>", "<field>", "</field>"]
        settings["gradient_checkpointing"] = True
        settings["selection_metric"] = "field_micro_f1"
        settings["metric_fields"] = ["field"]
        small_run.write_text(json.dumps(settings))
        for split in ("train", "valid"):
            path = small_run.parent / f"{split}.jsonl"
            records = [json.loads(line) for line in path.read_text().splitlines()]
            for record in records:
                record["prompt"] = "<extract>"
                record["target"] = f"<field>{record['target']}</field>"
            path.write_text("\n".join(json.dumps(record) for record in records))
    baseline = train(small_run)
    root = small_run.parent
    expected = torch.load(root / "run/last.pt", weights_only=True, map_location="cpu")
    model, tokenizer, _ = load_pilot_model(root / "run/best/config.json", torch.device("cpu"))
    if custom_task:
        assert {"<ocr>", "<extract>", "<field>", "</field>"} <= set(tokenizer.all_special_tokens)
    ids = torch.tensor([[tokenizer.bos_token_id, tokenizer.convert_tokens_to_ids("<ocr>")]])
    generated = model.generate(torch.zeros(1, 3, 4, 4), ids, max_new_tokens=2)
    assert generated.shape[1] == ids.shape[1] + 2
    assert torch.equal(generated[:, :ids.shape[1]], ids)

    values = json.loads(small_run.read_text())
    values["output_dir"] = "resumed"
    small_run.write_text(json.dumps(values))
    original = trainer.train_epoch
    calls = 0

    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated interruption")
        return original(*args, **kwargs)

    monkeypatch.setattr(trainer, "train_epoch", interrupted)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        train(small_run)
    monkeypatch.setattr(trainer, "train_epoch", original)
    result = train(small_run, resume=True)
    actual = torch.load(root / "resumed/last.pt", weights_only=True, map_location="cpu")
    assert result["best"] == baseline["best"]
    for key, value in expected["model"].items():
        torch.testing.assert_close(value, actual["model"][key], rtol=0, atol=0)


def test_overlapping_images_are_rejected(small_run):
    values = json.loads(small_run.read_text())
    values["validation_manifest"] = "train.jsonl"
    small_run.write_text(json.dumps(values))
    with pytest.raises(ValueError, match="share"):
        train(small_run)


def test_metrics_keep_repeated_values_and_report_cer_without_coordinates():
    values = sequence_metrics([{"prompt": "<ocr_with_boxes>", "target": "<x_0><y_0>abc<x_1><y_1>",
                                "prediction": "<x_0><y_0>adc<x_1><y_1>"}])
    assert values == {"sequence_exact_match": 0.0, "cer": 1 / 3, "wer": 1.0}


def test_encoder_eval_does_not_advance_python_rng():
    import random
    from pilot.models.encoder import ConvBlock, DSCBlockOriginal, MixDropout
    for block in (MixDropout(), ConvBlock(4, 4), DSCBlockOriginal(4, 4, stride=(1, 1))):
        block.eval()
        before = random.getstate()
        inputs = torch.randn(1, 4, 8, 8)
        first = block(inputs)
        second = block(inputs)
        assert random.getstate() == before
        torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_cached_generation_matches_uncached_and_predict_runs(tokenizer, tiny_model):
    model = tiny_model().eval()
    images = torch.randn(1, 3, 4, 4)
    prefix = torch.tensor([[tokenizer.bos_token_id, tokenizer.convert_tokens_to_ids("<ocr>")]])
    cached = model.generate(images, prefix, max_new_tokens=4, use_cache=True)
    uncached = model.generate(images, prefix, max_new_tokens=4, use_cache=False)
    assert torch.equal(cached, uncached)
    prediction = model.predict({"imgs": images, "token_prompt": prefix[:, 1:]}, max_length=6)
    assert len(prediction["str_pred"]) == 1
    assert prediction["time"] >= 0


def test_resume_accepts_the_original_training_settings_without_noise_strategy(small_run):
    expected = train(small_run)
    checkpoint_path = small_run.parent / "run/last.pt"
    state = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    state["settings"].pop("noise_strategy")
    torch.save(state, checkpoint_path)
    assert train(small_run, resume=True) == expected


@pytest.mark.parametrize("block_name", ["ConvBlock", "DSCBlockOriginal"])
def test_groupnorm_preserves_legacy_block_outputs_gradients_and_weights(block_name):
    from pilot.models import encoder

    group = getattr(encoder, block_name)(4, 4, dropout=0.0)
    legacy = copy.deepcopy(group)
    legacy.norm_layer = torch.nn.InstanceNorm2d(4, eps=1e-3, affine=False, track_running_stats=False)
    legacy.load_state_dict(group.state_dict(), strict=True)
    assert legacy.state_dict().keys() == group.state_dict().keys()
    inputs = torch.randn(2, 4, 8, 12, requires_grad=True)
    reference_inputs = inputs.detach().clone().requires_grad_()
    actual, expected = group(inputs), legacy(reference_inputs)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    actual.square().mean().backward()
    expected.square().mean().backward()
    torch.testing.assert_close(inputs.grad, reference_inputs.grad, rtol=2e-4, atol=2e-6)
    for parameter, reference in zip(group.parameters(), legacy.parameters(), strict=True):
        torch.testing.assert_close(parameter.grad, reference.grad, rtol=2e-4, atol=2e-6)
    group.eval()
    legacy.eval()
    torch.testing.assert_close(group(inputs), legacy(inputs), rtol=2e-5, atol=2e-6)


def test_validation_only_generates_and_does_not_evaluate_a_loss(small_run, tokenizer, tiny_model, monkeypatch):
    import pilot.training.trainer as trainer

    def forbidden(*args, **kwargs):
        raise AssertionError("Validation must not use teacher forcing or loss.")

    model = tiny_model()
    monkeypatch.setattr(model, "forward", forbidden)
    monkeypatch.setattr(model.decoder, "forward", forbidden)
    monkeypatch.setattr(trainer, "summed_loss", forbidden)
    data = SequenceDataset(small_run.parent / "valid.jsonl", tokenizer, [127] * 3, [50] * 3, 8)
    loader = DataLoader(data, batch_size=1, collate_fn=SequenceCollator(tokenizer.pad_token_id))
    metrics, predictions = trainer.validate(model, tokenizer, loader, torch.device("cpu"), torch.float32, 8)
    assert len(predictions) == len(data)
    assert {"cer", "wer", "sequence_exact_match", "eos_rate"} == set(metrics)


def test_visual_cache_projects_once_and_keeps_multi_token_prompt_results(tokenizer, tiny_model, monkeypatch):
    model = tiny_model().eval()
    prefix = torch.tensor([[tokenizer.bos_token_id, tokenizer.convert_tokens_to_ids("<ocr>"),
                            tokenizer.convert_tokens_to_ids("hello")]])
    images = torch.randn(1, 3, 4, 4)
    projections = model.decoder.model.model.decoder.layers[0].encoder_attn.k_proj
    original = projections.forward
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(projections, "forward", counted)
    cached = model.generate(images, prefix, max_new_tokens=4, min_new_tokens=4, use_cache=True)
    assert len(calls) == 1
    calls.clear()
    uncached = model.generate(images, prefix, max_new_tokens=4, min_new_tokens=4, use_cache=False)
    assert len(calls) == 4
    assert torch.equal(cached, uncached)
    assert torch.equal(cached[:, :prefix.shape[1]], prefix)


def test_best_checkpoint_maximizes_generated_field_f1(small_run, monkeypatch):
    import pilot.training.trainer as trainer

    settings = json.loads(small_run.read_text())
    settings.update(extra_tokens=["<field>", "</field>"], metric_fields=["field"], selection_metric="field_micro_f1")
    small_run.write_text(json.dumps(settings))
    for split in ("train", "valid"):
        path = small_run.parent / f"{split}.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        for row in records:
            row["target"] = f"<field>{row['target']}</field>"
        path.write_text("\n".join(json.dumps(row) for row in records))
    scores = iter([0.2, 0.8])
    original = trainer.validate

    def measured(*args, **kwargs):
        metrics, predictions = original(*args, **kwargs)
        metrics["field_micro_f1"] = next(scores)
        return metrics, predictions

    exports = []
    original_export = trainer.export_model

    def exported(*args, **kwargs):
        exports.append(1)
        original_export(*args, **kwargs)

    monkeypatch.setattr(trainer, "validate", measured)
    monkeypatch.setattr(trainer, "export_model", exported)
    result = train(small_run)
    assert result["best"] == 0.8 and len(exports) == 2
    assert (small_run.parent / "run/best/model.pt").is_file()
    history = [json.loads(line) for line in (small_run.parent / "run/metrics.jsonl").read_text().splitlines()]
    assert all("loss" not in row["validation"] for row in history)


def test_cpu_optimization_preserves_ids_and_bounds_position_cache(tokenizer, tiny_model):
    from pilot.optimization import optimize_for_cpu

    model = tiny_model().eval()
    prefix = torch.tensor([[tokenizer.bos_token_id, tokenizer.convert_tokens_to_ids("<ocr>")]])
    images = torch.randn(1, 3, 4, 4)
    with torch.inference_mode():
        baseline = model.generate(images, prefix, max_new_tokens=3)
        optimize_for_cpu(model, num_threads=1, positional_cache_size=1)
        actual = model.generate(images, prefix, max_new_tokens=3)
        assert torch.equal(baseline, actual)
        assert len(model._positional_encoding_cache) == 1
        model.generate(torch.randn(1, 3, 6, 8), prefix, max_new_tokens=3)
        assert len(model._positional_encoding_cache) == 1
        assert not any("cache" in name for name in model.state_dict())
    model.train()
    model(images, input_ids=prefix, use_cache=False).logits.sum().backward()
    assert not model._positional_encoding_cache
    assert model.encoder.weight.grad is not None


def test_checkpoint_conversion_outputs_loadable_pt(tmp_path):
    from convert_checkpoints import convert_one_checkpoint

    expected = {"encoder.conv.weight": torch.randn(2, 3, 3, 3), "decoder.model.weight": torch.randn(5, 2)}
    source = tmp_path / "legacy.pt"
    target = tmp_path / "weights.pt"
    torch.save({"encoder_state_dict": {"module.conv.weight": expected["encoder.conv.weight"]},
                "decoder_state_dict": {"module.decoder.model.weight": expected["decoder.model.weight"]}}, source)
    convert_one_checkpoint(source, target, "pt")
    actual = torch.load(target, map_location="cpu", weights_only=True)
    assert actual.keys() == expected.keys()
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key])


def test_cpu_benchmark_restores_eager_encoder_and_threads_on_compile_failure(small_run, monkeypatch):
    from examples.finetuning.cpu import benchmark_cpu

    model, tokenizer, public = load_pilot_model(small_run.parent / "pretrained/config.json", torch.device("cpu"))
    original_encoder = model.encoder
    original_threads = torch.get_num_threads()

    class UnavailableCompiler(torch.nn.Module):
        def forward(self, *args, **kwargs):
            raise RuntimeError("compiler unavailable")

    monkeypatch.setattr(torch, "compile", lambda *args, **kwargs: UnavailableCompiler())
    with pytest.raises(RuntimeError, match="compiler unavailable"):
        benchmark_cpu(model, tokenizer, public, small_run.parent / "valid.jsonl", max_length=8,
                      compile_encoder=True, num_threads=1, repeats=1)
    assert model.encoder is original_encoder
    assert not model.cpu_channels_last
    assert torch.get_num_threads() == original_threads
    assert not model._positional_encoding_cache
