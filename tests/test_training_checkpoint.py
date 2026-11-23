from __future__ import annotations

import numpy as np
import pytest
import torch

from multiflow import (
    ConditionalConcatFlow,
    PerturbationFlow,
    TrainingConfig,
    fit,
    load_checkpoint,
    save_checkpoint,
)


def test_tiny_cpu_training_and_checkpoint_round_trip(tmp_path):
    generator = np.random.default_rng(5)
    rna = generator.normal(size=(6, 2)).astype(np.float32)
    atac = generator.normal(size=(6, 2)).astype(np.float32)
    labels = np.tile(np.arange(2), 3)
    model = ConditionalConcatFlow(
        2,
        2,
        hidden_dim=4,
        num_layers=2,
        num_classes=2,
        dropout=0.0,
    )

    result = fit(
        model,
        rna,
        atac,
        labels=labels,
        config=TrainingConfig(epochs=1, batch_size=6, device="cpu", seed=13),
    )

    assert len(result.history) == 1
    assert all(np.isfinite(row["loss_total"]) for row in result.history)
    checkpoint = save_checkpoint(
        tmp_path / "model.pt",
        result.model,
        standardizer=result.standardizer,
        optimizer=result.optimizer,
        epoch=1,
        history=result.history,
        metadata={"dataset": "synthetic"},
    )
    restored_model, restored_standardizer, payload = load_checkpoint(checkpoint)

    assert restored_model.get_config() == result.model.get_config()
    assert payload["format_version"] == 1
    assert payload["epoch"] == 1
    assert payload["metadata"] == {"dataset": "synthetic"}
    assert restored_standardizer is not None
    for key, expected in result.model.state_dict().items():
        torch.testing.assert_close(restored_model.state_dict()[key], expected.cpu())

    for name, expected in result.standardizer.state_dict().items():
        torch.testing.assert_close(getattr(restored_standardizer, name), expected)


def test_perturbation_checkpoint_preserves_context_embeddings(tmp_path):
    context = torch.arange(24, dtype=torch.float32).reshape(3, 8)
    model = PerturbationFlow(
        2,
        3,
        context,
        hidden_dim=8,
        num_blocks=1,
        num_perturbations=2,
    ).eval()
    checkpoint = save_checkpoint(tmp_path / "perturbation.pt", model)
    restored, _, _ = load_checkpoint(checkpoint)

    torch.testing.assert_close(restored.context_embedding.weight, context)
    assert restored.get_config() == model.get_config()


def test_checkpoint_never_writes_metadata_that_weights_only_loader_cannot_read(tmp_path):
    model = ConditionalConcatFlow(2, 2, hidden_dim=4, num_layers=2)
    checkpoint = tmp_path / "unsafe-metadata.pt"
    try:
        save_checkpoint(checkpoint, model, metadata={"local_path": tmp_path})
    except (TypeError, ValueError):
        return

    _, _, payload = load_checkpoint(checkpoint)
    assert isinstance(payload["metadata"]["local_path"], str)


def test_training_rejects_empty_latent_arrays_without_dividing_by_zero():
    model = ConditionalConcatFlow(2, 2, hidden_dim=4, num_layers=2)
    empty = np.empty((0, 2), dtype=np.float32)
    with np.testing.assert_raises_regex(ValueError, "cell|row|empty|at least"):
        fit(
            model,
            empty,
            empty,
            config=TrainingConfig(
                epochs=1,
                batch_size=2,
                device="cpu",
                standardize=False,
            ),
        )


@pytest.mark.parametrize(
    ("model", "noise_mode"),
    [
        (
            ConditionalConcatFlow(
                2,
                2,
                hidden_dim=4,
                num_layers=2,
                num_classes=1,
            ),
            "independent",
        ),
        (
            PerturbationFlow(
                2,
                2,
                torch.zeros(2, 4),
                hidden_dim=4,
                num_blocks=1,
                num_perturbations=1,
            ),
            "shared",
        ),
    ],
)
def test_training_rejects_noise_contract_that_sampling_cannot_reproduce(model, noise_mode):
    values = np.zeros((2, 2), dtype=np.float32)
    labels = np.zeros(2, dtype=np.int64)
    with pytest.raises(ValueError, match="requires noise_mode"):
        fit(
            model,
            values,
            values,
            labels=labels,
            config=TrainingConfig(
                epochs=1,
                batch_size=2,
                device="cpu",
                noise_mode=noise_mode,
            ),
        )
