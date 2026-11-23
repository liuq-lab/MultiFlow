from __future__ import annotations

import re

import pytest
import torch

import multiflow
from multiflow import (
    BidirectionalCrossAttention,
    CellStateFlow,
    ConditionalConcatFlow,
    PerturbationFlow,
    build_model,
)


@pytest.mark.parametrize(
    ("model", "label", "perturbation"),
    [
        (CellStateFlow(3, 5, hidden_dim=8, num_blocks=1), None, None),
        (CellStateFlow(3, 5, hidden_dim=8, num_blocks=1, num_classes=2), 1, None),
        (
            PerturbationFlow(
                3,
                5,
                torch.arange(16, dtype=torch.float32).reshape(2, 8),
                hidden_dim=8,
                num_blocks=1,
                num_perturbations=3,
            ),
            1,
            2,
        ),
        (ConditionalConcatFlow(4, 4, hidden_dim=8, num_layers=2, num_classes=2), 1, None),
    ],
)
def test_model_forward_is_finite_and_configuration_round_trips(model, label, perturbation):
    batch = 4
    rna = torch.randn(batch, model.rna_dim)
    atac = torch.randn(batch, model.atac_dim)
    time = torch.linspace(0, 1, batch)
    labels = None if label is None else torch.full((batch,), label)
    perturbations = None if perturbation is None else torch.full((batch,), perturbation)

    output_rna, output_atac = model(
        rna,
        atac,
        time,
        label=labels,
        pert_label=perturbations,
    )

    assert output_rna.shape == rna.shape
    assert output_atac.shape == atac.shape
    assert torch.isfinite(output_rna).all()
    assert torch.isfinite(output_atac).all()

    rebuilt = build_model(model.get_config())
    assert type(rebuilt) is type(model)
    assert rebuilt.get_config() == model.get_config()


def test_conditioned_model_requires_a_label():
    model = CellStateFlow(2, 2, hidden_dim=4, num_blocks=1, num_classes=2)
    with pytest.raises(ValueError, match="label is required"):
        model(torch.randn(3, 2), torch.randn(3, 2), torch.rand(3))


def test_bidirectional_cross_attention_rejects_non_singleton_tokens_cleanly():
    module = BidirectionalCrossAttention(channels=4, attention_dim=3)
    rna = torch.randn(2, 5, 4)
    atac = torch.randn(2, 5, 4)
    with pytest.raises(ValueError, match="token|singleton"):
        module(rna, atac)


def test_public_version_and_exports_are_available():
    assert re.fullmatch(r"\d+\.\d+\.\d+(?:[a-z]+\d+)?", multiflow.__version__)
    assert multiflow.CellStateFlow is CellStateFlow


def test_perturbation_context_width_and_values_are_validated():
    with pytest.raises(ValueError, match="context width"):
        PerturbationFlow(
            3,
            4,
            torch.zeros(2, 2),
            hidden_dim=5,
            num_blocks=1,
        )
    invalid = torch.zeros(2, 4)
    invalid[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        PerturbationFlow(3, 4, invalid, hidden_dim=4, num_blocks=1)
