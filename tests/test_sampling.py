from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn as nn

from multiflow import LatentStandardizer, sample_paired_latents


class ZeroVelocity(nn.Module):
    def __init__(self, rna_dim: int, atac_dim: int, *, shared: bool) -> None:
        super().__init__()
        self.rna_dim = rna_dim
        self.atac_dim = atac_dim
        self.requires_shared_base = shared

    def forward(self, rna, atac, time, label=None, pert_label=None):
        del time, label, pert_label
        return torch.zeros_like(rna), torch.zeros_like(atac)


def test_sampling_is_repeatable_for_a_fixed_seed():
    model = ZeroVelocity(3, 5, shared=False)
    first = sample_paired_latents(model, 7, steps=2, batch_size=3, device="cpu", seed=17)
    second = sample_paired_latents(model, 7, steps=2, batch_size=3, device="cpu", seed=17)
    torch.testing.assert_close(first[0], second[0])
    torch.testing.assert_close(first[1], second[1])


def test_default_joint_source_does_not_replicate_modality_blocks():
    rna, atac = sample_paired_latents(
        ZeroVelocity(4, 4, shared=False),
        8,
        steps=1,
        batch_size=8,
        device="cpu",
        seed=23,
    )
    assert not torch.equal(rna, atac)


def test_seeded_sampling_is_independent_of_batch_size():
    model = ZeroVelocity(3, 5, shared=False)
    small_batches = sample_paired_latents(
        model,
        7,
        steps=1,
        batch_size=2,
        device="cpu",
        seed=19,
        rng_mode="batch_invariant",
    )
    one_batch = sample_paired_latents(
        model,
        7,
        steps=1,
        batch_size=7,
        device="cpu",
        seed=19,
        rng_mode="batch_invariant",
    )
    torch.testing.assert_close(small_batches[0], one_batch[0])
    torch.testing.assert_close(small_batches[1], one_batch[1])


def test_explicit_legacy_mode_remains_repeatable():
    model = ZeroVelocity(3, 5, shared=False)
    first = sample_paired_latents(
        model,
        7,
        steps=1,
        batch_size=3,
        device="cpu",
        seed=29,
        rng_mode="legacy_interleaved",
    )
    second = sample_paired_latents(
        model,
        7,
        steps=1,
        batch_size=3,
        device="cpu",
        seed=29,
        rng_mode="legacy_interleaved",
    )
    torch.testing.assert_close(first[0], second[0])
    torch.testing.assert_close(first[1], second[1])


def test_sampling_does_not_emit_tensor_copy_warnings():
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        sample_paired_latents(
            ZeroVelocity(2, 3, shared=False),
            2,
            steps=1,
            device="cpu",
            seed=1,
        )
    assert not recorded


def test_shared_source_noise_is_identical_for_zero_velocity():
    rna, atac = sample_paired_latents(
        ZeroVelocity(4, 4, shared=True),
        6,
        steps=1,
        batch_size=2,
        device="cpu",
        seed=3,
    )
    torch.testing.assert_close(rna, atac)


def test_sampling_applies_inverse_standardization():
    standardizer = LatentStandardizer(
        rna_mean=torch.full((1, 2), 10.0),
        rna_std=torch.full((1, 2), 2.0),
        atac_mean=torch.full((1, 3), -5.0),
        atac_std=torch.full((1, 3), 4.0),
    )
    raw = sample_paired_latents(
        ZeroVelocity(2, 3, shared=False),
        5,
        steps=1,
        device="cpu",
        seed=11,
    )
    restored = sample_paired_latents(
        ZeroVelocity(2, 3, shared=False),
        5,
        steps=1,
        device="cpu",
        seed=11,
        standardizer=standardizer,
    )
    expected = standardizer.inverse_transform(*raw)
    torch.testing.assert_close(restored[0], expected[0])
    torch.testing.assert_close(restored[1], expected[1])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n": 0},
        {"n": 1, "steps": 0},
        {"n": 1, "batch_size": 0},
    ],
)
def test_sampling_rejects_nonpositive_sizes(kwargs):
    with pytest.raises(ValueError, match="must be positive"):
        sample_paired_latents(ZeroVelocity(2, 2, shared=False), device="cpu", **kwargs)
