from __future__ import annotations

from pathlib import Path

import pytest
import torch

from multiflow import (
    CellStateFlow,
    ConditionalConcatFlow,
    LatentStandardizer,
    LegacyCheckpointError,
    PerturbationFlow,
    load_checkpoint,
    load_legacy_checkpoint,
    migrate_legacy_checkpoint,
)


class UnsafeLegacyMarker:
    pass


def _reverse_cell_key(key: str) -> str:
    output = key
    replacements = (
        ("label_embedding.", "label_emb."),
        ("middle_cross.", "mid_cross."),
        ("middle_residual.", "mid_res."),
        (".condition_projection.", ".emb_proj."),
        (".to_rna.", ".trans_rna."),
        (".to_atac.", ".trans_atac."),
    )
    for new, old in replacements:
        if new.startswith("."):
            output = output.replace(new, old)
        elif output.startswith(new):
            output = old + output[len(new) :]
    return output


def _reverse_perturbation_key(key: str) -> str:
    output = _reverse_cell_key(key)
    for new, old in (
        ("context_embedding.", "context_emb."),
        ("perturbation_embedding.", "pert_emb."),
        ("time_norm.", "time_ln."),
        ("context_norm.", "context_ln."),
        ("perturbation_norm.", "pert_ln."),
    ):
        if output.startswith(new):
            output = old + output[len(new) :]
    return output


def _reverse_concat_key(key: str) -> str:
    output = key
    for new, old in (
        ("fusion.", "fusion_linear."),
        ("time_mlp.", "time_embedding.time_embed."),
        ("input_block.condition_projection.", "input_block.emb_layer."),
        ("down.", "down_layers."),
        ("up.", "up_layers."),
        ("output.0.", "out1."),
        ("output.1.", "out_norm."),
        ("output.3.", "out2."),
        ("output.4.", "output_linear."),
    ):
        if output.startswith(new):
            output = old + output[len(new) :]
    return output


def _legacy_state(model: torch.nn.Module, rename) -> dict[str, torch.Tensor]:
    return {rename(key): value.detach().clone() for key, value in model.state_dict().items()}


def _assert_forward_parity(
    expected: torch.nn.Module,
    observed: torch.nn.Module,
    *,
    label: torch.Tensor | None,
    perturbation: torch.Tensor | None = None,
) -> None:
    generator = torch.Generator().manual_seed(42)
    batch = 4
    rna = torch.randn(batch, expected.rna_dim, generator=generator)
    atac = torch.randn(batch, expected.atac_dim, generator=generator)
    time = torch.rand(batch, generator=generator)
    expected.eval()
    observed.eval()
    with torch.no_grad():
        expected_output = expected(
            rna,
            atac,
            time,
            label=label,
            pert_label=perturbation,
        )
        observed_output = observed(
            rna,
            atac,
            time,
            label=label,
            pert_label=perturbation,
        )
    torch.testing.assert_close(observed_output[0], expected_output[0], rtol=0, atol=0)
    torch.testing.assert_close(observed_output[1], expected_output[1], rtol=0, atol=0)


def test_raw_cell_state_dict_migrates_with_exact_forward_parity(tmp_path):
    reference = CellStateFlow(
        3,
        5,
        hidden_dim=8,
        num_blocks=1,
        num_classes=2,
        cross_attention_dim=4,
    )
    source = tmp_path / "legacy-cell.pt"
    destination = tmp_path / "current-cell.pt"
    torch.save(_legacy_state(reference, _reverse_cell_key), source)

    migrate_legacy_checkpoint(source, destination)
    migrated, standardizer, payload = load_checkpoint(destination)

    assert standardizer is None
    assert payload["metadata"]["legacy_migration"]["container"] == "raw_state_dict"
    _assert_forward_parity(
        reference,
        migrated,
        label=torch.tensor([0, 1, 0, 1]),
    )


def test_format_v3_cell_checkpoint_loads_in_safe_mode(tmp_path):
    reference = CellStateFlow(
        3,
        5,
        hidden_dim=8,
        num_blocks=1,
        num_classes=2,
        cross_attention_dim=4,
    )
    source = tmp_path / "legacy-v3-cell.pt"
    torch.save(
        {
            "format_version": 3,
            "epoch": 600,
            "model_config": {
                "model_variant": "notebook_cross_attention_old_encoders_v1",
                "rna_dim": 3,
                "atac_dim": 5,
                "hidden_dim": 8,
                "num_blocks": 1,
                "num_classes": 2,
                "cross_attn_feature_dim": 4,
                "shared_base": False,
            },
            "model_state": _legacy_state(reference, _reverse_cell_key),
            "label_classes": ["B", "T"],
        },
        source,
    )

    migrated, standardizer, info = load_legacy_checkpoint(source)

    assert standardizer is None
    assert info["container"] == "format_v3"
    assert info["epoch"] == 600
    assert info["label_classes"] == ["B", "T"]
    _assert_forward_parity(
        reference,
        migrated,
        label=torch.tensor([0, 1, 0, 1]),
    )


def test_format_v5_concat_migrates_stats_and_exact_forward(tmp_path):
    reference = ConditionalConcatFlow(
        4,
        4,
        hidden_dim=8,
        num_layers=2,
        num_classes=2,
        dropout=0.0,
    )
    standardizer = LatentStandardizer(
        rna_mean=torch.arange(4).float().unsqueeze(0),
        rna_std=torch.arange(4).float().add(1).unsqueeze(0),
        atac_mean=torch.arange(4).float().add(10).unsqueeze(0),
        atac_std=torch.arange(4).float().add(2).unsqueeze(0),
    )
    source = tmp_path / "legacy-concat.pt"
    destination = tmp_path / "current-concat.pt"
    torch.save(
        {
            "format_version": 5,
            "completed_epochs": 17,
            "model_config": {
                "model_variant": "conditional_concat_corrected_v1",
                "rna_dim": 4,
                "atac_dim": 4,
                "hidden_dim": 8,
                "num_layers": 2,
                "num_classes": 2,
                "dropout": 0.0,
                "shared_base": True,
            },
            "model_state": _legacy_state(reference, _reverse_concat_key),
            "label_classes": ["B", "T"],
            "latent_stats": standardizer.state_dict(),
        },
        source,
    )

    migrate_legacy_checkpoint(source, destination)
    migrated, restored_standardizer, payload = load_checkpoint(destination)

    assert payload["epoch"] == 17
    assert restored_standardizer is not None
    for name, expected in standardizer.state_dict().items():
        torch.testing.assert_close(getattr(restored_standardizer, name), expected)
    _assert_forward_parity(
        reference,
        migrated,
        label=torch.tensor([0, 1, 0, 1]),
    )


def test_perturbation_payload_migrates_with_exact_forward_parity(tmp_path):
    context = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    reference = PerturbationFlow(
        3,
        5,
        context,
        hidden_dim=8,
        num_blocks=1,
        cross_attention_dim=4,
        num_perturbations=3,
    )
    source = tmp_path / "legacy-perturbation.pt"
    destination = tmp_path / "current-perturbation.pt"
    torch.save(
        {
            "epoch": 30,
            "model_state_dict": _legacy_state(reference, _reverse_perturbation_key),
            "context_matrix": context,
            "model_kwargs": {
                "rna_dim": 3,
                "atac_dim": 5,
                "hidden_dim": 8,
                "num_blocks": 1,
                "cross_attn_feature_dim": 4,
                "freeze_context": True,
                "context_scale": 1.0,
                "num_perturbations": 3,
                "perturbation_scale": 1.0,
            },
        },
        source,
    )

    migrate_legacy_checkpoint(source, destination)
    migrated, _, _ = load_checkpoint(destination)

    _assert_forward_parity(
        reference,
        migrated,
        label=torch.tensor([0, 1, 0, 1]),
        perturbation=torch.tensor([0, 1, 2, 0]),
    )


def test_unsafe_pickle_fallback_requires_explicit_trust(tmp_path):
    reference = CellStateFlow(
        2,
        2,
        hidden_dim=4,
        num_blocks=1,
        cross_attention_dim=2,
    )
    source = tmp_path / "unsafe-legacy.pt"
    torch.save(
        {
            "format_version": 3,
            "model_config": {
                "model_variant": "notebook_cross_attention_old_encoders_v1",
                "rna_dim": 2,
                "atac_dim": 2,
                "hidden_dim": 4,
                "num_blocks": 1,
                "cross_attn_feature_dim": 2,
            },
            "model_state": _legacy_state(reference, _reverse_cell_key),
            "unsafe_marker": UnsafeLegacyMarker(),
        },
        source,
    )

    with pytest.raises(LegacyCheckpointError, match="trust_source=True"):
        load_legacy_checkpoint(source)

    with pytest.warns(RuntimeWarning, match="weights_only=False"):
        migrated, _, _ = load_legacy_checkpoint(source, trust_source=True)
    _assert_forward_parity(reference, migrated, label=None)


def test_migration_metadata_does_not_store_absolute_source_path(tmp_path):
    reference = CellStateFlow(2, 2, hidden_dim=4, num_blocks=1)
    source = tmp_path / "legacy.pt"
    destination = tmp_path / "current.pt"
    torch.save(_legacy_state(reference, _reverse_cell_key), source)

    migrate_legacy_checkpoint(source, destination)
    _, _, payload = load_checkpoint(destination)

    migration = payload["metadata"]["legacy_migration"]
    assert migration["source_name"] == source.name
    assert str(Path(tmp_path)) not in str(migration)
