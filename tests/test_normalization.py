from __future__ import annotations

import torch

from multiflow import LatentStandardizer


def test_standardizer_transform_inverse_round_trip():
    rna = torch.tensor([[1.0, 10.0], [3.0, 14.0], [5.0, 18.0]])
    atac = torch.tensor([[20.0, -2.0, 7.0], [24.0, 0.0, 9.0], [28.0, 2.0, 11.0]])
    standardizer = LatentStandardizer.fit(rna, atac)

    standardized_rna, standardized_atac = standardizer.transform(rna, atac)
    restored_rna, restored_atac = standardizer.inverse_transform(
        standardized_rna,
        standardized_atac,
    )

    torch.testing.assert_close(restored_rna, rna)
    torch.testing.assert_close(restored_atac, atac)


def test_standardizer_state_dict_round_trip_preserves_modalities():
    standardizer = LatentStandardizer(
        rna_mean=torch.tensor([[1.0, 2.0]]),
        rna_std=torch.tensor([[3.0, 4.0]]),
        atac_mean=torch.tensor([[5.0, 6.0, 7.0]]),
        atac_std=torch.tensor([[8.0, 9.0, 10.0]]),
    )

    restored = LatentStandardizer.from_state_dict(standardizer.state_dict())

    for name, expected in standardizer.state_dict().items():
        torch.testing.assert_close(getattr(restored, name), expected)
