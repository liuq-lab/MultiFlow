import sys
import types

import numpy as np
import pytest
import torch
from scipy import sparse

from multiflow_omics.autoencoder import (
    AutoencoderConfig,
    MultimodalAutoencoder,
    config_from_state_dict,
    encode_latents,
    fit_autoencoder,
    latent_scale,
    load_autoencoder,
    negative_binomial_log_prob,
    save_autoencoder,
)
from multiflow_omics.metrics import log1p_cp10k


def _profiles(n_cells: int = 96, n_genes: int = 30, n_peaks: int = 50, seed: int = 0):
    rng = np.random.default_rng(seed)
    group = rng.integers(0, 2, size=n_cells)
    rate = np.where(group[:, None] == 0, 0.5, 3.0) * rng.uniform(0.5, 1.5, size=(1, n_genes))
    rna = sparse.csr_matrix(rng.poisson(rate).astype(np.float32))
    prob = np.where(group[:, None] == 0, 0.05, 0.4) * np.ones((1, n_peaks))
    atac = sparse.csr_matrix(rng.binomial(1, prob).astype(np.float32))
    return rna, atac


def test_parameter_names_follow_the_scdiffusionx_checkpoint_layout() -> None:
    model = MultimodalAutoencoder(30, 50, rna_dims=(16, 8, 4), atac_dims=(24, 12, 4))
    keys = list(model.state_dict())
    assert keys[0] == "theta"
    assert "encoder.rna.net.0.0.weight" in keys
    assert "encoder.rna.net.0.1.running_mean" in keys
    assert "decoder.atac.net.2.bias" in keys
    assert model.state_dict()["encoder.atac.net.0.0.weight"].shape == (24, 50)
    assert model.state_dict()["decoder.rna.net.2.weight"].shape == (30, 16)


def test_architecture_is_recovered_from_parameter_shapes() -> None:
    model = MultimodalAutoencoder(30, 50, rna_dims=(16, 8, 4), atac_dims=(24, 12, 6))
    assert config_from_state_dict(model.state_dict()) == model.get_config()


def test_negative_binomial_matches_torch_distribution() -> None:
    x = torch.tensor([0.0, 1.0, 7.0, 40.0]).view(-1, 1)
    mu = torch.tensor([0.3, 2.0, 15.0]).view(1, -1)
    theta = torch.tensor(4.0)
    reference = torch.distributions.NegativeBinomial(
        total_count=theta, probs=mu / (mu + theta)
    ).log_prob(x)
    assert torch.allclose(negative_binomial_log_prob(x, mu, theta), reference, atol=1e-4)


def test_decoders_return_counts_and_probabilities() -> None:
    model = MultimodalAutoencoder(30, 50, rna_dims=(16, 8, 4), atac_dims=(24, 12, 4)).eval()
    rna, atac = _profiles(8)
    x, y = torch.from_numpy(rna.toarray()), torch.from_numpy(atac.toarray())
    with torch.no_grad():
        z_rna, z_atac = model.encode(x, y)
        library = x.sum(dim=1, keepdim=True)
        mean = model.decode_rna(z_rna, library)
        prob = model.decode_atac(z_atac)
    assert z_rna.shape == (8, 4) and z_atac.shape == (8, 4)
    assert torch.allclose(mean.sum(dim=1, keepdim=True), library, rtol=1e-4)
    assert float(prob.min()) >= 0 and float(prob.max()) <= 1


def test_sampled_rna_counts_follow_the_decoder_negative_binomial() -> None:
    torch.manual_seed(0)
    model = MultimodalAutoencoder(30, 50, rna_dims=(16, 8, 4), atac_dims=(24, 12, 4)).eval()
    z = torch.randn(1, 4).repeat(20000, 1)
    library = torch.full((20000, 1), 300.0)
    with torch.no_grad():
        mean = model.decode_rna(z[:1], library[:1])[0]
        counts = model.sample_rna(z, library)
        theta = torch.exp(model.theta)
    assert torch.equal(counts, counts.round()) and (counts >= 0).all()
    assert torch.allclose(counts.mean(0), mean, rtol=0.1, atol=0.05)
    variance = mean + mean**2 / theta  # negative binomial: more spread than Poisson
    assert torch.allclose(counts.var(0), variance, rtol=0.25, atol=0.1)


def test_training_lowers_the_loss_and_is_reproducible() -> None:
    rna, atac = _profiles()
    config = AutoencoderConfig(
        epochs=15, batch_size=16, rna_dims=(16, 8, 4), atac_dims=(24, 12, 4), seed=3, device="cpu"
    )
    first = fit_autoencoder(rna, atac, config=config, verbose=False)
    second = fit_autoencoder(rna, atac, config=config, verbose=False)
    assert first.history[-1]["train_loss"] < first.history[0]["train_loss"]
    assert "valid_loss" in first.history[-1]
    assert not first.model.training
    for key, value in first.model.state_dict().items():
        assert torch.equal(value, second.model.state_dict()[key])


def test_checkpoint_round_trip_and_latent_scale(tmp_path) -> None:
    rna, atac = _profiles()
    model = MultimodalAutoencoder(30, 50, rna_dims=(16, 8, 4), atac_dims=(24, 12, 4))
    path = save_autoencoder(tmp_path / "autoencoder.pt", model, metadata={"note": "test"})
    loaded = load_autoencoder(path)
    assert not loaded.training
    z_rna, z_atac = encode_latents(loaded, rna, atac, batch_size=10)
    expected_rna, expected_atac = encode_latents(model, rna, atac, batch_size=96)
    assert np.allclose(z_rna, expected_rna, atol=1e-5)
    assert np.allclose(z_atac, expected_atac, atol=1e-5)
    scale = latent_scale(z_rna)
    assert np.isclose(scale, 10 * z_rna.std(axis=0, ddof=1).mean(), rtol=1e-5)


def test_lightning_checkpoint_loads_without_its_training_packages(tmp_path) -> None:
    model = MultimodalAutoencoder(30, 50, rna_dims=(16, 8, 4), atac_dims=(24, 12, 4))
    module = types.ModuleType("package_that_is_not_installed")
    exec("class DictConfig(dict):\n    pass\n", module.__dict__)
    sys.modules[module.__name__] = module
    try:
        path = tmp_path / "last.ckpt"
        torch.save(
            {
                "epoch": 299,
                "state_dict": model.state_dict(),
                "hyper_parameters": module.DictConfig(a=1),
            },
            path,
        )
    finally:
        del sys.modules[module.__name__]
    loaded = load_autoencoder(path)
    assert loaded.get_config() == model.get_config()
    for key, value in model.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[key])


def test_missing_state_dict_is_rejected(tmp_path) -> None:
    path = tmp_path / "wrong.pt"
    torch.save({"weights": 1}, path)
    with pytest.raises(ValueError, match="not an autoencoder checkpoint"):
        load_autoencoder(path)


def test_log1p_cp10k_matches_scanpy() -> None:
    sc = pytest.importorskip("scanpy")
    ad = pytest.importorskip("anndata")
    rna, _ = _profiles(20)
    dense = rna.toarray()
    dense[3] = 0  # a cell without counts
    adata = ad.AnnData(sparse.csr_matrix(dense))
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    assert np.allclose(log1p_cp10k(sparse.csr_matrix(dense)), adata.X.toarray(), atol=1e-5)
