"""Public API for MultiFlow."""

from . import metrics
from ._version import __version__
from .api import (
    debias_perturbation,
    decode,
    encode,
    generate,
    predict_perturbations,
    prepare_perturbation_fold,
    split_h5mu,
    train,
    train_autoencoder,
    train_rna_vae,
)
from .autoencoder import (
    AutoencoderConfig,
    MultimodalAutoencoder,
    fit_autoencoder,
    load_autoencoder,
    save_autoencoder,
)
from .checkpoint import load_checkpoint, save_checkpoint
from .datasets import download_dataset
from .h5mu import (
    read_obs,
    read_paired_latents,
    simulate_h5mu,
    validate_h5mu,
    write_generated_h5mu,
    write_toy_h5mu,
)
from .legacy import (
    LegacyCheckpointError,
    load_legacy_checkpoint,
    migrate_legacy_checkpoint,
)
from .models import (
    BidirectionalCrossAttention,
    CellStateFlow,
    ConditionalConcatFlow,
    PerturbationFlow,
    build_model,
)
from .normalization import LatentStandardizer
from .sampling import sample_paired_latents
from .training import TrainingConfig, TrainingResult, fit, seed_everything

# Concise public names.  The longer class names remain available so published
# checkpoints and ablations are unambiguous.
MultiFlow = CellStateFlow
MultiFlowPerturbation = PerturbationFlow
MultiFlowConcat = ConditionalConcatFlow

__all__ = [
    "__version__",
    "AutoencoderConfig",
    "BidirectionalCrossAttention",
    "CellStateFlow",
    "ConditionalConcatFlow",
    "LatentStandardizer",
    "LegacyCheckpointError",
    "MultiFlow",
    "MultiFlowConcat",
    "MultiFlowPerturbation",
    "MultimodalAutoencoder",
    "PerturbationFlow",
    "TrainingConfig",
    "TrainingResult",
    "build_model",
    "debias_perturbation",
    "decode",
    "download_dataset",
    "encode",
    "fit",
    "fit_autoencoder",
    "generate",
    "load_autoencoder",
    "load_checkpoint",
    "load_legacy_checkpoint",
    "metrics",
    "migrate_legacy_checkpoint",
    "predict_perturbations",
    "prepare_perturbation_fold",
    "read_obs",
    "read_paired_latents",
    "sample_paired_latents",
    "save_autoencoder",
    "save_checkpoint",
    "seed_everything",
    "simulate_h5mu",
    "split_h5mu",
    "train",
    "train_autoencoder",
    "train_rna_vae",
    "validate_h5mu",
    "write_generated_h5mu",
    "write_toy_h5mu",
]
