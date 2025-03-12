from typing import Callable
from sparse_autoencoder.loss.abstract_loss import AbstractLoss
import torch
import torch.nn as nn
from sparse_autoencoder.autoencoder.model import SparseAutoencoder
from sparse_autoencoder.autoencoder.model import AutoencoderForwardPassResult


class ReLUSparseAutoencoder(SparseAutoencoder):
    """Sparse Autoencoder with ReLU activation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.postact_fn = nn.ReLU()
    
    def forward(self, x):
        return super().forward(x), 0.0


class TopKSparseAutoencoder(SparseAutoencoder):
    """Sparse Autoencoder with TopK activation."""

    def __init__(
        self,
        n_input_features: int,
        n_learned_features: int,
        k: int,
        aux_k: int,
        geometric_median_dataset = None,
        n_components: int | None = None,
        postact_fn: Callable = nn.ReLU(),
        max_dead_steps: int = 10,
        aux_loss_weight: float = 1 / 32,
    ) -> None:
        """Initialize the TopK Sparse Autoencoder Model.

        Args:
            n_input_features: Number of input features
            n_learned_features: Number of learned features
            k: Number of top activations to keep
            geometric_median_dataset: Estimated geometric median of the dataset
            n_components: Number of source model components
            postact_fn: Activation function to apply after top-k selection
        """
        super().__init__(
            n_input_features=n_input_features,
            n_learned_features=n_learned_features,
            geometric_median_dataset=geometric_median_dataset,
            n_components=n_components,
        )
        self.k = k
        self.aux_k = aux_k
        self.postact_fn = postact_fn
        self.record_last_nonzero = torch.zeros(n_components, n_learned_features)
        self.max_dead_steps = max_dead_steps
        self.aux_loss_weight = aux_loss_weight
        # Initialize the decoder to be transposed of the encoder with unit norm
        self.decoder.weight.data = self.encoder.weight.data.transpose(-1, -2).clone()
        self.decoder.constrain_weights_unit_norm()


    def forward(
        self,
        x,
    ):
        """Forward Pass with TopK activation.

        Args:
            x: Input activations

        Returns:
            AutoencoderForwardPassResult containing learned and decoded activations
        """
        x = self.pre_encoder_bias(x)
        pre_activations = self.encoder(x)
        
        # Apply top-k activation
        topk_values, topk_indices = torch.topk(pre_activations, k=self.k, dim=-1)
        values = self.postact_fn(topk_values)
        
        # Create sparse activation tensor
        learned_activations = torch.zeros_like(pre_activations)
        learned_activations.scatter(-1, topk_indices, values)
        print(f"learned_activations.shape: {learned_activations.shape}, decoder.weight.shape: {self.decoder.weight.shape}")
        x = self.decoder(learned_activations)
        decoded_activations = self.post_decoder_bias(x)
        aux_loss = self.aux_loss(x, pre_activations, learned_activations, decoded_activations) * self.aux_loss_weight
        return AutoencoderForwardPassResult(learned_activations, decoded_activations), aux_loss

    def aux_loss(self,
        source_activations,
        pre_activations,
        learned_activations,
        decoded_activations):
        nonzero_features = torch.sum((learned_activations.abs() > 1e-3), dim=0) > 0
        self.record_last_nonzero += 1
        self.record_last_nonzero[nonzero_features] = 0
        aux_indices = self.record_last_nonzero >= self.max_dead_steps
        aux_indices = aux_indices.to(pre_activations.device)
        pre_activations *= aux_indices
        topk_aux_values, topk_aux_indices = torch.topk(pre_activations, k=self.aux_k, dim=-1)
        topk_aux_values = self.postact_fn(topk_aux_values)
        aux_activations = torch.zeros_like(pre_activations)
        aux_activations.scatter(-1, topk_aux_indices, topk_aux_values)
        decoded_aux_activations = self.decoder(aux_activations)
        decoded_aux_activations = self.post_decoder_bias(decoded_aux_activations)
        target = source_activations - decoded_activations.detach()
        return normalized_mse(decoded_aux_activations, target)


def normalized_mse(recon, target):
    target_mu = target.mean(dim=0, keepdim=True)
    loss = torch.nn.functional.mse_loss(recon, target) / torch.nn.functional.mse_loss(target_mu, target)
    return loss.nan_to_num(0)