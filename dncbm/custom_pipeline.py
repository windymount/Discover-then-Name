from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import quote_plus

from jaxtyping import Int64
from pydantic import NonNegativeInt, PositiveInt, validate_call
from sparse_autoencoder.activation_store.base_store import ActivationStore
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb
from sparse_autoencoder.metrics.abstract_metric import ComponentAggregationApproach, MetricLocation, MetricResult
from sparse_autoencoder.activation_resampler.activation_resampler import ActivationResampler

from sparse_autoencoder.activation_resampler.abstract_activation_resampler import (
    AbstractActivationResampler,
    ParameterUpdateResults,
)

from time import time


from sparse_autoencoder.activation_store.tensor_store import TensorActivationStore
from sparse_autoencoder.autoencoder.model import SparseAutoencoder
from sparse_autoencoder.loss.abstract_loss import AbstractLoss, LossReductionType
from sparse_autoencoder.metrics.metrics_container import MetricsContainer, default_metrics
from sparse_autoencoder.metrics.train.abstract_train_metric import TrainMetricData
from sparse_autoencoder.optimizer.abstract_optimizer import AbstractOptimizerWithReset
from sparse_autoencoder.tensor_types import Axis
from enum import Enum


if TYPE_CHECKING:
    from sparse_autoencoder.metrics.abstract_metric import MetricResult


class AlignmentLossType(Enum):
    maha_encoder = "mahalanobis_encoder"
    maha_decoder = "mahalanobis_decoder"
    max_cos_sim_encoder = "max_cos_sim_encoder"
    max_cos_sim_decoder = "max_cos_sim_decoder"


class Pipeline:
    """Pipeline for training a Sparse Autoencoder on TransformerLens activations.

    Includes all the key functionality to train a sparse autoencoder, with a specific set of
        hyperparameters.
    """

    optimizer: AbstractOptimizerWithReset
    """Optimizer to use."""

    progress_bar: tqdm | None
    """Progress bar for the pipeline."""

    total_activations_trained_on: int = 0
    """Total number of activations trained on state."""

    @property
    def n_components(self) -> int:
        """Number of source model components the SAE is trained on."""

        return 1  # since we are training on a single component, which is the out layer

    def __init__(
        self,
        activation_resampler: AbstractActivationResampler | None,
        autoencoder: SparseAutoencoder,
        loss: AbstractLoss,
        optimizer: AbstractOptimizerWithReset,
        checkpoint_directory: Path = None,
        log_frequency: PositiveInt = 100,
        metrics: MetricsContainer = default_metrics,
        device: torch.cuda = 'cuda',
        args=None
    ) -> None:

        self.activation_resampler = activation_resampler
        self.autoencoder = autoencoder
        self.checkpoint_directory = checkpoint_directory
        self.log_frequency = log_frequency
        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer
        self.device = device
        self.args = args

    @validate_call(config={"arbitrary_types_allowed": True})
    def train_autoencoder(
        self, activation_store: TensorActivationStore, train_batch_size: PositiveInt
    ) -> Int64[Tensor, Axis.names(Axis.COMPONENT, Axis.LEARNT_FEATURE)]:
        """Train the sparse autoencoder.

        Args:
            activation_store: Activation store from the generate section.
            train_batch_size: Train batch size.

        Returns:
            Number of times each neuron fired, for each component.
        """

        activations_dataloader = DataLoader(
            activation_store,
            batch_size=train_batch_size,
            shuffle=True
        )

        learned_activations_fired_count: Int64[
            Tensor, Axis.names(Axis.COMPONENT, Axis.LEARNT_FEATURE)
        ] = torch.zeros(
            (self.n_components, self.autoencoder.n_learned_features),
            dtype=torch.int64,
            device=self.device,)

        for id, store_batch in enumerate(activations_dataloader):
            # Zero the gradients
            self.optimizer.zero_grad()

            # Move the batch to the device (in place)
            batch = store_batch.detach().to(self.device)

            # Forward pass
            learned_activations, reconstructed_activations = self.autoencoder.forward(
                batch)

            # Get loss & metrics
            metrics: list[MetricResult] = []
            total_loss, loss_metrics = self.loss.scalar_loss_with_log(
                batch,
                learned_activations,
                reconstructed_activations,
                component_reduction=LossReductionType.MEAN
            )
            metrics.extend(loss_metrics)

            with torch.no_grad():
                for metric in self.metrics.train_metrics:
                    calculated = metric.calculate(
                        TrainMetricData(batch, learned_activations,
                                        reconstructed_activations)
                    )
                    metrics.extend(calculated)

            # Store count of how many neurons have fired
            with torch.no_grad():
                fired = learned_activations > 0
                learned_activations_fired_count.add_(fired.sum(dim=0))

            # Backwards pass
            total_loss.backward()
            self.optimizer.step()
            self.autoencoder.post_backwards_hook()

            # Log training metrics
            self.total_activations_trained_on += train_batch_size
            if (
                wandb.run is not None
                and int(self.total_activations_trained_on / train_batch_size) % self.log_frequency
                == 0
            ):
                log = {}
                for metric_result in metrics:
                    log.update(metric_result.wandb_log)
                wandb.log(
                    log,
                    step=self.total_activations_trained_on,
                    commit=False,
                )
        return learned_activations_fired_count

    def save_checkpoint(self, *, is_final: bool = False) -> Path:
        """Save the model as a checkpoint.

        Args:
            is_final: Whether this is the final checkpoint.

        Returns:
            Path to the saved checkpoint.
        """
        # Create the name
        name: str = f"sparse_autoencoder_{'final' if is_final else self.total_activations_trained_on}"
        safe_name = quote_plus(name, safe="_")
        self.checkpoint_directory.mkdir(parents=True, exist_ok=True)
        file_path: Path = self.checkpoint_directory / f"{safe_name}.pt"

        torch.save(
            self.autoencoder.state_dict(),
            file_path,
        )
        return file_path

    def update_parameters(self, parameter_updates: list[ParameterUpdateResults]) -> None:
        """Update the parameters of the model from the results of the resampler.

        Args:
            parameter_updates: Parameter updates from the resampler.
        """
        for component_idx, component_parameter_update in enumerate(parameter_updates):
            # Update the weights and biases
            self.autoencoder.encoder.update_dictionary_vectors(
                component_parameter_update.dead_neuron_indices,
                component_parameter_update.dead_encoder_weight_updates,
                component_idx=component_idx,
            )
            self.autoencoder.encoder.update_bias(
                component_parameter_update.dead_neuron_indices,
                component_parameter_update.dead_encoder_bias_updates,
                component_idx=component_idx,
            )
            self.autoencoder.decoder.update_dictionary_vectors(
                component_parameter_update.dead_neuron_indices,
                component_parameter_update.dead_decoder_weight_updates,
                component_idx=component_idx,
            )

            # Reset the optimizer
            for parameter, axis in self.autoencoder.reset_optimizer_parameter_details:
                self.optimizer.reset_neurons_state(
                    parameter=parameter,
                    neuron_indices=component_parameter_update.dead_neuron_indices,
                    axis=axis,
                    component_idx=component_idx,
                )

    def get_activation_store(self, activation_fname):
        activations = torch.load(activation_fname)
        activation_store = TensorActivationStore(
            activations.shape[0], self.autoencoder.n_input_features, self.n_components)
        activation_store.empty()
        activation_store.extend(activations, component_idx=0)
        return activation_store

    # considering train_val_fnames to contain a single fname
    def validation(self, activation_store, train_batch_size):
        activations_dataloader = DataLoader(
            activation_store, batch_size=train_batch_size, shuffle=True)

        with torch.no_grad():
            total_losses = torch.zeros((4, len(activations_dataloader)))
            with tqdm(desc="Validation", total=len(activations_dataloader),) as progress_bar:
                for batch_id, store_batch in enumerate(activations_dataloader):
                    batch = store_batch.detach().to(self.device)
                    # Forward pass
                    learned_activations, reconstructed_activations = self.autoencoder.forward(
                        batch)
                    _, loss_metrics = self.loss.scalar_loss_with_log(
                        batch,
                        learned_activations,
                        reconstructed_activations,
                        component_reduction=LossReductionType.MEAN
                    )
                    for loss_id, loss_metric in enumerate(loss_metrics):
                        total_losses[loss_id,
                                     batch_id] = loss_metric.component_wise_values
                    mean_losses = total_losses.mean(dim=1)
                    progress_bar.update(1)
                return loss_metrics, mean_losses

    def run_pipeline(
        self,
        train_batch_size: PositiveInt,
        val_frequency: NonNegativeInt | None = None,
        checkpoint_frequency: NonNegativeInt | None = None,
        num_epochs=None,
        train_fnames=None,
        train_val_fnames=None,
        start_time=0,
        resample_epoch_freq: NonNegativeInt = 0,
    ) -> None:

        last_validated: int = 0
        last_checkpoint: int = 0

        assert (train_fnames is not None)
        num_train_pieces = len(train_fnames)
        train_order = torch.randperm(num_train_pieces)
        train_piece_idx = 0

        self.actual_epochs = num_epochs * num_train_pieces
        print(f"Piece init completed at {time() - start_time} seconds")

        with tqdm(
            desc="Activations trained on",
            total=self.actual_epochs,
        ) as progress_bar:

            self.progress_bar = progress_bar

            for epoch in range(self.actual_epochs):
                # if the train activations are saved in more than one piece, shuffle the order
                if train_piece_idx >= num_train_pieces:
                    train_order = torch.randperm(num_train_pieces)
                    train_piece_idx = 0

                train_activation_store = self.get_activation_store(
                    train_fnames[train_order[train_piece_idx]])
                print(
                    f"Activation store created at {time() - start_time} seconds")
                print(
                    f"{train_fnames[train_order[train_piece_idx]]}: {train_activation_store._data.shape[0]} NUM SAMPLES.")
                train_piece_idx += 1
                self.current_epoch = epoch

                # Update the counters
                n_activation_vectors_in_store = len(train_activation_store)
                last_validated += n_activation_vectors_in_store
                last_checkpoint += n_activation_vectors_in_store

                # Train
                progress_bar.set_postfix({"stage": "train"})
                batch_neuron_activity: Int64[Tensor, Axis.LEARNT_FEATURE] = self.train_autoencoder(
                    train_activation_store, train_batch_size=train_batch_size
                )

                print(f"Training completed at {time() - start_time} seconds")

                # Resample dead neurons (if needed)
                if (self.activation_resampler is not None) and ((epoch+resample_epoch_freq) < (self.actual_epochs-1)):
                    # Get the updates
                    parameter_updates = self.activation_resampler.step_resampler(
                        batch_neuron_activity=batch_neuron_activity,
                        activation_store=train_activation_store,
                        autoencoder=self.autoencoder,
                        loss_fn=self.loss,
                        train_batch_size=train_batch_size,
                    )

                    if parameter_updates is not None:
                        progress_bar.set_postfix({"stage": "resampling"})
                        print(f"Resampling at epoch: {epoch}")
                        if wandb.run is not None:
                            wandb.log(
                                {
                                    "resample/dead_neurons": [
                                        len(update.dead_neuron_indices)
                                        for update in parameter_updates
                                    ]
                                },
                                commit=False,
                            )
                        for id, update in enumerate(parameter_updates):
                            print(
                                f"component id: {id}; {len(update.dead_neuron_indices)}")
                        print(
                            "###########################################################")
                        # Update the parameters
                        self.update_parameters(parameter_updates)

                    print(
                        f"Resampling completed at {time() - start_time} seconds")
                else:
                    print(
                        f"Resampling skipped at {time() - start_time} seconds")

                del train_activation_store
                print(
                    f"Activation store deleted at {time() - start_time} seconds")

                # Get validation metrics (if needed)
                if val_frequency != 0 and last_validated >= val_frequency:
                    last_validated = 0
                    progress_bar.set_postfix({"stage": "validate"})
                    assert (train_val_fnames is not None)

                    total_mean_losses = torch.zeros((4, len(train_val_fnames)))
                    for id, train_val_fname in enumerate(train_val_fnames):
                        train_val_activation_store = self.get_activation_store(
                            train_val_fname)
                        print(
                            f"{train_val_fname}: {train_val_activation_store._data.shape[0]} NUM SAMPLES.")
                        loss_metrics, tmp_mean_losses = self.validation(train_val_activation_store,
                                                                        train_batch_size)

                        total_mean_losses[:, id] = tmp_mean_losses
                        del train_val_activation_store
                    mean_losses = total_mean_losses.mean(dim=1)

                    if wandb.run is not None:
                        log = {}
                        for id, metric in enumerate(loss_metrics):
                            metric.location = MetricLocation.VALIDATE
                            metric.component_wise_values = torch.tensor(
                                [mean_losses[id]])
                            # import pdb; pdb.set_trace()
                            log.update(metric.wandb_log)
                        wandb.log(
                            log, step=self.total_activations_trained_on, commit=True,)

                    print(
                        f"Validation completed at {time() - start_time} seconds")
                else:
                    print(
                        f"Validation skipped at {time() - start_time} seconds")

                # Checkpoint (if needed)
                if checkpoint_frequency != 0 and last_checkpoint >= checkpoint_frequency:
                    progress_bar.set_postfix({"stage": "checkpoint"})
                    last_checkpoint = 0
                    self.save_checkpoint()

                    print(
                        f"Checkpoint save completed at {time() - start_time} seconds")
                else:
                    print(
                        f"Checkpoint save skipped at {time() - start_time} seconds")

                # Update the progress bar
                progress_bar.update(1)

                print(
                    f"Epoch {epoch} completed at {time() - start_time} seconds")

        # Save the final checkpoint
        self.save_checkpoint(is_final=True)


class PipelineWithAlignment(Pipeline):
    def __init__(self, align_lambda, embd_dictionary, align_loss_type: AlignmentLossType = AlignmentLossType.maha_encoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.align_lambda = align_lambda
        # Choose alignment loss based on type
        if align_loss_type == AlignmentLossType.maha_encoder:
            self.alignment_loss = MahaLossOnEncoder(embd_dictionary)
        elif align_loss_type == AlignmentLossType.maha_decoder:
            self.alignment_loss = MahaLossOnDecoder(embd_dictionary)
        elif align_loss_type == AlignmentLossType.max_cos_sim_encoder:
            self.alignment_loss = MaxCosSimLossOnEncoder(embd_dictionary)
        elif align_loss_type == AlignmentLossType.max_cos_sim_decoder:
            self.alignment_loss = MaxCosSimLossOnDecoder(embd_dictionary)
        else:
            raise ValueError(f"Unknown alignment loss type: {align_loss_type}")

    @validate_call(config={"arbitrary_types_allowed": True})
    def train_autoencoder(
        self, activation_store: TensorActivationStore, train_batch_size: PositiveInt
    ) -> Int64[Tensor, Axis.names(Axis.COMPONENT, Axis.LEARNT_FEATURE)]:
        """Train the sparse autoencoder.

        Args:
            activation_store: Activation store from the generate section.
            train_batch_size: Train batch size.

        Returns:
            Number of times each neuron fired, for each component.
        """

        activations_dataloader = DataLoader(
            activation_store,
            batch_size=train_batch_size,
            shuffle=True
        )

        learned_activations_fired_count: Int64[
            Tensor, Axis.names(Axis.COMPONENT, Axis.LEARNT_FEATURE)
        ] = torch.zeros(
            (self.n_components, self.autoencoder.n_learned_features),
            dtype=torch.int64,
            device=self.device,)

        for id, store_batch in enumerate(activations_dataloader):
            # Zero the gradients
            self.optimizer.zero_grad()

            # Move the batch to the device (in place)
            batch = store_batch.detach().to(self.device)

            # Forward pass
            learned_activations, reconstructed_activations = self.autoencoder.forward(
                batch)

            # Get loss & metrics
            metrics: list[MetricResult] = []
            total_loss, loss_metrics = self.loss.scalar_loss_with_log(
                batch,
                learned_activations,
                reconstructed_activations,
                component_reduction=LossReductionType.MEAN
            )
            metrics.extend(loss_metrics)

            with torch.no_grad():
                for metric in self.metrics.train_metrics:
                    calculated = metric.calculate(
                        TrainMetricData(batch, learned_activations,
                                        reconstructed_activations)
                    )
                    metrics.extend(calculated)

            # Store count of how many neurons have fired
            with torch.no_grad():
                fired = learned_activations > 0
                learned_activations_fired_count.add_(fired.sum(dim=0))
            # Calculate alignment loss
            alignment_loss = self.alignment_loss.forward(self.autoencoder)
            total_loss += self.align_lambda * alignment_loss
            
            # Log alignment loss metric
            metrics.append(MetricResult(
                component_wise_values=[alignment_loss.item()],
                name="alignment_loss",
                location=MetricLocation.TRAIN,
                aggregate_approach=ComponentAggregationApproach.MEAN
            ))


            # Backwards pass
            total_loss.backward()
            self.optimizer.step()
            self.autoencoder.post_backwards_hook()

            # Log training metrics
            self.total_activations_trained_on += train_batch_size
            if (
                wandb.run is not None
                and int(self.total_activations_trained_on / train_batch_size) % self.log_frequency
                == 0
            ):
                log = {}
                for metric_result in metrics:
                    log.update(metric_result.wandb_log)
                wandb.log(
                    log,
                    step=self.total_activations_trained_on,
                    commit=False,
                )
        return learned_activations_fired_count


class MahalanobisAlignmentLoss():
    def __init__(self, embd_dictionary):
        self.embd_dictionary = embd_dictionary
        self.sigma_inv = torch.inverse(torch.cov(embd_dictionary.T) + 1e-6 * torch.eye(embd_dictionary.shape[1]).to(embd_dictionary.device))
        self.mu = torch.mean(embd_dictionary, dim=0)

    def forward(self, weights):
        weights = weights / weights.norm(dim=1, keepdim=True)
        weights = weights - self.mu.unsqueeze(0)
        return torch.diag(weights @ self.sigma_inv @ weights.T).mean()


class MahaLossOnEncoder(MahalanobisAlignmentLoss):
    def forward(self, sae):
        return super().forward(sae.encoder.weight.squeeze(0))


class MahaLossOnDecoder(MahalanobisAlignmentLoss):
    def forward(self, sae):
        return super().forward(sae.decoder.weight.squeeze(0).T)


class MaxCosSimLoss():
    def __init__(self, embd_dictionary):
        self.embd_dictionary = embd_dictionary
        self.embd_dictionary = self.embd_dictionary / self.embd_dictionary.norm(dim=1, keepdim=True)

    def forward(self, weights):
        weights = weights / weights.norm(dim=1, keepdim=True)
        cos_sim = torch.matmul(weights, self.embd_dictionary.T)
        return -cos_sim.max(dim=1).values.mean()


class MaxCosSimLossOnEncoder(MaxCosSimLoss):
    def forward(self, sae):
        return super().forward(sae.encoder.weight.squeeze(0))


class MaxCosSimLossOnDecoder(MaxCosSimLoss):
    def forward(self, sae):
        return super().forward(sae.decoder.weight.squeeze(0).T)


class EmbeddingAlignedResampler(ActivationResampler):
    """Activation resampler that considers embedding alignment."""
    
    def __init__(
        self,
        embd_dictionary: torch.Tensor,
        cosine_similarity_threshold: float = 0.8,
        use_encoder: bool = True,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.embd_dictionary = embd_dictionary
        self.cosine_similarity_threshold = cosine_similarity_threshold
        self.use_encoder = use_encoder
        
    def _get_dead_neuron_indices(self) -> list[Int64[Tensor, Axis.names(Axis.LEARNT_FEATURE_IDX)]]:
        """Get indices of neurons to resample.
        
        Combines standard dead neuron detection with detection of redundant embedding alignments.
        
        Returns:
            List of tensors containing indices of neurons to resample for each component
        """
        # First get standard dead neurons
        dead_indices = super()._get_dead_neuron_indices()
        
        # Track statistics for each component
        resample_stats = []
        
        # For each component, identify and add redundant neurons
        for component_idx in range(self._n_components):
            if self.use_encoder:
                weights = self.autoencoder.encoder.weight[component_idx]
                normalized_weights = torch.nn.functional.normalize(weights, dim=1)
            else:
                weights = self.autoencoder.decoder.weight[component_idx]
                normalized_weights = torch.nn.functional.normalize(weights.T, dim=1)
                
            normalized_embds = torch.nn.functional.normalize(self.embd_dictionary, dim=1)
            
            # Calculate cosine similarity matrix and get max similarities
            cos_sim = torch.matmul(normalized_weights, normalized_embds.T)
            max_sims, max_embd_idx = cos_sim.max(dim=1)
            
            # Create mask for neurons above similarity threshold
            above_threshold = max_sims >= self.cosine_similarity_threshold
            
            # For each embedding, find redundant neurons
            n_embds = len(self.embd_dictionary)
            embd_masks = max_embd_idx.unsqueeze(1) == torch.arange(n_embds, device=max_embd_idx.device)
            aligned_groups = embd_masks & above_threshold.unsqueeze(1)
            
            # Find best neuron for each embedding group
            group_sims = max_sims.unsqueeze(1).expand(-1, n_embds) * aligned_groups
            best_neurons = group_sims.argmax(dim=0)
            
            # Create mask for redundant neurons
            redundant_mask = torch.zeros_like(max_sims, dtype=torch.bool)
            redundant_groups = []  # Track groups with redundancies
            
            for embd_idx in range(n_embds):
                group_mask = aligned_groups[:, embd_idx]
                if group_mask.sum() > 1:  # More than one neuron in group
                    best_neuron = best_neurons[embd_idx]
                    redundant_indices = torch.where(group_mask & (torch.arange(len(max_sims), device=max_sims.device) != best_neuron))[0]
                    redundant_mask |= (group_mask & (torch.arange(len(max_sims), device=max_sims.device) != best_neuron))
                    
                    if len(redundant_indices) > 0:
                        redundant_groups.append({
                            'embedding_idx': embd_idx,
                            'best_neuron': best_neuron.item(),
                            'redundant_neurons': redundant_indices.tolist(),
                            'similarity': max_sims[best_neuron].item()
                        })
            
            # Combine dead and redundant indices
            original_dead = dead_indices[component_idx]
            redundant_indices = torch.where(redundant_mask)[0]
            combined_indices = torch.unique(torch.cat([original_dead, redundant_indices.to(original_dead.device)]))
            dead_indices[component_idx] = combined_indices
            
            # Collect statistics for this component
            stats = {
                'component_idx': component_idx,
                'n_dead_neurons': len(original_dead),
                'n_redundant_neurons': len(redundant_indices),
                'n_total_resampled': len(combined_indices),
                'redundant_groups': redundant_groups,
                'mean_max_similarity': max_sims.mean().item(),
                'n_above_threshold': above_threshold.sum().item()
            }
            resample_stats.append(stats)
            
            # Log statistics if wandb is available
            if wandb.run is not None:
                # Log summary statistics
                log_dict = {
                    f"resample/redundant_neurons_summary": len(redundant_indices),
                    f"resample/neurons_above_threshold_summary": above_threshold.sum().item(),
                    f"resample/neurons_dead_summary": len(original_dead),
                    f"resample/total_resampled_summary": len(combined_indices),
                    f"resample/mean_max_similarity_summary": max_sims.mean().item(),
                }
                wandb.log(log_dict)
        
        # Store stats for potential later use
        self.last_resample_stats = resample_stats
        
        return dead_indices
    
    def resample_dead_neurons(
        self,
        activation_store: ActivationStore,
        autoencoder: SparseAutoencoder,
        loss_fn: AbstractLoss,
        train_batch_size: int,
    ) -> list[ParameterUpdateResults]:
        """Store autoencoder reference and call parent implementation."""
        # Store reference to autoencoder for use in _get_dead_neuron_indices
        self.autoencoder = autoencoder
        return super().resample_dead_neurons(
            activation_store=activation_store,
            autoencoder=autoencoder,
            loss_fn=loss_fn,
            train_batch_size=train_batch_size
        )