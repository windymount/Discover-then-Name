from dncbm.custom_pipeline import AlignmentLossType, Pipeline, PipelineWithAlignment, EmbeddingAlignedResampler
from dncbm.customized_sae import ReLUSparseAutoencoder, TopKSparseAutoencoder
import os
from pathlib import Path

import torch
import numpy as np
import math
import datetime

from sparse_autoencoder import (
    ActivationResampler,
    AdamWithReset,
    L2ReconstructionLoss,
    LearnedActivationsL1Loss,
    LossReducer,
)
import wandb
from time import time

from dncbm.arg_parser import get_common_parser
from dncbm.utils import common_init


parser = get_common_parser()
args = parser.parse_args()
common_init(args)
start_time = time()


# Create autoencoder based on specified type
autoencoder_input_dim: int = args.autoencoder_input_dim_dict[
    args.ae_input_dim_dict_key[args.modality]]
n_learned_features = int(autoencoder_input_dim * args.expansion_factor)

if args.sae_type == "ReLUSAE":
    autoencoder = ReLUSparseAutoencoder(
        n_input_features=autoencoder_input_dim,
        n_learned_features=n_learned_features, 
        n_components=len(args.hook_points)
    ).to(args.device)
    loss = LossReducer(LearnedActivationsL1Loss(
    l1_coefficient=float(args.l1_coeff),), L2ReconstructionLoss(),)
elif args.sae_type == "TopKSAE":
    autoencoder = TopKSparseAutoencoder(
        n_input_features=autoencoder_input_dim,
        n_learned_features=n_learned_features,
        k=args.topk_k,
        aux_k=args.topk_aux_k,
        max_dead_steps=1000000 / args.train_sae_bs,
        n_components=len(args.hook_points),
    ).to(args.device)
    loss = L2ReconstructionLoss()
else:
    raise ValueError(f"Unknown SAE type: {args.sae_type}")

max_epoch = args.randmax_max_epoch if args.randmax_max_epoch is not None else args.num_epochs

print(f"Autoencoder ({args.sae_type}) created at {time() - start_time} seconds")

# Load from checkpoint if specified
if args.checkpoint is not None:
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint)
    autoencoder.load_state_dict(checkpoint)
    print(f"Loaded checkpoint at {time() - start_time} seconds")

print(
    f"------------Getting Image activations from directory: {args.data_dir_activations[args.modality]}")
print(f"------------Getting Image activations from model: {args.img_enc_name}")



optimizer = AdamWithReset(
    params=autoencoder.parameters(),
    named_parameters=autoencoder.named_parameters(),
    lr=float(args.lr),
    betas=(float(args.adam_beta_1),
           float(args.adam_beta_2)),
    eps=float(args.adam_epsilon),
    weight_decay=float(args.adam_weight_decay),
    has_components_dim=True,
)

# Load optimizer state from checkpoint if available
if args.checkpoint is not None and 'optimizer_state_dict' in checkpoint:
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Loaded optimizer state from checkpoint at {time() - start_time} seconds")

print(f"Optimizer created at {time() - start_time} seconds")
actual_resample_interval = 1

if args.resample_aligned_neuron and args.alignment_lam > 0.0:
    # Load embeddings dictionary if not already loaded
    if 'embd_dictionary' not in locals():
        embeddings_path = os.path.join(args.vocab_dir, args.embeddings_filename)
        embd_dictionary = torch.load(embeddings_path).float().cuda()
        print(f"Loaded embeddings dictionary for alignment from {embeddings_path}")
    
    # Create aligned resampler with max_n_resamples parameter
    activation_resampler = EmbeddingAlignedResampler(
        embd_dictionary=embd_dictionary,
        cosine_similarity_threshold=0.8,
        use_encoder=args.align_loss_type.endswith("encoder"),
        resample_interval=actual_resample_interval,
        n_activations_activity_collate=actual_resample_interval,
        max_n_resamples=args.max_n_resamples if args.max_n_resamples > 0 else math.inf,
        n_learned_features=n_learned_features,
        resample_epoch_freq=args.resample_freq,
        resample_dataset_size=args.resample_dataset_size,
    )
    print("Using embedding-aligned resampler")
else:
    # Use standard resampler with max_n_resamples parameter
    activation_resampler = ActivationResampler(
        resample_interval=actual_resample_interval,
        n_activations_activity_collate=actual_resample_interval,
        max_n_resamples=args.max_n_resamples if args.max_n_resamples > 0 else math.inf,
        n_learned_features=n_learned_features,
        resample_epoch_freq=args.resample_freq,
        resample_dataset_size=args.resample_dataset_size,
    )

print(f"Activation resampler created at {time() - start_time} seconds")
if args.sae_type == "TopKSAE":
    activation_resampler = None

if args.use_wandb:
    print("wandb started!")
    wandb_project_name = "SAECBM"
    wandb_group_name = f"SAEImg_{args.sae_dataset}_{args.img_enc_name_for_saving}_{args.hook_points[0]}_{args.save_suffix}"

    print(f"wandb started! {wandb_project_name}")

    wandb_dir = os.path.join(args.save_dir[args.modality], ".cache/")
    wandb_path = Path(wandb_dir)
    wandb_path.mkdir(exist_ok=True)
    wandb.init(
        project=wandb_project_name,
        group=wandb_group_name,
        dir=wandb_dir,
        name=args.config_name,
        config=args,)

    wandb.define_metric("custom_steps")
    wandb.define_metric("train/loss_instability_across_batches",
                        step_metric="custom_steps")

    print(f"Wandb initialized at {time() - start_time} seconds")


if args.alignment_lam >= 0.0:
    # Load embeddings dictionary for alignment
    embeddings_path = os.path.join(args.vocab_dir, args.embeddings_filename)
    embd_dictionary = torch.load(embeddings_path).float().cuda()
    print(f"Loaded embeddings dictionary for alignment from {embeddings_path}")
    
    pipeline = PipelineWithAlignment(
        align_lambda=args.alignment_lam,
        align_loss_type=AlignmentLossType[args.align_loss_type],
        embd_dictionary=embd_dictionary,
        activation_resampler=activation_resampler,
        autoencoder=autoencoder,
        checkpoint_directory=Path(
            f"{args.save_dir_sae_ckpts[args.modality]}{args.save_suffix}"),
        loss=loss,
        optimizer=optimizer,
        device=args.device,
        args=args,
        max_epoch=max_epoch,
    )
    print(f"Pipeline with alignment created at {time() - start_time} seconds")
else:
    pipeline = Pipeline(
        activation_resampler=activation_resampler,
        autoencoder=autoencoder,
        checkpoint_directory=Path(
            f"{args.save_dir_sae_ckpts[args.modality]}{args.save_suffix}"),
        loss=loss,
        optimizer=optimizer,
        device=args.device,
        args=args,
        max_epoch=max_epoch,
    )
    print(f"Standard pipeline created at {time() - start_time} seconds")

fnames = os.listdir(args.data_dir_activations[args.modality])
print(f"Getting fnames from {args.data_dir_activations[args.modality]}")

train_fnames = []
train_val_fnames = []
for fname in fnames:
    if fname == "train_val":
        train_val_fnames.append(os.path.join(
            os.path.abspath(args.data_dir_activations[args.modality]), fname))
    elif fname == "train":
        train_fnames.append(os.path.join(
            os.path.abspath(args.data_dir_activations[args.modality]), fname))
if args.val_freq == 0:
    train_fnames = train_fnames + train_val_fnames
    train_val_fnames = None

print(f"Train and Train_val fnames created at {time() - start_time} seconds")

# It takes the train activations and inside split it into train_activations and train_val_activations
pipeline.run_pipeline(
    train_batch_size=int(args.train_sae_bs),
    checkpoint_frequency=int(args.ckpt_freq),
    val_frequency=int(args.val_freq),
    num_epochs=args.num_epochs,
    train_fnames=train_fnames,
    train_val_fnames=train_val_fnames,
    start_time=start_time,
    resample_epoch_freq=args.resample_freq,
    concept_activation_fname=args.concept_activation_fname
)

print(f"-------total time taken------ {np.round(time()-start_time,3)}")
