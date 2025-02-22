# import sys
import math
import torchvision
import os.path as osp
import torch
import random
import numpy as np
from tqdm import tqdm
from dncbm import config
from pathlib import Path
from dncbm.data_utils import probe_classnames
import os
import clip

import torch.utils


def save_activation_hook(model, input, output):
    """
    Hook to save intermediate activations
    """
    model.activations = output


def get_img_model(args):
    if args.img_enc_name.startswith('clip'):
        model, preprocess = clip.load(
            args.img_enc_name[5:], device=args.device)
    elif args.img_enc_name.startswith("resnet50"):
        model = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])
        preprocess = torchvision.transforms.Compose([torchvision.transforms.Resize(
            256), torchvision.transforms.CenterCrop(224), torchvision.transforms.ToTensor(), normalize])
    return model, preprocess


def get_sae_ckpt(args, autoencoder):
    """
    Loads the SAE checkpoint given configuration in args
    """
    save_dir_ckpt = args.save_dir_sae_ckpts[args.modality]
    ckpt_path = osp.join(save_dir_ckpt, f'sparse_autoencoder_final.pt')
    print(f"Loading SAE checkpoint from: {ckpt_path}")
    state_dict = torch.load(ckpt_path)
    autoencoder.load_state_dict(state_dict)
    return autoencoder


def get_probe_classifier_ckpt(args, which_ckpt=None, name_only=False):
    """
    Loads and returns the probe classifier checkpoint and filename, given args
    """
    if which_ckpt is None:
        which_ckpt = args.probe_classifier_which_ckpt

    checkpoint_save_path = osp.join(
        args.probe_cs_save_dir, args.probe_config_name, "on_concepts_ckpts")

    whole_ckpt_fname = osp.join(
        checkpoint_save_path, f"on_concepts_{which_ckpt}_{args.probe_config_name}.pt")
    if not name_only:
        print(f"Loading classifier checkpoint from: {checkpoint_save_path}")
        state_dict = torch.load(whole_ckpt_fname)

        return state_dict, whole_ckpt_fname
    else:
        return whole_ckpt_fname


def set_seed(seed):
    """
    Set seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_printable_class_name(probe_dataset, class_idx):
    """
    Returns cleaned up class names for visualizations
    """
    if probe_dataset == "places365":
        class_name = " ".join(
            probe_classnames.probe_classes_dict[probe_dataset][class_idx].split("/")[2:])
        class_name = " ".join(class_name.split("_")).capitalize()
    elif probe_dataset == "imagenet":
        class_name = probe_classnames.imagenet_classes_clip[class_idx]
        class_name = class_name.capitalize()
    else:
        class_name = probe_classnames.probe_classes_dict[probe_dataset][class_idx]
        class_name = class_name.capitalize()
    return class_name


def common_init(args, disable_make_dirs=False):
    """
    Performs initializations of variables common to several scripts, and creates directories where applicable
    """
    set_seed(args.seed)

    # Update config_name to include alignment type and resampling strategy
    align_suffix = f"_align{args.alignment_lam}"
    if args.alignment_lam > 0:
        align_suffix += f"_{args.align_loss_type}"
        if args.resample_aligned_neuron:
            align_suffix += "_aligned_resample"
            
    # Add max_n_resamples to config name if it's set
    resample_suffix = f"_rf{args.resample_freq}"
    if args.max_n_resamples > 0:
        resample_suffix += f"_maxr{args.max_n_resamples}"
        
    args.config_name = f"vocab_{args.vocab_embedding_file}_lr{args.lr}_l1coeff{args.l1_coeff}_ef{args.expansion_factor}{resample_suffix}_hook{args.hook_points[0]}_bs{args.train_sae_bs}_epo{args.num_epochs}{align_suffix}"
    
    # Update CSV config name to include max_n_resamples
    args.config_name_csv = f"{args.img_enc_name},{args.hook_points[0]},{args.sae_dataset},{args.lr},{args.l1_coeff},{args.expansion_factor},{args.resample_freq},{args.max_n_resamples},{args.train_sae_bs},{args.num_epochs},{args.alignment_lam},{args.align_loss_type},{args.resample_aligned_neuron}"

    args.img_enc_name_for_saving = args.img_enc_name.replace('/', '')

    # Directory names
    args.autoencoder_input_dim_dict = config.autoencoder_input_dim_dict
    args.data_dir_root = config.data_dir_root
    args.save_dir_root = config.save_dir_root
    args.probe_cs_save_dir_root = config.probe_cs_save_dir_root
    args.vocab_dir = config.vocab_dir
    args.analysis_dir = config.analysis_dir

    # Set the embeddings filename based on the vocab_embedding_file argument
    args.embeddings_filename = f"embeddings_{args.img_enc_name_for_saving}_{args.vocab_embedding_file}.pth"

    args.data_dir_activations = {}
    args.data_dir_activations["img"] = osp.join(
        args.data_dir_root, 'activations_img', args.sae_dataset, args.img_enc_name_for_saving, args.hook_points[0])

    args.probe_data_dir_activations = {}
    args.probe_data_dir_activations["img"] = osp.join(
        args.data_dir_root, 'activations_img', args.probe_dataset, args.img_enc_name_for_saving, args.hook_points[0])

    args.probe_split_idxs_dir = {}
    args.probe_split_idxs_dir["img"] = osp.join(
        args.data_dir_root, 'activations_img', args.probe_dataset)
    
    args.ae_input_dim_dict_key = {}
    args.ae_input_dim_dict_key["img"] = f"{args.img_enc_name_for_saving}_{args.hook_points[0]}"

    args.save_dir = {}
    args.save_dir_sae_ckpts = {}

    args.save_dir["img"] = Path(osp.join(
        args.save_dir_root, f"SAEImg/{args.sae_dataset}/{args.img_enc_name_for_saving}/{args.hook_points[0]}/{args.config_name}"))

    if not disable_make_dirs:
        os.makedirs(osp.join(args.save_dir_root,
                    f"SAEImg/{args.sae_dataset}/{args.img_enc_name_for_saving}/{args.hook_points[0]}"), exist_ok=True)
        # os.makedirs(osp.join(args.save_dir_root, f"SAEText/{args.sae_dataset}/{args.text_enc_name_for_saving}/{args.hook_points[0]}"), exist_ok=True)

    for modality in args.save_dir:
        if not disable_make_dirs:
            args.save_dir[modality].mkdir(exist_ok=True)
        args.save_dir_sae_ckpts[modality] = Path(
            osp.join(args.save_dir[modality], "sae_checkpoints"))

        if not disable_make_dirs:
            args.save_dir_sae_ckpts[modality].mkdir(exist_ok=True)

    args.enc_name = {}
    args.enc_name["img"] = args.img_enc_name
    args.enc_name_for_saving = {}
    args.enc_name_for_saving["img"] = args.img_enc_name_for_saving

    bias_str = "nobias"

    if args.probe_classification_loss == "CE" and args.probe_sparsity_loss is None:
        args.probe_config_name = f"lr{args.probe_lr}_bs{args.probe_train_bs}_epo{args.probe_epochs}_{bias_str}"
    else:
        args.probe_config_name = f"lr{args.probe_lr}_bs{args.probe_train_bs}_epo{args.probe_epochs}_{bias_str}_cl{args.probe_classification_loss}_sp{args.probe_sparsity_loss}_spl{args.probe_sparsity_loss_lambda}"

    args.probe_dataset_root_dir = config.probe_dataset_root_dir_dict[args.probe_dataset]

    args.probe_features_save_dir = osp.join(
        config.probe_cs_save_dir_root, args.sae_dataset, args.img_enc_name_for_saving, args.hook_points[0], "on_features", args.probe_dataset)

    args.probe_cs_save_dir = osp.join(
        config.probe_cs_save_dir_root, args.sae_dataset, args.img_enc_name_for_saving, args.hook_points[0], args.config_name, args.probe_dataset)

    args.probe_labels_dir = {}
    args.probe_labels_dir['img'] = osp.join(
        args.data_dir_root, 'activations_img', args.probe_dataset)

    args.probe_nclasses = config.probe_dataset_nclasses_dict[args.probe_dataset]

    args.probe_config_name_csv = f"{args.probe_lr},{args.probe_train_bs},{args.probe_epochs},{bias_str},{args.probe_classification_loss},{args.probe_sparsity_loss},{args.probe_sparsity_loss_lambda}"
    args.probe_csv_path = osp.join(config.probe_cs_save_dir_root, 'probe_results.csv')

def get_probe_dataset(probe_dataset, probe_split, probe_dataset_root_dir, preprocess_fn, split_idxs=None):
    """
    Loads and returns a downstream dataset given the dataset name, split, root directory, and preprocessing function
    """
    if probe_dataset == "imagenet":
        dataset = torchvision.datasets.ImageFolder(
            os.path.join(probe_dataset_root_dir, probe_split), transform=preprocess_fn)
    elif probe_dataset == "cifar100":
        dataset = torchvision.datasets.CIFAR100(
            root=os.path.join(probe_dataset_root_dir), train=probe_split == "train", download=False, transform=preprocess_fn)
    elif probe_dataset == "places365":
        if probe_split == 'train':
            suffix = '-standard'
        else:
            suffix = ''
        dataset = torchvision.datasets.Places365(
            root=os.path.join(probe_dataset_root_dir), split=f"{probe_split}{suffix}", download=False, small=True, transform=preprocess_fn)
    elif probe_dataset == "cifar10":
        dataset = torchvision.datasets.CIFAR10(
            root=os.path.join(probe_dataset_root_dir), train=probe_split == "train", download=False, transform=preprocess_fn)
    else:
        raise NotImplementedError
    if split_idxs is not None:
        dataset = torch.utils.data.Subset(dataset, split_idxs)
    return dataset


def cos_similarity_cubed(clip_feats, target_feats, device='cuda', batch_size=10000, min_norm=1e-3):
    """
    Substract mean from each vector, then raises to third power and compares cos similarity
    Does not modify any tensors in place
    """
    with torch.no_grad():
        torch.cuda.empty_cache()
        
        clip_feats = clip_feats - torch.mean(clip_feats, dim=0, keepdim=True)
        target_feats = target_feats - torch.mean(target_feats, dim=0, keepdim=True)
        
        clip_feats = clip_feats**3
        target_feats = target_feats**3
        
        clip_feats = clip_feats/torch.clip(torch.norm(clip_feats, p=2, dim=0, keepdim=True), min_norm)
        target_feats = target_feats/torch.clip(torch.norm(target_feats, p=2, dim=0, keepdim=True), min_norm)
        
        similarities = []
        for t_i in tqdm(range(math.ceil(target_feats.shape[1]/batch_size))):
            curr_similarities = []
            curr_target = target_feats[:, t_i*batch_size:(t_i+1)*batch_size].to(device).T
            for c_i in range(math.ceil(clip_feats.shape[1]/batch_size)):
                curr_similarities.append(curr_target.float() @ clip_feats[:, c_i*batch_size:(c_i+1)*batch_size].to(device).float())
            similarities.append(torch.cat(curr_similarities, dim=1))
    return torch.cat(similarities, dim=0)


def soft_wpmi(clip_feats, target_feats, top_k=100, a=10, lam=1, device='cuda',
                        min_prob=1e-7, p_start=0.998, p_end=0.97):
    
    with torch.no_grad():
        torch.cuda.empty_cache()
        clip_feats = torch.nn.functional.softmax(a*clip_feats, dim=1)

        inds = torch.topk(target_feats, dim=0, k=top_k)[1]
        prob_d_given_e = []

        p_in_examples = p_start-(torch.arange(start=0, end=top_k)/top_k*(p_start-p_end)).unsqueeze(1).to(device)
        for orig_id in tqdm(range(target_feats.shape[1])):
            
            curr_clip_feats = clip_feats.gather(0, inds[:,orig_id:orig_id+1].expand(-1,clip_feats.shape[1])).to(device)
            
            curr_p_d_given_e = 1+p_in_examples*(curr_clip_feats-1)
            curr_p_d_given_e = torch.sum(torch.log(curr_p_d_given_e+min_prob), dim=0, keepdim=True)
            prob_d_given_e.append(curr_p_d_given_e)
            torch.cuda.empty_cache()

        prob_d_given_e = torch.cat(prob_d_given_e, dim=0)
        print(prob_d_given_e.shape)
        #logsumexp trick to avoid underflow
        prob_d = (torch.logsumexp(prob_d_given_e, dim=0, keepdim=True) - 
                  torch.log(prob_d_given_e.shape[0]*torch.ones([1]).to(device)))
        mutual_info = prob_d_given_e - lam*prob_d
    return mutual_info


def correlation_coefficient(clip_feats, target_feats, device='cuda', batch_size=2048):
    """Calculate pairwise correlation coefficients between two sets of features.
    
    Args:
        clip_feats: First set of features (n_samples, n_features1)
        target_feats: Second set of features (n_samples, n_features2)
        device: Device to perform computation on
        batch_size: Batch size for memory efficiency
        
    Returns:
        Tensor of correlation coefficients (n_features1, n_features2)
    """
    with torch.no_grad():
        torch.cuda.empty_cache()
        
        # Center the features
        clip_feats = clip_feats - torch.mean(clip_feats, dim=0, keepdim=True)
        target_feats = target_feats - torch.mean(target_feats, dim=0, keepdim=True)
        
        # Calculate standard deviations
        clip_std = torch.std(clip_feats, dim=0, keepdim=True)
        target_std = torch.std(target_feats, dim=0, keepdim=True)
        
        # Normalize features
        clip_feats = clip_feats / (clip_std + 1e-8)
        target_feats = target_feats / (target_std + 1e-8)
        
        # Calculate correlation coefficients in batches
        correlations = []
        for t_i in tqdm(range(math.ceil(target_feats.shape[1]/batch_size))):
            curr_correlations = []
            curr_target = target_feats[:, t_i*batch_size:(t_i+1)*batch_size].to(device)
            
            for c_i in range(math.ceil(clip_feats.shape[1]/batch_size)):
                # Calculate correlation for current batch
                curr_clip = clip_feats[:, c_i*batch_size:(c_i+1)*batch_size].to(device)
                curr_corr = (curr_target.T @ curr_clip) / (clip_feats.shape[0] - 1)
                curr_correlations.append(curr_corr)
                
            correlations.append(torch.cat(curr_correlations, dim=1))
            
        return torch.cat(correlations, dim=0)



