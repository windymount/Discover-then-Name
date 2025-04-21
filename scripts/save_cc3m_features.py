import os
import torch
from tqdm.auto import tqdm
from pprint import pprint
import os.path as osp
import torchvision
# From custom
from dncbm.utils import common_init, get_img_model
from dncbm.data_utils.cc3m import (CC3MImg, CustomDataCollatorImg)
from dncbm import arg_parser


class FetchFeatures:
    def __init__(self, args=None):
        self.model, self.preprocess = get_img_model(args)
        self.model = torch.nn.DataParallel(self.model.visual)
        self.args = args

    def get_cc3m_loader(self, shard, clip_preprocess, batch_size):
        collator = CustomDataCollatorImg()
        cc3m_obj = CC3MImg()
        dtset = cc3m_obj.get_wds_dataset(shard, clip_preprocess, batch_size, collator=collator)
        loader = cc3m_obj.get_dataloader(dtset, batch_size=None, shuffle=False, num_workers=16)

        return loader
    
    def get_cc3m_out(self, loader):
        count = 0
        with torch.no_grad():
            idxs = []
            outs = []

            for (inputs, idx) in tqdm(loader, desc="Processing batches", unit="batch"):
                count += inputs.shape[0]
                idxs.extend(idx)
                inputs = inputs.to(self.args.device)
                if args.debug:
                    outs.append(torch.zeros((1, self.model.module.conv1.weight.shape[1])))
                    continue
                outs.append(self.model(inputs.type(self.model.module.conv1.weight.dtype)).detach().cpu())
                print(f" total data points: {count}")
        return torch.cat(outs, dim=0), idxs
    
    def save_cc3m_features(self):
        data_dir_tar = osp.join(args.data_dir_root, "CC3M_TAR")
        train_path = "training"
        val_path = 'validation'

        train_shard     = os.path.join(data_dir_tar, train_path, "{00000..00299}.tar")
        train_val_shard = os.path.join(data_dir_tar, train_path, "{00300..00331}.tar")
        val_shard       = os.path.join(data_dir_tar, val_path, '{00000..00001}.tar')

        shard_list = [train_shard, val_shard, train_val_shard]
        type_list = ['train', 'val', 'train_val']
        save_dir_activations = args.data_dir_activations[args.modality]
        os.makedirs(save_dir_activations, exist_ok=True)

        for split, shard in zip(type_list, shard_list):
            loader = self.get_cc3m_loader(shard, clip_preprocess=self.preprocess, batch_size=args.batch_size)
            out, indices = self.get_cc3m_out(loader=loader)
            if not args.debug:
                torch.save(out, osp.join(save_dir_activations, split))
            torch.save(indices, osp.join(save_dir_activations, f"{split}_indices.pt"))
            print(f"Save the activations from CLIP model to {save_dir_activations}")


class FetchConceptFeatures(FetchFeatures):
    def __init__(self, args=None, concept_file_name=None):
        self.model, self.preprocess = get_img_model(args)
        self.args = args
        with open(concept_file_name, "r") as f:
            self.concepts = f.read().strip().split("\n")
        
    def get_cc3m_loader(self, shard, clip_preprocess, batch_size):
        collator = CustomDataCollatorImg()
        cc3m_obj = CC3MImg()
        dtset = cc3m_obj.get_wds_dataset(shard, torchvision.transforms.Compose([torchvision.transforms.Resize(384),
                                                                                torchvision.transforms.ToTensor()]), batch_size, collator=collator)
        loader = cc3m_obj.get_dataloader(dtset, batch_size=None, shuffle=False, num_workers=16)
        return loader

    def get_cc3m_out(self, loader):
        count = 0
        self.model = self.model.to(self.args.device)
        self.vision_model = torch.nn.DataParallel(self.model.vision_model)
        with torch.no_grad():
            out = []
            indices = []
            # Pre-compute text embeddings for concepts
            # Process concepts in batches
            batch_size = 512
            text_embeds = []
            for i in range(0, len(self.concepts), batch_size):
                batch_concepts = self.concepts[i:i + batch_size]
                batch_inputs = self.preprocess(text=batch_concepts, return_tensors="pt", padding='max_length')['input_ids'].to(self.args.device)
                batch_embeds = self.model.get_text_features(batch_inputs).detach()
                batch_embeds = batch_embeds / batch_embeds.norm(dim=-1, keepdim=True)
                text_embeds.append(batch_embeds)
            text_embeds = torch.cat(text_embeds, dim=0)
            print(text_embeds.shape)
            for (inputs, idx) in tqdm(loader, desc="Processing batches", unit="batch"):
                indices.extend(idx)
                if args.debug:
                    out.append(torch.zeros((1, len(self.concepts))))
                    continue
                count += inputs.shape[0]
                inputs = inputs.to(self.args.device)
                mean = torch.tensor([0.5, 0.5, 0.5], device=self.args.device).view(1, 3, 1, 1)
                std = torch.tensor([0.5, 0.5, 0.5], device=self.args.device).view(1, 3, 1, 1)
                normalized_images = (inputs - mean) / std
                # Get image embeddings
                print(f"Processed {count} data points")
                image_embeds = self.vision_model(pixel_values=normalized_images).pooler_output.detach()
                image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
                
                # Compute similarity between each image and all concepts
                similarity = image_embeds @ text_embeds.T
                out.append(similarity.cpu())
                
            out = torch.cat(out, dim=0)
            assert out.shape[1] == len(self.concepts), f"Expected output shape (n_samples, {len(self.concepts)}) but got {out.shape}"
        return out, indices


if __name__ == '__main__':
    parser = arg_parser.get_common_parser()
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--concept_file_name", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    common_init(args)
    pprint(vars(args))

    fetch_act = FetchConceptFeatures(args, args.concept_file_name) if args.concept_file_name else FetchFeatures(args)
    fetch_act.save_cc3m_features()
  


    