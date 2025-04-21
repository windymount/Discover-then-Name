

autoencoder_input_dim_dict = {'clip_RN50_out': 1024,
                              'clip_ViT-B16_out': 512,  
                              'clip_ViT-L14_out': 768, }

data_dir_root = './data'
save_dir_root = './SAE'
probe_cs_save_dir_root = './probe'
vocab_dir = './vocab'
analysis_dir = './analysis'



probe_dataset_root_dir_dict = {
    "places365": "/home/geyan/datasets/places365_torch",
    "imagenet": "/home/geyan/datasets/imagenet",
    "cifar10": "/home/geyan/datasets/cifar10",
    "cifar100": "/home/geyan/datasets/cifar100",
    "cc3m": "/home/geyan/datasets/cc3m",
    "cc12m": "/home/geyan/datasets/cc12m",
}

probe_dataset_nclasses_dict = {"places365": 365,
                               'imagenet': 1000, "cifar10": 10, "cifar100": 100, "cc3m": 1000, "cc12m": 12000}
