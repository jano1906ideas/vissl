import torch
from collections import OrderedDict
import sys
import os

def download_and_save_deit3(link, size):
    checkpoint = torch.hub.load_state_dict_from_url(link,
                                                    map_location='cpu', check_hash=True)
    
    new_checkpoint = []
    for k in checkpoint["model"]:
        if not k.startswith("head."):
            new_k = "trunk." + k
            new_checkpoint.append([new_k, checkpoint["model"][k]])
        else:
            new_k = "heads.0.clf.0." + k[len("head."):]
            new_checkpoint.append([new_k, checkpoint["model"][k]])

    new_checkpoint = OrderedDict(new_checkpoint)

    print(new_checkpoint.keys())
    save = {"model": new_checkpoint}
    torch.save(save, f"checkpoints/{size}/deit.pth")


def download_and_save_ibot(link, size):
    checkpoint = torch.hub.load_state_dict_from_url(
        link,
        map_location='cpu', check_hash=True)
    
    new_checkpoint = []
    for k in checkpoint["state_dict"]:
        new_k = "trunk." + k
        new_checkpoint.append([new_k, checkpoint["state_dict"][k]])

    new_checkpoint = OrderedDict(new_checkpoint)

    new_checkpoint.keys()

    save = {"model": new_checkpoint}
    torch.save(save, f"checkpoints/trunk_only/{size}/ibot.pth")


def download_and_save_dino(link, size):
    checkpoint = torch.hub.load_state_dict_from_url(
        link, map_location='cpu',
        check_hash=True)

    new_checkpoint = []

    for k in checkpoint:
        if not k.startswith("head."):
            new_k = "trunk." + k
            new_checkpoint.append([new_k, checkpoint[k]])

    new_checkpoint = OrderedDict(new_checkpoint)

    print(new_checkpoint.keys())

    save = {"model": new_checkpoint}
    torch.save(save, f"checkpoints/trunk_only/{size}/dino.pth")


def download_and_save_moco3(link, size):
    checkpoint = torch.hub.load_state_dict_from_url(
        link, map_location='cpu',
        check_hash=True)

    new_checkpoint = []

    for k in checkpoint["state_dict"]:
        if k.startswith("module.momentum_encoder") and not k.startswith("module.momentum_encoder.head"):
            new_k = "trunk." + k[len('module.momentum_encoder.'):]
            new_checkpoint.append([new_k, checkpoint["state_dict"][k]])

    new_checkpoint = OrderedDict(new_checkpoint)

    print(new_checkpoint.keys())

    save = {"model": new_checkpoint}
    torch.save(save, f"checkpoints/trunk_only/{size}/moco3.pth")


models_funs = {
    "deit3": download_and_save_deit3,
    "ibot": download_and_save_ibot,
    "dino": download_and_save_dino,
    "moco3": download_and_save_moco3
}

models_sizes_links = {
    "deit3": {"small": "https://dl.fbaipublicfiles.com/deit/deit_3_small_224_1k.pth", "base": "https://dl.fbaipublicfiles.com/deit/deit_3_base_224_1k.pth"},
    "ibot": {"small": "https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vits_16/checkpoint_teacher.pth", "base": "https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitb_16/checkpoint_teacher.pth"},
    "dino": {"small": "https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth", "base": "https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"},
    "moco3": {"small": "https://dl.fbaipublicfiles.com/moco-v3/vit-s-300ep/vit-s-300ep.pth.tar", "base": "https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/vit-b-300ep.pth.tar"},
}


def main(model, size):
    # Your code here
    print("Downloading and saving:", model, size)
    link = models_sizes_links[model][size]
    models_funs[model](link, size)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <model> <size>")
        sys.exit(1)

    model = sys.argv[1]
    size = sys.argv[2]
    os.makedirs(os.path.join("checkpoints", size), exist_ok=True)
    os.makedirs(os.path.join("checkpoints", "trunk_only", size), exist_ok=True)
    main(model, size)