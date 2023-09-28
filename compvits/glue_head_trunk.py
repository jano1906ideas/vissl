import argparse
import torch
import os
from collections import OrderedDict

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trunk", type=str, required=True)
    parser.add_argument("--head", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    return parser

def main(trunk, head, output_dir, model):
    trunk = torch.load(trunk, "cpu")
    head = torch.load(head, "cpu")

    new_checkpoint = [(k, v) for k,v in trunk["model"].items()]
    for k, v in head["state_dict"].items():
        k:str
        new_k = "heads.0.clf.0."+k.split(".")[-1]
        new_checkpoint.append([new_k, v])
    new_checkpoint = OrderedDict(new_checkpoint)
    save = {"model": new_checkpoint}
    torch.save(save, os.path.join(output_dir, model+".pth"))

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args.trunk, args.head, args.output_dir, args.model)