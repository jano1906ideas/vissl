import random
from compvits.constants import DIVISION_MASKS
from compvits.scripts.train_linear_head import LinearClassifier
from compvits import deit_utils
import os
import json
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from argparse import ArgumentParser
import torch
from torch import nn

def build_transform(input_size = 224):
    resize_im = input_size > 32
    t = []
    if resize_im:
        size = int((256 / 224) * input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


















@torch.no_grad()
def evaluate(data_loader, model, device, Ks=list(range(13)), Ms=[16]):
    metric_logger = deit_utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            outputs = [
                [[k, m], model((images, k, m))]
                for k in Ks
                for m in Ms
            ]
        accuracies = [
            [[k, m, i], accuracy(out, target)[0]]
            for [[k, m], outs] in outputs
            for i, out in enumerate(outs)
        ]
        
        batch_size = images.shape[0]
        for [[k, m, i], acc] in accuracies:
            metric_logger.meters[f'acc1_K{k}_M{m}_i{i}'].update(acc.item(), n=batch_size)
            
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

class Model(nn.Module):
    def __init__(self, trunk, head):
        super().__init__()
        self.trunk = trunk
        self.head = head
    
    def forward(self, input):
        x, k, m = input
        output = self.trunk.comp_seq(x, k, m)
        output = torch.stack(output, dim=0)
        output = self.head(output)
        return output


def get_parser():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    return parser


import compvits.deit_arch as deit_arch
import compvits.ibot_arch as ibot_arch

def load_arch(model_name):
    if model_name == "deit":
        arch = deit_arch.deit_small_patch16_LS()
        checkpoint = torch.hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/deit/deit_3_small_224_1k.pth",
                                                    map_location='cpu', check_hash=True)
        arch.load_state_dict(checkpoint["model"])
        return arch

    trunk = ibot_arch.vit_small()
    state = torch.load(f"/home/jan.olszewski/git/vissl/checkpoints/trunk_only/small/{model_name}.pth", "cpu")
    trunk.load_state_dict({k[len("trunk."):]: v for k,v in state["model"].items() if "trunk" in k})
    head = LinearClassifier(384)
    state = torch.load(f"/home/jan.olszewski/git/vissl/checkpoints/heads/small/{model_name}.pth", "cpu")
    head.load_state_dict({k[len("module."):] : v for k,v in state["state_dict"].items()})
    model = Model(trunk, head).to("cuda")
    return model

def main(model_name: str):
    output_dir = "logs/small/sequential_accuracy"
    os.makedirs(os.path.join(output_dir, model_name), exist_ok=True)
    
    dataset = ImageFolder("/home/jan.olszewski/git/vissl/datasets/imagenet-1k-kaggle/ILSVRC/Data/CLS-LOC/val", transform=build_transform())
    dataloader = DataLoader(dataset, 32, num_workers=6)

    model = load_arch(model_name).to("cuda")

    log_stats = evaluate(dataloader, model, "cuda")
    with open(os.path.join(output_dir, model_name, "metrics.json"), "a") as f:
        f.write(json.dumps(log_stats) + "\n")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args.model)