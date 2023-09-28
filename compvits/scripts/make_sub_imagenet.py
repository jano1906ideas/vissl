import os, shutil
from argparse import ArgumentParser
from typing import List, Optional

def get_parser():
    parser = ArgumentParser("sub_imagenet_creator",
                            "Extract subset of ImageFolder formatted dataset to another directory.")
    parser.add_argument("--root", type=str, required=True, help="path to directory containing 'train' or 'val' as subdirectories")
    parser.add_argument("--out_dir", type=str, required=True, help="path to output directory")
    parser.add_argument("--no_val", action="store_true", help="set this flag to ignore validation set")
    parser.add_argument("--no_train", action="store_true", help="set this flag to ignore training set")
    parser.add_argument("--n_classes", type=int, default=10, help="number of extracted subclasses from dataset")
    parser.add_argument("--classes", nargs="+", required=False, help="name list of subclasses to be extracted from dataset, ignore 'n_classes'")
    return parser

def make_sub_inet(root: str, out_dir: str, no_val: bool = False, no_train: bool = False, n_classes: int = 10, classes: Optional[List[str]] = None):
    dirs = [d.name for d in os.scandir(root) if d.is_dir() and d.name in ["val", "train"]]
    assert no_val or "val" in dirs
    assert no_train or "train" in dirs
    exports = []
    if not no_val: exports.append("val")
    if not no_train: exports.append("train")
    if len(exports) == 0: return
    src_dirs = [os.path.join(root, exp) for exp in exports]
    exp_dirs = [os.path.join(out_dir, exp) for exp in exports]
    if classes is None: 
        classes = []
        it = os.scandir(src_dirs[0])
        for _ in range(n_classes):
            classes.append(next(it).name)
    
    for exp in exports:
        assert all([os.path.isdir(os.path.join(root, exp, cl)) for cl in classes]), f"Error, {os.path.join(root, exp, cl)} is not a directory!"

    os.makedirs(out_dir, exist_ok=True)
    for exp_dir, src_dir in zip(exp_dirs, src_dirs):
        os.makedirs(exp_dir, exist_ok=False)
        for cl in classes:
            shutil.copytree(os.path.join(src_dir, cl), os.path.join(exp_dir, cl))
    
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    make_sub_inet(**dict(args._get_kwargs()))