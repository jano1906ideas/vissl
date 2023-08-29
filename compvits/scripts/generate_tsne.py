import os
import torch
import random
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


@dataclass
class Config:
    in1k_path: str
    cat_dict_path: str
    data_path: str
    M: int
    K: int
    model: str
    num_classes: int


args = Config(in1k_path='/Downloads/val',
              cat_dict_path='/Users/piotrwojcik/PycharmProjects/vissl/compvits/plots/in1k_categories.txt',
              data_path='/Users/piotrwojcik/Downloads/extract_features/',
              M=2,
              K=3,
              model='deitb',
              num_classes=10)


def get_names_dict(cat_dict_path: str):
    cat_dict = {}

    with open(cat_dict_path, 'r') as file:
        # Read each line in the file
        for line in file:
            # Split each line into key and value using ', ' as the separator
            parts = line.strip().split(': ')

            # Ensure that there are at least two parts (key and value)
            if len(parts) >= 2:
                # Extract the key and value
                key = int(parts[0])
                value = ', '.join(parts[1:])
                value = value.split(',')[0].strip().replace('\'', '')

                # Add the key-value pair to the dictionary
                cat_dict[key] = value

    return cat_dict


if __name__ == '__main__':
    random.seed(123)

    names = get_names_dict(args.cat_dict_path)

    features_path = os.path.join(args.data_path, f"M{str(args.M)}", args.model, f"K{str(args.K)}")

    cls_features = np.load(os.path.join(features_path, 'rank0_chunk0_test_lastCLS_features.npy'))
    in_classes = np.load(os.path.join(features_path, 'rank0_chunk0_test_lastCLS_targets.npy'))

    selected_classes = random.sample(list(range(1000)), args.num_classes)

    mask = np.isin(in_classes, list(selected_classes))
    mask = np.squeeze(mask)

    cls_features = cls_features[mask]
    in_classes = in_classes[mask]

    tsne = TSNE(random_state=123, n_components=2, verbose=1, learning_rate=200, perplexity=50,
                n_iter=2000).fit_transform(cls_features)

    for class_label in np.unique(in_classes):

        class_indices = np.where(in_classes == class_label)[0]
        plt.scatter(tsne[class_indices, 0], tsne[class_indices, 1], label=names[class_label], s=10)

    plt.title(f"t-SNE Plot, ImageNet-1k representations (K={args.K}, M={args.M}, model={args.model})")
    plt.legend(loc='upper left', bbox_to_anchor=(1.1, 0.5), fontsize='x-small')

    plt.tight_layout()

    plt.show()

