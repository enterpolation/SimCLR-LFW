import warnings
import click
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from model import SimCLR
from dataset import LFW, augmentation
from sklearn.manifold import TSNE

warnings.simplefilter("ignore")


@click.command()
@click.option("-dp", "--data_path", help="Path to store the dataset.")
@click.option("-mp", "--model_path", help="Path to the serialized model.")
@click.option("-pp", "--plot_path", help="Path to store the plot.")
@click.option("-k", "--k_classes", type=int, help="Number of classes to plot.")
def plot(
    data_path: str,
    model_path: str,
    plot_path: str,
    k_classes: int,
    n_objects=500,
) -> None:
    """
    Plotting classes with TSNE decomposition.
    """
    train_dataset = LFW(
        split="train",
        root=data_path,
        transform=augmentation["train"],
        download=True,
    )

    simclr_model = SimCLR()
    simclr_model.load_state_dict(torch.load(model_path))
    simclr_model.eval()

    classes = []
    for i in range(n_objects):
        _, _, label = train_dataset[i]
        classes.append(label)

    unique, counts = np.unique(classes, return_counts=True)
    sorted_indexes = np.argsort(counts)[::-1]
    sorted_by_freq = unique[sorted_indexes]
    classes = set(sorted_by_freq[:k_classes])

    samples = []
    labels = []
    for i in range(n_objects):
        _, image, label = train_dataset[i]
        if label in classes:
            samples.append(image[None, :, :, :])
            labels.append(label)

    samples = torch.cat(samples)
    z = simclr_model(samples)

    decompositor = TSNE(n_components=2)
    lowd_data = decompositor.fit_transform(z.detach())

    sns.set_style("darkgrid")
    plt.title("TSNE representation")
    sns.scatterplot(
        x=lowd_data[:, 0], y=lowd_data[:, 1], hue=labels, palette="crest"
    )
    plt.xlabel("1st component")
    plt.ylabel("2nd component")
    plt.savefig(plot_path, dpi=800)


if __name__ == "__main__":
    plot()
