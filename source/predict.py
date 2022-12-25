import warnings
import torch
import click

from numpy import dot
from numpy.linalg import norm
from model import SimCLR
from dataset import augmentation
from PIL import Image

warnings.simplefilter("ignore")


def similarity(model: SimCLR, x_1: torch.Tensor, x_2: torch.Tensor) -> float:
    """
    Cosine similarity;
    :param model: contrastive model;
    :param x_1: first image;
    :param x_2: second image;
    :return: similarity.
    """
    z_1 = model(x_1.unsqueeze(0))[0].detach().numpy()
    z_2 = model(x_2.unsqueeze(0))[0].detach().numpy()
    return dot(z_1, z_2) / (norm(z_1) * norm(z_2))


@click.command()
@click.option("-mp", "--model_path", help="Path to the serialized model.")
@click.option("-imgp1", "--image_path_1", help="Path to image №1.")
@click.option("-imgp2", "--image_path_2", help="Path to image №2.")
def predict(
    model_path: str,
    image_path_1: str,
    image_path_2: str,
) -> None:
    """
    Prediction function;
    :param model_path: path to the serialized model;
    :param image_path_1: path to the first image;
    :param image_path_2: path to the second image;
    :return: None.
    """
    with Image.open(image_path_1) as img_1:
        img_1.load()
        img_1 = augmentation["valid"](img_1)

    with Image.open(image_path_2) as img_2:
        img_2.load()
        img_2 = augmentation["valid"](img_2)

    model = SimCLR()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"Cosine similarity: {similarity(model, img_1, img_2):.4f}")


if __name__ == "__main__":
    predict()
