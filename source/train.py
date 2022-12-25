import time
import warnings
import torch
import click
import config
import numpy as np
import pytorch_metric_learning as torchml

from tqdm import tqdm
from model import SimCLR
from dataset import LFW, augmentation
from pytorch_metric_learning import losses


warnings.simplefilter("ignore")


def set_seed(seed=42) -> None:
    """
    Set seed for reproducibility.
    :param seed: seed;
    :return: None.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)


def inference(
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    loss_fn: torchml.losses,
    dataloader: torch.utils.data.DataLoader,
    device="cuda",
    is_training=True,
) -> float:
    """
    Inference function;
    :return: loss per epoch.
    """
    if is_training:
        model.train()
    else:
        model.eval()

    loss_epoch = 0

    for x_1, x_2, classes in dataloader:
        if is_training:
            opt.zero_grad()

        # Get 2 augmented images
        x_1 = x_1.to(device)
        x_2 = x_2.to(device)

        # Get image embeddings
        z_1 = model(x_1)
        z_2 = model(x_2)

        # Concatenate embeddings and classes
        embeddings = torch.cat((z_1, z_2), dim=0)
        labels = torch.cat((classes, classes), dim=0)

        loss = loss_fn(embeddings, labels)

        if is_training:
            loss.backward()
            opt.step()

        loss_epoch += loss.item()

    return loss_epoch


def train(
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    loss_fn: torchml.losses,
    num_epoch: int,
    tr_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    save_path: str,
    device="cuda",
) -> None:
    """
    Train function.
    """
    train_losses = []
    validation_losses = []

    for epoch in tqdm(range(num_epoch)):
        start = time.time()

        train_loss_epoch = inference(model, opt, loss_fn, tr_loader, device)
        val_loss_epoch = inference(
            model, opt, loss_fn, val_loader, device, is_training=False
        )

        train_losses.append(train_loss_epoch / len(tr_loader))
        validation_losses.append(val_loss_epoch / len(val_loader))

        print(
            f"\nEpoch №{epoch + 1} | "
            f"Train loss: {train_loss_epoch / len(tr_loader):.4f} | "
            f"Validation loss: {val_loss_epoch / len(val_loader):.4f} | "
            f"Time: {time.time() - start:.2f} seconds"
        )

        torch.save(model.state_dict(), save_path)


@click.command()
@click.option("-dp", "--data_path", help="Path to store the dataset.")
@click.option(
    "-mp", "--model_path", help="Path to store the serialized model (.pth)."
)
def main(data_path: str, model_path: str) -> None:
    """
    Main function.
    """
    print(f"Training on {config.DEVICE}...")
    set_seed()

    train_dataset = LFW(
        split="train",
        root=data_path,
        transform=augmentation["train"],
        download=True,
    )

    valid_dataset = LFW(
        split="test",
        root=data_path,
        transform=augmentation["valid"],
        download=True,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config.BATCH_SIZE, shuffle=False
    )

    model = SimCLR().to(config.DEVICE)
    criterion = losses.SupConLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.LEARNING_RATE)

    train(
        model=model,
        opt=optimizer,
        loss_fn=criterion,
        num_epoch=config.NUM_EPOCH,
        device=config.DEVICE,
        tr_loader=train_loader,
        val_loader=valid_loader,
        save_path=model_path,
    )


if __name__ == "__main__":
    main()
