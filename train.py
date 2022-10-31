import sys
import yaml
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader

from BaseLineModel import BaseLineModel
from AquaNet import AquaNet

from SeaAnimalsDataset import SeaAnimalsDataset
from Trainer import Trainer
from EarlyStopping import EarlyStopping
from utils import get_device


def create_model(config: dict):
    if config["model"]["name"] == "Baseline":
        model = BaseLineModel()
    elif config["model"]["name"] == "AquaNet":
        model = AquaNet()
    else:
        raise NameError

    return model


def get_transform():
    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),  # Shape: HWC, Scales data into [0,1] by div / 255
            transforms.Normalize(
                mean=[0, 0, 0], std=[1, 1, 1]
            ),  # image_channel = (image - mean) / std
        ]
    )

    return transform


def create_dataloaders(config: dict):
    augmentations = []
    if config["augmentation"]["name"] == "all":
        augmentations = ["00", "01", "02", "03", "04", "05", "06"]
    elif not config["augmentation"]["name"]:
        augmentations = ["00"]
    else:
        augmentations.append(config["augmentation"]["name"])

    transformation = get_transform()  # get transformations on the image

    sea_animals_train = SeaAnimalsDataset(
        img_path="train_augment",
        transform=transformation,
        train=True,
        augmentations=augmentations,
    )
    sea_animals_val = SeaAnimalsDataset(
        img_path="val", transform=transformation, train=False
    )
    sea_animals_test = SeaAnimalsDataset(
        img_path="test", transform=transformation, train=False
    )
    train_loader = DataLoader(
        sea_animals_train, batch_size=8, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        sea_animals_val, batch_size=8, shuffle=True, drop_last=False
    )
    test_loader = DataLoader(
        sea_animals_test, batch_size=8, shuffle=True, drop_last=False
    )

    return train_loader, val_loader, test_loader


def make(config: dict):
    """Return model and dataloader with/without data augmentation"""

    model = create_model(config)  # get model
    train_loader, val_loader, test_loader = create_dataloaders(config)

    return model, train_loader, val_loader, test_loader


def main(config: dict):
    model, train_loader, val_loader, test_loader = make(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    loss_fn = nn.CrossEntropyLoss()
    epochs = 10
    device = get_device()
    exp_name = f'{config["model"]["name"]}_{config["augmentation"]["name"]}'
    early_stopping = EarlyStopping(
        patience=5,
        verbose=True,
        path=f"{exp_name}_checkpoint.pt",
    )

    trainer = Trainer(
        config=config,
        model=model.to(device),
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=epochs,
        device=device,
        early_stopping=early_stopping,
    )
    results = trainer.train()

    result = trainer.predict()

    print(f"\ntrain-Val results: {results}")
    print(f"\navg test-f1: {result}")

    with open(f"{exp_name}_results.txt", "w") as file:
        file.write("\ntrain-Val results: \n")
        for key, val in results.items():
            file.write(f"{key}: ")
            for item in val:
                file.write(str(item))
            file.write("\n")
        file.write(f"\n avg test-f1: {str(result)}")


if __name__ == "__main__":
    argv = sys.argv[1]
    config = yaml.safe_load(open(argv, "r"))
    main(config)
