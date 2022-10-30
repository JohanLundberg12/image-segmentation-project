from typing import List
import os
from PIL import Image


def get_name(path: str):
    name, ext = os.path.splitext(path.split("_")[-2])

    return name


class SeaAnimalsDataset:
    def __init__(
        self, img_path, transform, train: bool = True, augmentations: List[str] = ["00"]
    ):
        self.img_path = img_path
        self.transform = transform
        self.total_imgs = os.listdir(self.img_path)

        if train:
            self.total_imgs = [
                img_path
                for img_path in self.total_imgs
                if img_path[-6:-4] in augmentations
            ]
        self.total_labels = list(
            map(lambda img_path: get_name(img_path), self.total_imgs)
        )

        dictionary = {
            "Corals": 0,
            "Crabs": 1,
            "Dolphin": 2,
            "Eel": 3,
            "Jelly Fish": 4,
            "Lobster": 5,
            "Nudibranchs": 6,
            "Octopus": 7,
            "Penguin": 8,
            "Puffers": 9,
            "Sea Rays": 10,
            "Sea Urchins": 11,
            "Seahorse": 12,
            "Seal": 13,
            "Sharks": 14,
            "Squid": 15,
            "Starfish": 16,
            "Tortoise": 17,
            "Whale": 18,
        }
        self.total_labels = [dictionary[k] for k in self.total_labels]

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.img_path, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        label = self.total_labels[idx]
        out_image = self.transform(image)

        return out_image, label
