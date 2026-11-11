"""Extract 512-d ResNet18 embeddings for every product image.

Run from the project root:  python -m src.features.embed
Output: data/processed/embeddings.npz  (ids, X)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

PROCESSED = Path("data/processed")

# ImageNet statistics, required by the pretrained weights
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),   # squash, so the whole product stays visible
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class ImageDataset(Dataset):
    def __init__(self, paths):
        self.paths = list(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")   # some files are grayscale
        return TRANSFORM(img)


def build_model() -> torch.nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()    # drop the classifier -> output is the 512-d vector
    model.eval()
    return model


@torch.no_grad()
def extract(paths, batch_size=64, workers=0) -> np.ndarray:
    model = build_model()
    loader = DataLoader(ImageDataset(paths), batch_size=batch_size,
                        shuffle=False, num_workers=workers)
    chunks = []
    for i, batch in enumerate(loader):
        chunks.append(model(batch).numpy())
        print(f"batch {i + 1}/{len(loader)}", end="\r")
    print()
    return np.vstack(chunks).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=0,
                    help="keep 0 on Windows unless you know it works")
    args = ap.parse_args()

    products = pd.read_csv(PROCESSED / "products.csv")
    X = extract(products["image_path"], args.batch_size, args.workers)

    assert X.shape == (len(products), 512), X.shape
    assert np.isfinite(X).all(), "NaN or inf in embeddings"

    out = PROCESSED / "embeddings.npz"
    np.savez_compressed(out, ids=products["id"].to_numpy(), X=X)
    print(f"saved {X.shape} to {out}")


if __name__ == "__main__":
    main()