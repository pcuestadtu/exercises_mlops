from __future__ import annotations

import matplotlib.pyplot as plt
import torch
from mpl_toolkits.axes_grid1 import ImageGrid
from torch.utils.data import TensorDataset

# UPDATED PATH: based on your previous 'ls' command
DATA_PATH = "data/corruptmnist/corruptmnist_v1"

def corrupt_mnist() -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Return train and test dataloaders for corrupt MNIST."""
    train_images, train_target = [], []
    
    # Loop over the 6 training files
    for i in range(6):
        # We use f-strings to dynamically load train_images_0.pt, train_images_1.pt, etc.
        train_images.append(torch.load(f"{DATA_PATH}/train_images_{i}.pt"))
        train_target.append(torch.load(f"{DATA_PATH}/train_target_{i}.pt"))
    
    # Merge the list of tensors into one big tensor
    train_images = torch.cat(train_images)
    train_target = torch.cat(train_target)

    # Load the single test file
    test_images: torch.Tensor = torch.load(f"{DATA_PATH}/test_images.pt")
    test_target: torch.Tensor = torch.load(f"{DATA_PATH}/test_target.pt")

    # UNSQUEEZE: 
    # Current shape: [N, 28, 28] (Batch, Height, Width)
    # Required shape: [N, 1, 28, 28] (Batch, Channel, Height, Width) for CNNs
    train_images = train_images.unsqueeze(1).float()
    test_images = test_images.unsqueeze(1).float()
    
    train_target = train_target.long()
    test_target = test_target.long()

    # TensorDataset wraps tensors so we can easily iterate over them
    train_set = TensorDataset(train_images, train_target)
    test_set = TensorDataset(test_images, test_target)

    return train_set, test_set

def show_image_and_target(images: torch.Tensor, target: torch.Tensor) -> None:
    """Plot images and their labels in a grid."""
    # Calculate grid size (square root of number of images)
    row_col = int(len(images) ** 0.5)
    
    fig = plt.figure(figsize=(10.0, 10.0))
    grid = ImageGrid(fig, 111, nrows_ncols=(row_col, row_col), axes_pad=0.3)
    
    for ax, im, label in zip(grid, images, target):
        # We need .squeeze() to remove the channel dim we added earlier 
        # (1, 28, 28) -> (28, 28) for matplotlib
        ax.imshow(im.squeeze(), cmap="gray")
        ax.set_title(f"Label: {label.item()}")
        ax.axis("off")
    
    # Save instead of show, since you are on WSL/Server
    plt.savefig("data_inspection.png")
    print("Saved plot to data_inspection.png")

if __name__ == "__main__":
    train_set, test_set = corrupt_mnist()
    print(f"Size of training set: {len(train_set)}")
    print(f"Size of test set: {len(test_set)}")
    print(f"Shape of a training point {(train_set[0][0].shape, train_set[0][1].shape)}")
    print(f"Shape of a test point {(test_set[0][0].shape, test_set[0][1].shape)}")
    
    # Visualize the first 25 images
    show_image_and_target(train_set.tensors[0][:25], train_set.tensors[1][:25])