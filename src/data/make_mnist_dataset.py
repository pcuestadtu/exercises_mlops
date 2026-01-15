import torch
from torch.utils.data import Dataset
import glob
import os

class CorruptMnist(Dataset):
    def __init__(self, data_dir, train=True):
        """
        Args:
            data_dir (str): Path to the folder containing the .pt files (e.g., data/corruptmnist/corruptmnist_v1)
            train (bool): If True, loads training data. If False, loads test data.
        """
        self.data_dir = data_dir
        self.train = train
        
        # 1. Select the file prefix based on the mode
        # If train=True, we look for 'train_images_*.pt'
        # If train=False, we look for 'test_images.pt'
        prefix = "train" if self.train else "test"
        
        # 2. Find the files using 'glob'
        # glob helps us find files using wildcards (*).
        # ideally this finds [train_images_0.pt, train_images_1.pt, ...] OR [test_images.pt]
        image_pattern = os.path.join(data_dir, f"{prefix}_images*.pt")
        target_pattern = os.path.join(data_dir, f"{prefix}_target*.pt")
        
        image_files = glob.glob(image_pattern)
        target_files = glob.glob(target_pattern)
        
        # Sort files to ensure images_0 matches target_0
        image_files.sort()
        target_files.sort()
        
        # Check if we actually found files
        if len(image_files) == 0:
            raise FileNotFoundError(f"No files found for pattern: {image_pattern}")

        # 3. Load and Merge
        # We loop through the list of files, load them, and add them to a list
        images_list = []
        targets_list = []
        
        for img_file, tgt_file in zip(image_files, target_files):
            print(f"Loading {img_file}...")
            images_list.append(torch.load(img_file))
            targets_list.append(torch.load(tgt_file))
            
        # 4. Concatenate (Merge)
        # We squash the list of tensors into one giant tensor.
        # Training: 6 files of 5000 images -> 1 tensor of 30,000 images
        # Test: 1 file of 5000 images -> 1 tensor of 5000 images
        self.data = torch.cat(images_list, dim=0)
        self.targets = torch.cat(targets_list, dim=0)
        
        # 5. Dimension Check (Best Practice)
        # PyTorch Convolutional layers expect input as [Batch, Channels, Height, Width]
        # Sometimes saved data is just [Batch, H, W] (missing the channel dim).
        # If it is 3D, we add the channel dimension (unsqueeze).
        if self.data.ndim == 3:
            self.data = self.data.unsqueeze(1) 

    def __len__(self):
        # This allows you to call len(dataset)
        return len(self.targets)

    def __getitem__(self, idx):
        # This allows you to do dataset[0] to get the first image
        # It returns a tuple: (image, label)
        return self.data[idx], self.targets[idx]

if __name__ == "__main__":
    # A quick test to verify it works for both Train and Test
    path = "data/corruptmnist/corruptmnist_v1" # <--- Update this if your path is different
    
    print("--- Checking Training Data ---")
    train_set = CorruptMnist(data_dir=path, train=True)
    print(f"Train Size: {len(train_set)}")
    print(f"Train Shape: {train_set.data.shape}")
    
    print("\n--- Checking Test Data ---")
    test_set = CorruptMnist(data_dir=path, train=False)
    print(f"Test Size: {len(test_set)}")
    print(f"Test Shape: {test_set.data.shape}")