import torch
import matplotlib.pyplot as plt

# 1. Load data (assuming you already have the file paths set)
file_path = "data/corruptmnist/corruptmnist_v1/test_images.pt"
target_path = "data/corruptmnist/corruptmnist_v1/test_target.pt"
images = torch.load(file_path)
targets = torch.load(target_path)

# 2. Setup a 2-row, 5-column grid
# Increased height to 6 so images aren't squashed
figure, axes = plt.subplots(2, 5, figsize=(15, 6)) 
axes = axes.flatten() # <--- CRITICAL FIX: flattens the 2x5 grid into a 1D list of 10 plots

for i in range(10):
    # We skip the first 10 to see new examples
    img = images[i]
    label = targets[i].item()
    
    # Handle dimensions (same as before)
    if img.ndim == 1:
        img = img.view(28, 28)
    elif img.ndim == 3:
        img = img.squeeze(0)
        
    # Plot
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(f"Label: {label}")
    axes[i].axis('off')

plt.tight_layout()
plt.savefig("corruption_check_10.png")
print("Saved 10 examples to corruption_check_10.png")