import torch
from torch import nn

class MyAwesomeModel(nn.Module):
    """My awesome model."""

    def __init__(self) -> None:
        super().__init__()
        # Input: [Batch, 1, 28, 28]
        # Conv1: 28x28 -> (kernel 3) -> 26x26
        self.conv1 = nn.Conv2d(1, 32, 3, 1) 
        
        # Input: [Batch, 32, 13, 13] (after maxpool 26->13)
        # Conv2: 13x13 -> (kernel 3) -> 11x11
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        
        # Input: [Batch, 64, 5, 5] (after maxpool 11->5)
        # Conv3: 5x5 -> (kernel 3) -> 3x3
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        
        self.dropout = nn.Dropout(0.5)
        
        # After final maxpool (3->1), we have [Batch, 128, 1, 1]
        # Flattening gives 128 features.
        self.fc1 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Block 1
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2, 2)
        
        # Block 2
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2, 2)
        
        # Block 3
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2, 2)
        
        # Head
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc1(x)

if __name__ == "__main__":
    model = MyAwesomeModel()
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")