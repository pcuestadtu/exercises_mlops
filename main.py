import matplotlib.pyplot as plt
import torch
import pandas
import numpy
print("hola Pau")

print("adeu Pau")
import typer
# IMPORTANT: Adjusted imports to match your folder structure
from src.data.data import corrupt_mnist
from src.models.awesome_model import MyAwesomeModel

# Setup standard device check (GPU vs Mac MPS vs CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

app = typer.Typer()

@app.command()
def train(lr: float = 1e-3, batch_size: int = 32, epochs: int = 10) -> None:
    """Train a model on Corrupt MNIST."""
    print("Training day and night")
    print(f"{lr=}, {batch_size=}, {epochs=}")

    # 1. Initialize Model and Data
    model = MyAwesomeModel().to(DEVICE)
    train_set, _ = corrupt_mnist()
    
    # DataLoader handles batching automatically
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    statistics = {"train_loss": [], "train_accuracy": []}
    
    # 2. Training Loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)
            
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()
            
            # Tracking stats
            statistics["train_loss"].append(loss.item())
            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            statistics["train_accuracy"].append(accuracy)
            
            running_loss += loss.item()

            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item():.4f}")
        
        print(f"Epoch {epoch} finished. Avg Loss: {running_loss / len(train_dataloader):.4f}")

    print("Training complete")
    
    # 3. Save Model
    # We use state_dict as discussed before
    torch.save(model.state_dict(), "trained_model.pt")
    print("Model saved to trained_model.pt")
    
    # 4. Save Plot
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
    fig.savefig("training_statistics.png")
    print("Plot saved to training_statistics.png")


@app.command()
def evaluate(model_checkpoint: str) -> None:
    """Evaluate a trained model."""
    print("Evaluating like my life depended on it")
    print(f"Loading checkpoint: {model_checkpoint}")

    model = MyAwesomeModel().to(DEVICE)
    # Load weights
    model.load_state_dict(torch.load(model_checkpoint, map_location=DEVICE))

    _, test_set = corrupt_mnist()
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=32)

    model.eval()
    correct, total = 0, 0
    
    with torch.no_grad():
        for img, target in test_dataloader:
            img, target = img.to(DEVICE), target.to(DEVICE)
            y_pred = model(img)
            correct += (y_pred.argmax(dim=1) == target).float().sum().item()
            total += target.size(0)
            
    print(f"Test accuracy: {correct / total:.2%}")


if __name__ == "__main__":
    app()
