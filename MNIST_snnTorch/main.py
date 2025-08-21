# Code B: Spiking Neural Network (snnTorch)

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# --- 1. Data Loading and Preparation ---
def get_mnist_loaders(batch_size=128):
    """Creates DataLoaders for the MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    # Use a subset for faster training to match the NumPy version
    train_subset = torch.utils.data.Subset(train_dataset, range(10000))
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    return train_loader


# --- 2. SNN Architecture ---
class SpikingMNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        num_inputs = 784
        num_hidden = 64
        num_outputs = 10
        beta = 0.95  # High beta means slow decay of membrane potential

        # Use a surrogate gradient for backpropagation through spikes
        spike_grad = surrogate.fast_sigmoid()

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_inputs, num_hidden),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.Linear(num_hidden, num_outputs),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
        )

    def forward(self, x):
        # Reset neuron states at the beginning of each forward pass
        for layer in self.net:
            if hasattr(layer, 'reset_hidden'):
                layer.reset_hidden()

        # The output is the final membrane potential of the output neurons
        _, mem_out = self.net(x)
        return mem_out


# --- 3. Training and Visualization ---
def train_and_visualize_snntorch():
    train_loader = get_mnist_loaders()
    net = SpikingMNISTNet().to("cpu")  # Use CPU for simplicity

    # Loss and Optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    num_epochs = 10  # SNNs often require fewer epochs

    # History for plotting
    loss_history = []
    accuracy_history = []

    print("--- Training Spiking Neural Network (snnTorch) ---")
    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0

        for data, targets in train_loader:
            data = data.to("cpu")
            targets = targets.to("cpu")

            # Forward pass
            mem_out = net(data)

            # Calculate loss
            loss = loss_fn(mem_out, targets)

            # Backward pass and weight update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Record metrics
            epoch_loss += loss.item()
            _, predicted = torch.max(mem_out.data, 1)
            epoch_total += targets.size(0)
            epoch_correct += (predicted == targets).sum().item()

        avg_loss = epoch_loss / len(train_loader)
        avg_acc = epoch_correct / epoch_total
        loss_history.append(avg_loss)
        accuracy_history.append(avg_acc)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Code B: Spiking Neural Network (snnTorch)", fontsize=16)

    ax1.plot(loss_history, color='orange')
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True)

    ax2.plot(accuracy_history, color='green')
    ax2.set_title("Training Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True)

    plt.show()


if __name__ == "__main__":
    train_and_visualize_snntorch()