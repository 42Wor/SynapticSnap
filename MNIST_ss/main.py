# Code A: Advanced Traditional Neural Network (NumPy)

import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets
import torchvision.transforms as transforms
import torch

# --- 1. Data Loading and Preparation (using torchvision, converting to NumPy) ---
def get_mnist_data_numpy(batch_size=128):
    """Loads MNIST data and converts it to NumPy arrays for our network."""
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)

    # Use a subset of 10,000 samples to match the SNN example for a fair comparison
    subset_indices = list(range(10000))
    train_subset = torch.utils.data.Subset(train_dataset, subset_indices)

    # Create a loader for the subset
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=len(train_subset))

    # Extract all data and convert to NumPy
    images, labels = next(iter(train_loader))

    # Flatten images and normalize pixel values to be between 0 and 1
    X_train = images.view(images.shape[0], -1).numpy()

    # One-hot encode the labels
    y_train = np.zeros((labels.shape[0], 10))
    y_train[np.arange(labels.shape[0]), labels] = 1

    return X_train, y_train


# --- 2. Neural Network Components (NumPy) ---
def relu(Z):
    """ReLU activation function."""
    return np.maximum(0, Z)


def relu_derivative(Z):
    """Derivative of ReLU for backpropagation."""
    return Z > 0


def softmax(Z):
    """Softmax activation for the output layer."""
    expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return expZ / np.sum(expZ, axis=1, keepdims=True)


def cross_entropy_loss(A2, Y):
    """Cross-entropy loss function."""
    m = Y.shape[0]
    log_likelihood = -np.log(A2[np.arange(m), Y.argmax(axis=1)])
    loss = np.sum(log_likelihood) / m
    return loss


def get_accuracy(A2, Y):
    """Calculates accuracy from predictions."""
    predictions = np.argmax(A2, axis=1)
    labels = np.argmax(Y, axis=1)
    return np.sum(predictions == labels) / Y.shape[0]


# --- 3. Training and Visualization ---
def train_and_visualize_numpy():
    # Load data
    X, Y = get_mnist_data_numpy()
    num_samples, num_inputs = X.shape

    # Network architecture
    num_hidden = 64
    num_outputs = 10

    # Initialize weights and biases
    np.random.seed(1)
    W1 = np.random.randn(num_inputs, num_hidden) * 0.01
    b1 = np.zeros((1, num_hidden))
    W2 = np.random.randn(num_hidden, num_outputs) * 0.01
    b2 = np.zeros((1, num_outputs))

    # Hyperparameters
    learning_rate = 1e-1  # Start with a relatively high learning rate
    num_epochs = 10
    batch_size = 128

    # Advanced Learning Rate Schedule: Step Decay
    # At epoch 5, drop LR by a factor of 10. At epoch 8, drop again.
    step_decay_schedule = {5: 1e-2, 8: 1e-3}

    # History for plotting
    loss_history = []
    accuracy_history = []

    print("--- Training Traditional Neural Network (NumPy) ---")
    for epoch in range(num_epochs):
        # Apply step decay
        if epoch in step_decay_schedule:
            learning_rate = step_decay_schedule[epoch]
            print(f"--- Epoch {epoch}: Learning rate dropped to {learning_rate} ---")

        # Shuffle data for each epoch
        permutation = np.random.permutation(num_samples)
        X_shuffled = X[permutation]
        Y_shuffled = Y[permutation]

        for i in range(0, num_samples, batch_size):
            # Get mini-batch
            X_batch = X_shuffled[i:i + batch_size]
            Y_batch = Y_shuffled[i:i + batch_size]
            m_batch = X_batch.shape[0]

            # --- Forward Pass ---
            Z1 = X_batch.dot(W1) + b1
            A1 = relu(Z1)
            Z2 = A1.dot(W2) + b2
            A2 = softmax(Z2)  # Final predictions (probabilities)

            # --- Backward Pass (Backpropagation) ---
            dZ2 = A2 - Y_batch
            dW2 = (1 / m_batch) * A1.T.dot(dZ2)
            db2 = (1 / m_batch) * np.sum(dZ2, axis=0, keepdims=True)

            dA1 = dZ2.dot(W2.T)
            dZ1 = dA1 * relu_derivative(Z1)
            dW1 = (1 / m_batch) * X_batch.T.dot(dZ1)
            db1 = (1 / m_batch) * np.sum(dZ1, axis=0, keepdims=True)

            # --- Update Weights and Biases ---
            W1 -= learning_rate * dW1
            b1 -= learning_rate * db1
            W2 -= learning_rate * dW2
            b2 -= learning_rate * db2

        # Calculate loss and accuracy for the whole dataset at the end of the epoch
        Z1 = X.dot(W1) + b1
        A1 = relu(Z1)
        Z2 = A1.dot(W2) + b2
        A2 = softmax(Z2)

        loss = cross_entropy_loss(A2, Y)
        accuracy = get_accuracy(A2, Y)
        loss_history.append(loss)
        accuracy_history.append(accuracy)
        print(f"Epoch {epoch + 1}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Code A: Traditional Neural Network (NumPy)", fontsize=16)

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
    train_and_visualize_numpy()