# Code B: Spiking Neural Network with snnTorch

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Data Preparation ---
# SNNs process data over time. We need to convert our static data into a temporal sequence.
# We will represent each input feature as a spike train over several time steps.
time_steps = 25

# Original data
X_data = torch.tensor([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]], dtype=torch.float)
y_data = torch.tensor([[0], [1], [1], [0]], dtype=torch.float)

# Convert data to a temporal format [num_samples, time_steps, num_features]
# We will simply repeat the input for each time step.
train_data = X_data.unsqueeze(1).repeat(1, time_steps, 1)
train_labels = y_data


# --- 2. Define the SNN Architecture ---
# We use Leaky Integrate-and-Fire (LIF) neurons, which are common in SNNs.
class SNN(nn.Module):
    def __init__(self):
        super().__init__()
        num_inputs = 3
        num_hidden = 10
        num_outputs = 1
        beta = 0.9  # Membrane potential decay rate

        # Surrogate gradient for backpropagation
        spike_grad = surrogate.fast_sigmoid()

        # Network layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x):
        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Record the output spikes over time
        spk2_recording = []

        # Loop over each time step
        for step in range(time_steps):
            cur1 = self.fc1(x[:, step, :])  # Process one time step of data
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_recording.append(spk2)

        return torch.stack(spk2_recording, dim=1)


# --- 3. Training the SNN ---
net = SNN()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

num_epochs = 200
loss_history = []

print("--- Training SNN with snnTorch ---")
for epoch in range(num_epochs):
    # Forward pass
    spk_out = net(train_data)

    # The network's prediction is the total number of spikes at the output
    prediction = torch.sum(spk_out, dim=1) / time_steps  # Normalize by time

    # Calculate loss
    loss = loss_fn(prediction, train_labels)
    loss_history.append(loss.item())

    # Zero gradients, backward pass, update weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# --- 4. Visualization ---
plt.figure(figsize=(8, 5))
plt.plot(loss_history)
plt.title("Code B: SNN Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.grid(True)
plt.show()

# --- 5. Testing the SNN ---
test_input = torch.tensor([[1, 0, 0]], dtype=torch.float).unsqueeze(1).repeat(1, time_steps, 1)
with torch.no_grad():
    test_spikes = net(test_input)
    prediction = torch.sum(test_spikes, dim=1) / time_steps
    print(f"\nTesting SNN with [1, 0, 0] -> ?:")
    print(f"Output spike probability: {prediction.item():.2f}")
    print(f"Predicted class: {torch.round(prediction).item()}")