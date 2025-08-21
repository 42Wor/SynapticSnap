# Code A: Traditional Neural Network with Step Decay Learning Rate

import numpy as np
import matplotlib.pyplot as plt


class AdvancedScheduledNetwork:
    def __init__(self):
        np.random.seed(1)
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1
        self.weights_history = [self.synaptic_weights.flatten().copy()]

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        """
        Trains the network using a Step Decay learning rate schedule.
        """
        learning_rate = 1.0  # Initial learning rate
        learning_rate_history = []

        # --- Step Decay Parameters ---
        # We will drop the learning rate at these iteration points.
        decay_schedule = {200: 0.1, 1500: 0.01, 5000: 0.001}

        for iteration in range(number_of_training_iterations):
            # Check if the current iteration is a point to decay the learning rate
            if iteration in decay_schedule:
                learning_rate = decay_schedule[iteration]
                print(f"\n--- Iteration {iteration}: Learning rate dropped to {learning_rate} ---\n")

            # Forward pass
            output = self.think(training_set_inputs)
            error = training_set_outputs - output

            # Backward pass (weight adjustment)
            adjustment = np.dot(training_set_inputs.T, error * self.__sigmoid_derivative(output)) * learning_rate
            self.synaptic_weights += adjustment

            # Store history for plotting
            self.weights_history.append(self.synaptic_weights.flatten().copy())
            learning_rate_history.append(learning_rate)

        self.plot_training_dynamics(learning_rate_history)

    def think(self, inputs):
        return self.__sigmoid(np.dot(inputs, self.synaptic_weights))

    def plot_training_dynamics(self, learning_rate_history):
        """Visualizes the learning dynamics."""
        plt.figure(figsize=(12, 5))
        plt.suptitle("Code A: Traditional Network with Step Decay", fontsize=16)

        # Plot 1: Learning Rate Decay
        plt.subplot(1, 2, 1)
        plt.plot(learning_rate_history, color='r')
        plt.title("Learning Rate (Step Decay)")
        plt.xlabel("Training Iteration")
        plt.ylabel("Learning Rate")
        plt.grid(True)

        # Plot 2: Synaptic Weight Changes
        weights_history = np.array(self.weights_history)
        plt.subplot(1, 2, 2)
        for i in range(weights_history.shape[1]):
            plt.plot(weights_history[:, i], label=f'Weight {i + 1}')
        plt.title("Synaptic Weight Changes")
        plt.xlabel("Training Iteration")
        plt.ylabel("Weight Value")
        plt.legend()
        plt.grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()


if __name__ == "__main__":
    # Initialize the network
    neural_network = AdvancedScheduledNetwork()

    # The training data
    training_set_inputs = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_outputs = np.array([[0, 1, 1, 0]]).T

    # Train the network
    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    # Test with a new situation
    print("\nTesting network with [1, 0, 0] -> ?:")
    print(neural_network.think(np.array([1, 0, 0])))