import numpy as np
import matplotlib.pyplot as plt

class SnappingSynapseNeuralNetwork:
    def __init__(self):
        # Seed the random number generator for consistency
        np.random.seed(1)
        # Initialize synaptic weights with random values between -1 and 1
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1
        # Store the history of weight changes for visualization
        self.weights_history = [self.synaptic_weights.flatten().copy()]

    def __sigmoid(self, x):
        """The sigmoid activation function, which normalizes the output to a value between 0 and 1."""
        return 1 / (1 + np.exp(-x))

    def __sigmoid_derivative(self, x):
        """The derivative of the sigmoid function, used to calculate the weight adjustment."""
        return x * (1 - x)

    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations, initial_learning_rate=1.0, decay_rate=0.001):
        """
        Train the neural network with a decaying learning rate to simulate the 'snapping' effect.
        """
        learning_rate_history = []
        for iteration in range(number_of_training_iterations):
            # Forward pass: calculate the network's output
            output = self.think(training_set_inputs)

            # Calculate the error (the difference between the desired output and the predicted output)
            error = training_set_outputs - output

            # Calculate the dynamic learning rate for this iteration
            learning_rate = initial_learning_rate / (1 + decay_rate * iteration)
            learning_rate_history.append(learning_rate)

            # Calculate the weight adjustment, scaled by the current learning rate
            adjustment = np.dot(training_set_inputs.T, error * self.__sigmoid_derivative(output)) * learning_rate

            # Update the synaptic weights
            self.synaptic_weights += adjustment
            self.weights_history.append(self.synaptic_weights.flatten().copy())

        # Plot the learning rate decay and weight changes
        self.plot_training_dynamics(learning_rate_history)


    def think(self, inputs):
        """The 'thinking' process of the neural network (forward propagation)."""
        return self.__sigmoid(np.dot(inputs, self.synaptic_weights))

    def plot_training_dynamics(self, learning_rate_history):
        """Visualize the learning rate decay and the evolution of synaptic weights."""
        plt.figure(figsize=(12, 5))

        # Plot learning rate decay
        plt.subplot(1, 2, 1)
        plt.plot(learning_rate_history)
        plt.title("Learning Rate Decay")
        plt.xlabel("Training Iteration")
        plt.ylabel("Learning Rate")
        plt.grid(True)

        # Plot synaptic weight changes
        weights_history = np.array(self.weights_history)
        plt.subplot(1, 2, 2)
        for i in range(weights_history.shape[1]):
            plt.plot(weights_history[:, i], label=f'Weight {i+1}')
        plt.title("Synaptic Weight Changes Over Time")
        plt.xlabel("Training Iteration")
        plt.ylabel("Weight Value")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Initialize the neural network
    neural_network = SnappingSynapseNeuralNetwork()

    print("Random starting synaptic weights: ")
    print(neural_network.synaptic_weights)

    # The training set: 4 examples with 3 input features and 1 output
    training_set_inputs = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_outputs = np.array([[0, 1, 1, 0]]).T

    # Train the neural network for 10,000 iterations
    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    print("\nNew synaptic weights after training: ")
    print(neural_network.synaptic_weights)

    # Test the neural network with a new situation
    print("\nConsidering new situation [1, 0, 0] -> ?: ")
    print(neural_network.think(np.array([1, 0, 0])))