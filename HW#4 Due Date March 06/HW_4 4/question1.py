
# output of file: Output: [0.49313988]





import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

class SimpleNeuralNetwork:
    def __init__(self, layer_sizes, activation=sigmoid):
        """
        layer_sizes: list of integers specifying neurons per layer.
                     e.g. [2, 2, 1] means:
                       - 2 inputs
                       - 2 neurons in hidden layer
                       - 1 neuron in output layer
        """
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.num_layers = len(layer_sizes)
        
        # Random initialization of weights & biases for demonstration
        # (You can set them to your known values or train them.)
        self.weights = []
        self.biases = []
        
        # For each layer (except input), create weights and biases
        for i in range(1, self.num_layers):
            in_size = layer_sizes[i-1]
            out_size = layer_sizes[i]
            
            # Randomly initialize
            w = np.random.randn(out_size, in_size) * 0.1
            b = np.zeros((out_size, 1))
            
            self.weights.append(w)
            self.biases.append(b)

    def forward_pass(self, x):
        """
        x is a 1D list or array of length layer_sizes[0].
        Returns the output of the network.
        """
        # Convert x to column vector for matrix ops
        a = np.array(x).reshape(-1, 1)
        
        # Propagate through each layer
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            # Hidden layers
            z = np.dot(w, a) + b
            a = self.activation(z)
        
        # For the output layer (last weight/bias), often same or different activation
        w_out, b_out = self.weights[-1], self.biases[-1]
        z_out = np.dot(w_out, a) + b_out
        output = self.activation(z_out)
        
        return output.flatten()  # Return as 1D array

if __name__ == "__main__":
    # Example usage
    # Create a 2 -> 2 -> 1 network
    nn = SimpleNeuralNetwork(layer_sizes=[2, 2, 1])
    
    # Input
    x = [1.0, 2.0]
    
    # Forward pass
    output = nn.forward_pass(x)
    print("Output:", output)
