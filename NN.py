import numpy as np

class MyNN():
    def __init__(self, layer_sizes, learning_rate=0.01, task='c',activation='sigmoid', momentum=0.9,optimizer=None):
        """
        Initializes the neural network with random weights and biases.
        
        Args:
        layer_sizes (list): List containing the number of neurons in each layer.
        learning_rate (float): The learning rate for weight updates.
        task (str): Regression/Clasification : r/c
        optimizer (str): momentum or None
        activation (str): activation function name, default is sigmoid
        momentum (int):

        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.momentum = momentum
        
        self.task = task  
        self.optimizer = optimizer
        self.activation = activation
        
        self.weights = [np.random.randn(n_in, n_out) for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.biases = [np.random.randn(n_out) for n_out in layer_sizes[1:]]
        self.velocity_w = [np.zeros_like(w) for w in self.weights]
        self.velocity_b = [np.zeros_like(b) for b in self.biases]

    @staticmethod
    def sigmoid(x):
        x = np.clip(x, -500, 500) 
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return MyNN.sigmoid(x) * (1 - MyNN.sigmoid(x))

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        return (x > 0).astype(float)

    def activate(self, x):
        if self.activation == "sigmoid":
            return self.sigmoid(x)
        elif self.activation == "relu":
            return self.relu(x)
        else:
            return x  # fallback


    def forward_pass(self, inputs):
        """
        Perform a forward pass through the network.
        nutshell:
        activating layers from input to output(doing w1*x1+w2*x2 ... + b for each neuron in the layer 
        and going to the next layer. And z score in each iter is the weighted sums of each neuron in 2d vector)
        
        Args:
        inputs (ndarray)[[x1,x2]]: The input data.
        
        Returns:
        list: The list of activations for each layer.
        activations: final outputs of the layers(layer neurons)
        z_values: nonactivated outputs
        """
        activations = [inputs] # [ndarray([[x1,x2 ]])] 
        z_values = []  # Z values (inputs to the activation functions)
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(activations[-1], w) + b
            z_values.append(z)
            if i == len(self.weights) - 1 and self.task == 'r':
                activations.append(z)
            else:
                activations.append(self.activate(z))

        return activations, z_values

    def update_weights(self, grads_w, grads_b):
        """
        Update the weights and biases using the calculated gradients.
		nutshell:
		just updating the current weights and bias
        Args:
        grads_w (list): List of weight gradients.
        grads_b (list): List of bias gradients.
        """
        if self.optimizer is None:
            for i in range(len(self.weights)):
                self.weights[i] -= self.learning_rate * grads_w[i]
                self.biases[i] -= self.learning_rate * grads_b[i]
        else:
            for i in range(len(self.weights)):
                self.velocity_w[i] = self.momentum * self.velocity_w[i] + (1 - self.momentum) * grads_w[i]
                self.velocity_b[i] = self.momentum * self.velocity_b[i] + (1 - self.momentum) * grads_b[i]
                
                self.weights[i] -= self.learning_rate * self.velocity_w[i]
                self.biases[i] -= self.learning_rate * self.velocity_b[i]

    def backpropagate(self, inputs, true_output):
        """
        Perform backpropagation to calculate the gradients and update weights.
        
        Args:
        inputs (ndarray) [[x1,x2]]: The input data.
        true_output (ndarray) [[y]]: The expected output.
        
        Returns:
        tuple: Updated weights and biases.
        """
        activations, z_values = self.forward_pass(inputs)
        error = activations[-1] - true_output  # errors in each
        if self.task == 'r' and self.activation == 'linear':
            grad_output = error
        else:
            if self.activation == "sigmoid":
                grad_output = error * self.sigmoid_derivative(z_values[-1])
            elif self.activation == "relu":
                grad_output = error * self.relu_derivative(z_values[-1])
            else:
                grad_output = error  

        grads_w = []
        grads_b = []

        # Backpropagate the gradients for each layer
        for i in reversed(range(len(self.weights))): 
            grad_w = np.outer(activations[i], grad_output)
            grad_b = grad_output.flatten()
            grads_w.append(grad_w)
            grads_b.append(grad_b)
            
            if i > 0:
                if self.activation == "sigmoid":
                    grad_output = np.dot(grad_output, self.weights[i].T) * self.sigmoid_derivative(z_values[i-1])
                elif self.activation == "relu":
                    grad_output = np.dot(grad_output, self.weights[i].T) * self.relu_derivative(z_values[i-1])
                else:
                    grad_output = np.dot(grad_output, self.weights[i].T)
        grads_w = grads_w[::-1]
        grads_b = grads_b[::-1]
        self.update_weights(grads_w, grads_b)

    def train(self, X_train, y_train, epochs=1000):
        for epoch in range(epochs):
            total_loss = 0
            for x, y in zip(X_train, y_train): # [(x1,y1), (x2,y2)..]
                # Reshape the input data and target data
                x = x.reshape(1, -1) # ndarray(shape=(1,nfeatures)) [[x1,x2,]]
                y = y.reshape(-1, 1) # ndarray(shape=(1,1)), [[y]]

                # Perform backpropagation
                self.backpropagate(x, y)

                # Forward pass
                activations, _ = self.forward_pass(x)

                # Calculate loss (use MSE for regression, cross-entropy for classification)
                if self.task == 'r':
                    loss = np.mean((activations[-1] - y) ** 2)  # MSE loss for regression
                else:
                    # Binary cross-entropy loss for classification
                    loss = -np.mean(y * np.log(activations[-1]) + (1 - y) * np.log(1 - activations[-1]))

                total_loss += loss

            # Print loss every 100 epochs
            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {total_loss / len(X_train)}")

    def predict(self, X):
        # Make predictions for new data
        activations, _ = self.forward_pass(X.reshape(1, -1))
        return activations[-1]
