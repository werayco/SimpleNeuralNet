import numpy as np
# lets build a simple neural network with numpy, having 2 i/p, 3 hidden layer and 1 o/p layer

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize biases and weights
        self.w1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.random.randn(self.hidden_size, 1)
        self.w2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.random.randn(self.output_size, 1)

    
    def relu(self,x):
        return np.maximum(0, x)
    
    def relu_derivative(self,x):
        return np.where(x <= 0, 0, 1)

    def forward_propagation(self, X):
        # Input to first hidden layer
        self.z2 = np.dot(X, self.w1) + self.b1
        self.a2 = self.relu(self.z2)

        # First hidden to final output
        z3 = np.dot(self.a2, self.w2) + self.b2
        a3 = self.relu(z3)

        return a3

    def back_propagation(self, X, y):
        output = self.forward_propagation(X)

        dEdA = -1 * (y - output)
        dA3dZ3 = self.relu_derivative(output)
        dZ3dA2 = self.w2
        dA2dZ2 = self.relu_derivative(self.z2)

        #gradients for W1
        delta_one = np.multiply(dEdA, dA3dZ3)
        dEdW11 = np.dot(delta_one, np.transpose(dZ3dA2))
        dEdW111 = np.dot(dA2dZ2, dEdW11)
        dEdW1 = np.dot(X.T, dEdW111)

        #gradients for W2
        delta_two = np.multiply(dEdA, dA3dZ3)
        dEdW2 = np.dot(np.transpose(delta_two), self.a2)

        #fine-tunning weights and biases using gradient descent
        self.w1 = self.w1 - self.learning_rate * dEdW1
        self.w2 = self.w2 - self.learning_rate * dEdW2
        self.b2 = self.b2 - self.learning_rate*delta_two

        # lets make some calculations for b1
        first=np.transpose(np.dot(dZ3dA2.T,dA2dZ2))
        delta1=np.multiply(delta_one,first)
        self.b1 = self.b1 - self.learning_rate*delta1

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            self.back_propagation(X, y) 
            if epoch % 10000 == 0:
                error = np.mean(np.square(y - self.forward_propagation(X)))
                print(f'Error at epoch {epoch}: {error}')


input_data = np.array([[3, 6], [4, 1], [2, 8]])
output_data = np.array([[0], [1], [1]])
# Standardization
mean = np.mean(input_data)
std_dev = np.std(input_data)
standardized_data = (input_data - mean) / std_dev


nn = NeuralNetwork(input_size=2, hidden_size=3, output_size=1, learning_rate=0.1) # just keep changing the learning rate and observe abeg
nn.train(standardized_data, output_data, epochs=10000)
