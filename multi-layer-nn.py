import torch
import torch.nn.functional as F
import torch.nn as nn

class ActivationFunction:
    def __init__(self):
        pass

    def __call__(self, x):
        raise NotImplementedError

    def derivative(self, x):
        raise NotImplementedError
    
class LeakyRelu(ActivationFunction):
    def __init__(self,negative_slope=0.01):
        self.negative_slope = negative_slope
    
    def __call__(self, x):
        return torch.where(x > 0, x, x * self.negative_slope)

    def derivative(self, x):
        dx = torch.ones_like(x)
        dx[x <= 0] = self.negative_slope
        return dx

class Softmax(ActivationFunction):
    def __init__(self,):
        super().__init__()
    
    def __call__(self, x):
        # Subtract max for numerical stability
        exp_score = torch.exp(x - torch.max(x, dim=1, keepdim=True).values)
        probs = exp_score / torch.sum(torch.exp(exp_score), dim=1, keepdim=True)
        return probs

    def derivative(self, x):
        raise NotImplementedError

class MultiLayerNet:
    def __init__(self,input_size, hidden_sizes, output_size):
        self.params = {}
        self.cache = {}
        self.reg_lambda = 0.001
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.num_layers = len(layer_sizes) - 1
        self.activation_function = LeakyRelu()
        

        for i in range(1,1+self.num_layers):
            self.params[f"W{i}"]=torch.randn(layer_sizes[i-1], layer_sizes[i], dtype=torch.float64)/torch.sqrt(torch.tensor(2.0 / layer_sizes[i-1]))
            self.params[f"b{i}"] = torch.zeros(layer_sizes[i], dtype=torch.float64)


    def forward(self,X):
        self.cache[f"a0"] = X

        for i in range(1, self.num_layers):
            z = torch.matmul(self.cache[f"a{i-1}"], self.params[f"W{i}"]) + self.params[f"b{i}"]
            self.cache[f"z{i}"] = z
            self.cache[f"a{i}"] = self.activation_function(z)
        
        # Final layer , no activation only softmax
        n = self.num_layers
        z_final = torch.matmul(self.cache[f"a{n-1}"], self.params[f"W{n}"]) + self.params[f"b{n}"]
        self.cache[f"z{n}"] = z_final
        probs = Softmax().__call__(z_final) # Softmax()(z_final)
        return probs

    def backward(self,y,y_hat):
        batch_size = y.shape[0]
        grads = {}
        if y.dim() == 1:
            y = F.one_hot(y.long(), num_classes=y_hat.shape[1]).to(torch.float64)

        dz = (y_hat - y) / batch_size

        for i in reversed(range(1,1+self.num_layers)):
            a_prev = self.cache[f"a{i-1}"]
            # Calculate weight gradients including L2 regularization
            grads[f"dW{i}"] = torch.matmul(a_prev.permute(1,0),dz)+self.reg_lambda*self.params[f"W{i}"]
            grads[f"db{i}"] = torch.sum(dz,dim=0)
            if i > 1:
                # Backpropagate error to previous layer
                da_prev = torch.matmul(dz,self.params[f"W{i}"].permute(1,0))
                # chain rule: multiply by activation derivative
                dz = self.activation_function.derivative(self.cache[f"z{i-1}"]) * da_prev
        
        return grads
    
    def loss(self,X,y):
        probs = self.forward(X)
        # Adding a tiny epsilon prevents log(0) which causes NaNs
        correct_logprobs = -torch.log(probs[range(len(X)), y.long()] + 1e-12)
        return torch.mean(correct_logprobs)

    
    def train(self,X,y,num_epochs,learning_rate=0.01):
        batch_size = 4
        num_batches = len(X) // batch_size
        for epoch in range(num_epochs):
            for _ in range(num_batches):
                batch_mask = torch.randint(high=len(X), size=(batch_size,))
                X_batch = X[batch_mask]
                y_batch = y[batch_mask]
                # Forward Propagation
                y_hat = self.forward(X_batch)
                # Backward Propagation
                grads = self.backward(y_batch,y_hat)
                # Update parameters
                for i in range(1, self.num_layers + 1):
                    self.params[f'W{i}'] -= learning_rate * grads[f'dW{i}']
                    self.params[f'b{i}'] -= learning_rate * grads[f'db{i}']
                
                # Print loss for monitoring training progress
                if epoch % 10 == 0:
                    loss = self.loss(X, y)
                    print(f"Epoch {epoch}, loss: {loss}")


# 1. Setup the Data (XOR Problem)
# Inputs: [0,0], [0,1], [1,0], [1,1]
# Targets: 0, 1, 1, 0
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float64)
y = torch.tensor([0, 1, 1, 0], dtype=torch.float64)

# 2. Initialize the Network
# input_size=2, hidden_sizes=[8], output_size=2
model = MultiLayerNet(input_size=2, hidden_sizes=[16], output_size=2)

# 3. Train the Model
print("Starting Training...")
# We use a slightly higher learning rate and more epochs for this tiny dataset
model.train(X, y, num_epochs=100, learning_rate=0.01)

# 4. Predict
print("\n--- Final Predictions ---")
with torch.no_grad():
    # Forward pass to get probabilities
    probs = model.forward(X)
    # Get the index of the highest probability (the class prediction)
    predictions = torch.argmax(probs, dim=1)

for i in range(len(X)):
    print(f"Input: {X[i].tolist()} -> Target: {int(y[i])} -> Predicted: {predictions[i].item()}")

# Check accuracy
accuracy = (predictions == y).float().mean()
print(f"\nModel Accuracy: {accuracy * 100}%")