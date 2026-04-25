import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoLayerNN:
    def __init__(self,input_size, hidden_size, output_size) -> None:
        self.parameters = {}
        self.grads = {}
        self.parameters["W1"] = torch.randn(input_size,hidden_size,dtype=torch.float64)
        self.parameters["b1"] = torch.zeros(hidden_size,dtype=torch.float64)
        self.parameters["W2"] = torch.randn(hidden_size,output_size,dtype=torch.float64)
        self.parameters["b2"] = torch.zeros(output_size,dtype=torch.float64)
    
    def forward(self,X_ip):
        z1_I0_H = torch.matmul(X_ip,self.parameters["W1"])+self.parameters["b1"] # (4,2)X(2,6)
        a1_I0_H =  F.leaky_relu(z1_I0_H, negative_slope=0.01)
        z2_H_O = torch.matmul(a1_I0_H,self.parameters["W2"])+self.parameters["b2"] # (4,6)X(6,2)
        exp_score_H_O = torch.exp(z2_H_O)
        probs_H_O = exp_score_H_O / torch.sum(exp_score_H_O,axis=1,keepdims=True)
        self.grads["y_hat"] = probs_H_O # (4,2)
        self.grads["a1"] = a1_I0_H # (4,6)
        self.grads["z1"] = z1_I0_H # (4,6)
        return probs_H_O
    
    def backward(self,X_ip,y_ip):
        # 0. One-hot encode y_ip if it isn't already [batch, output_size]
        # Assuming y_ip is passed as [0, 1, 1, 0], we convert it:
        if y_ip.dim() == 1: # broadcast (1,4) -> (4,2)
            y_ip = F.one_hot(y_ip.to(torch.int64), num_classes=self.parameters["W2"].shape[1]).to(torch.float64)
        # dz2: [4, 2] (y_hat - y)    
        dz2_I0_O = self.grads["y_hat"] - y_ip 
        # [6, 2] = [6, 4] @ [4, 2]
        dw2_H_O = torch.matmul(self.grads["a1"].permute(1,0),dz2_I0_O) 
        # db2: [2] (Sum across batch)
        db2_O = torch.sum(dz2_I0_O,axis=0)
        # da1: [4, 6] = [4, 2] @ [2, 6]
        da1_I0_H = torch.matmul(dz2_I0_O,self.parameters["W2"].permute(1,0))
        # dz1 : [4,6] (Derivative of Leaky ReLU)
        # We multiply element-wise by the slope
        dz1_mask = torch.ones_like(self.grads["z1"])
        # print(f"Before setting mask : ",dz1_mask)
        dz1_mask[self.grads["z1"] <= 0] = 0.01
        # print(f"After setting mask : ",dz1_mask)
        dz1_I0_H = da1_I0_H * dz1_mask
        # dw1: [2,6] = [2,4] @ [4,6]
        dw1_I1_H = torch.matmul(X_ip.permute(1,0),dz1_I0_H) 
        # db1: [6] (Sum across batch)
        db1_H = torch.sum(dz1_I0_H,axis=0)
        self.grads["dW2"] = dw2_H_O
        self.grads["db2"] = db2_O
        self.grads["dW1"] = dw1_I1_H
        self.grads["db1"] = db1_H
        return self.grads
    """
    The Indexing: probs_H_O[range(4), [0, 1, 1, 0]] 
    picks the specific probability the model assigned to the true label 
    for each sample.
        -If Sample 1 is Class 0, it picks probs[0, 0].
        -If Sample 2 is Class 1, it picks probs[1, 1].

    The Negative Log: We take the -log(x)
    If the model is 100% confident in the correct class 
       - (prob = 1.0), -log(1.0) = 0 (Zero loss).
    If the model is very uncertain on whether this is correct class 
       - (prob = 0.01), -log(0.01) ~ 4.6 (High loss).
    """
    def loss(self,X_ip,y_ip):
        probs_H_O = self.forward(X_ip)
        correct_logprobs = -torch.log(probs_H_O[range(len(X_ip)), y_ip.long()])
        data_loss = torch.sum(correct_logprobs)
        return 1.0/len(X_ip) * data_loss # Normalize : avg loss per sample

    def train(self, X, y, num_epochs, learning_rate=0.1):
        for epoch in range(num_epochs):
            probs = self.forward(X)
            grads = self.backward(X,y)
            # Update parameters
            self.parameters['W1'] -= learning_rate * self.grads["dW1"]
            self.parameters['b1'] -= learning_rate * self.grads["db1"]
            self.parameters['W2'] -= learning_rate * self.grads["dW2"]
            self.parameters['b2'] -= learning_rate * self.grads["db2"]
            # Print loss for monitoring training progress
            if epoch % 10 == 0:
                loss = self.loss(X, y)
                print(f"Epoch {epoch}: loss = {loss}")

    
model = TwoLayerNN(input_size=2, hidden_size=6, output_size=2)
X_ip =  torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]],dtype=torch.float64) # (4,2)
y_ip = torch.tensor([0, 1, 1, 0],dtype=torch.float64) # (1,4)
print(f"X -> {X_ip.shape}")
print(f"y -> {y_ip.shape}")
probs = model.forward(X_ip) 
print(f"Forward pass: probs(y_hat) = {probs.shape}\n")
predictions = torch.argmax(probs, axis=1)
print("Predictions: ", predictions)
grads_backwd = model.backward(X_ip,y_ip)
print("Backward \n")
for key,val in grads_backwd.items():
    print(f"{key}  -> {val.shape}")

model.train(X_ip, y_ip, num_epochs=100)
print("==========================")
pred_probs = model.forward(X_ip) 
predictions = torch.argmax(pred_probs, axis=1)
print("Predictions: ", predictions)


