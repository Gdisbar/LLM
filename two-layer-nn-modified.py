import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoLayerNN:
    def __init__(self,input_size, hidden_size, output_size) -> None:
        self.parameters = {}
        self.grads = {}
        # Xavier initialization
        self.parameters["W1"] = torch.randn(input_size,hidden_size,dtype=torch.float64)/torch.sqrt(torch.tensor(input_size,dtype=torch.float64))
        self.parameters["W2"] = torch.randn(hidden_size,output_size,dtype=torch.float64)/torch.sqrt(torch.tensor(hidden_size,dtype=torch.float64))
        self.parameters["b1"] = torch.zeros(hidden_size,dtype=torch.float64)
        self.parameters["b2"] = torch.zeros(output_size,dtype=torch.float64)
    
    def forward(self,X_ip):
        z1_I0_H = torch.matmul(X_ip,self.parameters["W1"])+self.parameters["b1"] # (4,2)X(2,6)
        a1_I0_H =  F.leaky_relu(z1_I0_H, negative_slope=0.01)
        z2_H_O = torch.matmul(a1_I0_H,self.parameters["W2"])+self.parameters["b2"] # (4,6)X(6,2)
        # --- Numerical Stability Fix ---
        # Subtract max for stability: e^(x - max)
        shift_z2 = z2_H_O - torch.max(z2_H_O, dim=1, keepdim=True).values
        exp_score_H_O = torch.exp(shift_z2)
        probs_H_O = exp_score_H_O / torch.sum(exp_score_H_O, dim=1, keepdim=True)
        # self.grads["y_hat"] = probs_H_O # (4,2)
        self.grads["a1"] = a1_I0_H # (4,6)
        self.grads["z1"] = z1_I0_H # (4,6)
        return probs_H_O
    
    def backward(self,y_hat,X_ip,y_ip,reg_lambda):
        # 0. One-hot encode y_ip if it isn't already [batch, output_size]
        # Assuming y_ip is passed as [0, 1, 1, 0], we convert it:
        if y_ip.dim() == 1: # broadcast (1,4) -> (4,2)
            y_ip = F.one_hot(y_ip.to(torch.int64), num_classes=self.parameters["W2"].shape[1]).to(torch.float64)
        # dz2: [4, 2] (y_hat - y)    
        dz2_I0_O = y_hat - y_ip 
        # [6, 2] = [6, 4] @ [4, 2]
        dw2_H_O = torch.matmul(self.grads["a1"].permute(1,0),dz2_I0_O) 
        # db2: [2] (Sum across batch)
        db2_O = torch.sum(dz2_I0_O,dim=0)
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
        db1_H = torch.sum(dz1_I0_H,dim=0)
        # avoid exploding gradient problem
        num_samples = X_ip.shape[0]
        # Add the derivative of 0.5 * reg_lambda * W^2, which is (reg_lambda * W)
        self.grads["dW2"] = (dw2_H_O / num_samples) + (reg_lambda * self.parameters["W2"])
        self.grads["db2"] = db2_O / num_samples
        self.grads["dW1"] = (dw1_I1_H / num_samples) + (reg_lambda * self.parameters["W1"])
        self.grads["db1"] = db1_H / num_samples
        return self.grads

    def loss(self,X_ip,y_ip,reg_lambda):
        probs_H_O = self.forward(X_ip)
        correct_logprobs = -torch.log(probs_H_O[range(len(X_ip)), y_ip.long()])
        data_loss = torch.sum(correct_logprobs)
        # L2 regularization
        data_loss += 0.5 * reg_lambda * (torch.sum(self.parameters['W1'] ** 2) + torch.sum(self.parameters['W2'] ** 2))
        return 1.0/len(X_ip) * data_loss # Normalize : avg loss per sample

    def train(self, X, y, num_epochs, learning_rate,reg_lambda):
        # Mini-batch training
        batch_size = 4
        num_batches = len(X) // batch_size
        # Adam optimization
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8
        mW1, vW1 = 0, 0
        mW2, vW2 = 0, 0
        t = 0 # Timestep for bias correction
        for epoch in range(num_epochs):
            for _ in range(num_batches):
                t+=1
                # Select a random batch of data with replacement
                batch_mask = torch.randint(high=len(X), size=(batch_size,))
                # batch_mask = torch.randperm(len(X))[:batch_size] # without replacement
                X_batch = X[batch_mask]
                y_batch = y[batch_mask]
                # probs_H_O
                y_hat_H_O = self.forward(X_batch)
                grads = self.backward(y_hat_H_O,X_batch,y_batch,reg_lambda)
                # Update loop for all parameters using Adam logic
                for param_name in ['W1', 'W2', 'b1', 'b2']:
                    # This is a simplified manual Adam loop
                    if param_name == 'W1':
                        mW1 = beta1 * mW1 + (1 - beta1) * grads["dW1"]
                        vW1 = beta2 * vW1 + (1 - beta2) * (grads["dW1"]**2)
                        m_corr = mW1 / (1 - beta1**t)
                        v_corr = vW1 / (1 - beta2**t)
                        self.parameters['W1'] -= learning_rate * m_corr / (torch.sqrt(v_corr) + eps)
                    elif param_name == 'W2':
                        mW2 = beta1 * mW2 + (1 - beta1) * grads["dW2"]
                        vW2 = beta2 * vW2 + (1 - beta2) * (grads["dW2"]**2)
                        m_corr = mW2 / (1 - beta1**t)
                        v_corr = vW2 / (1 - beta2**t)
                        self.parameters['W2'] -= learning_rate * m_corr / (torch.sqrt(v_corr) + eps)
                    # Note: Biases usually use Adam too for consistency
                    else:
                        self.parameters[param_name] -= learning_rate * grads["d" + param_name]
                # Print loss for monitoring training progress
                if epoch % 10 == 0:
                    loss = self.loss(X, y,reg_lambda)
                    print(f"Epoch {epoch}: loss = {loss}")

    
model = TwoLayerNN(input_size=2, hidden_size=6, output_size=2)
X_ip =  torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]],dtype=torch.float64) # (4,2)
y_ip = torch.tensor([0, 1, 1, 0],dtype=torch.float64) # (1,4)
print(f"X -> {X_ip.shape}")
print(f"y -> {y_ip.shape}")
probs_H_O = model.forward(X_ip) 
print(f"Forward pass: probs(y_hat) = {probs_H_O.shape}\n")
grads_backwd = model.backward(probs_H_O,X_ip,y_ip,reg_lambda = 0.1)
print("Backward pass:")
for key,val in grads_backwd.items():
    print(f"{key}  -> {val.shape}")

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
model = TwoLayerNN(input_size=2, hidden_size=6, output_size=2)
model.train(X_ip, y_ip, num_epochs=100,learning_rate=0.01,reg_lambda=0.001)
print("==========================")
pred_probs = model.forward(X_ip) 
predictions = torch.argmax(pred_probs, dim=1)
print("Predictions: ", predictions)

