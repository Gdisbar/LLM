import torch

def rnn_forward(input_sequence: list, initial_hidden_state: list, Wx: list, Wh: list, b: list) -> torch.Tensor:
    """
    Implements a simple RNN cell forward pass using PyTorch.

    Args:
        input_sequence: List of input vectors for each time step.
        initial_hidden_state: The initial hidden state vector.
        Wx: Weight matrix for input-to-hidden connections.
        Wh: Weight matrix for hidden-to-hidden connections.
        b: Bias vector.

    Returns:
        torch.Tensor: The final hidden state after processing the entire sequence,
                      rounded to four decimal places.
    """
    Wx = torch.tensor(Wx, dtype=torch.float32)
    Wh = torch.tensor(Wh, dtype=torch.float32)
    b = torch.tensor(b, dtype=torch.float32)
    # input_sequence is (seq_len, input_size)
    inputs = torch.tensor(input_sequence, dtype=torch.float32)
    # h current starts as the initial_hidden_state
    h_t = torch.tensor(initial_hidden_state, dtype=torch.float32)
    # Iterate through each time step (The RNN Loop)
    for x_t in inputs:
        # x_t is the input at the current time step
        # h_t = tanh(Wx @ x_t + Wh @ h_t + b)
        z = torch.matmul(Wx,x_t)+torch.matmul(Wh,h_t)+b
        h_t = torch.tanh(z)

    return torch.round(h_t, decimals=4)


print(rnn_forward([[1.0], [2.0], [3.0]], # 1X3
                  [0.0], 
                  [[0.5]], 
                  [[0.8]], 
                  [0.0]))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class SimpleRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Initializes the RNN with random weights and zero biases using PyTorch modules.
        """
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size

        # Use PyTorch built-in RNNCell with tanh activation
        self.rnn_cell = nn.RNNCell(input_size, hidden_size, nonlinearity='tanh')
        # Use PyTorch built-in Linear for output projection
        self.fc = nn.Linear(hidden_size, output_size)

        # Initialize weights with small random values (std=0.01) and zero biases
        nn.init.normal_(self.rnn_cell.weight_ih, mean=0.0, std=0.01)
        nn.init.normal_(self.rnn_cell.weight_hh, mean=0.0, std=0.01)
        nn.init.zeros_(self.rnn_cell.bias_ih)
        nn.init.zeros_(self.rnn_cell.bias_hh)
        nn.init.normal_(self.fc.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.fc.bias)

        self.optimizer = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the RNN for a given sequence of inputs.
        Returns tensor of shape (seq_len, 1, output_size) to match original.
        """
        x = torch.as_tensor(x, dtype=torch.float32)
        # Initialize hidden state
        h = torch.zeros(1, self.hidden_size, dtype=torch.float32)
        outputs = []

        for t in range(x.shape[0]):
            x_t = x[t].unsqueeze(0)  # shape: (1, input_size)
            h = self.rnn_cell(x_t, h)  # shape: (1, hidden_size)
            y_t = self.fc(h)           # shape: (1, output_size)
            outputs.append(y_t.unsqueeze(0))  # shape: (1, 1, output_size)

        return torch.cat(outputs, dim=0)  # shape: (seq_len, 1, output_size)

    def backward(self, x: torch.Tensor, y: torch.Tensor, learning_rate: float) -> None:
        """
        Backpropagation through time to adjust weights based on error gradient.
        Uses SGD optimizer and 1/2 MSE loss summed over time steps.
        """
        x = torch.as_tensor(x, dtype=torch.float32)
        y = torch.as_tensor(y, dtype=torch.float32)

        # Initialize or update optimizer
        if self.optimizer is None:
            self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)
        else:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = learning_rate

        self.optimizer.zero_grad()

        # Forward pass
        outputs = self.forward(x)  # shape: (seq_len, 1, output_size)

        # Reshape y to match output: (seq_len, 1, output_size)
        y_reshaped = y.unsqueeze(1)

        # 1/2 * MSE loss summed over time steps (matching original behavior)
        # F.mse_loss with reduction='sum' gives sum of squared errors
        # Multiply by 0.5 for 1/2 MSE
        loss = 0.5 * F.mse_loss(outputs, y_reshaped, reduction='sum')

        # Backpropagate
        loss.backward()
        self.optimizer.step()

    def train_manual(self, x: torch.Tensor, y: torch.Tensor, learning_rate: float) -> None:
        # x: (seq_len, input_size) -> (4, 1)
        # y: (seq_len, output_size) -> (4, 1)

        # 1. Forward Pass with Caching
        h_states = [torch.zeros(1, self.hidden_size)] # h_0: (1, 5)
        y_hats = []
        curr_h = h_states[0]
        for x_t in x:
            curr_h = self.rnn_cell(x_t.unsqueeze(0), curr_h) # (1, 1) -> (1, 5)
            h_states.append(curr_h)
            y_hats.append(self.fc(curr_h)) # (1, 5) -> (1, 1)

        # 2. Initialize Gradients (Gradients match parameter shapes)
        d_w_ih = torch.zeros_like(self.rnn_cell.weight_ih) # (5, 1)
        d_w_hh = torch.zeros_like(self.rnn_cell.weight_hh) # (5, 5)
        d_b_ih = torch.zeros_like(self.rnn_cell.bias_ih)   # (5,)
        d_b_hh = torch.zeros_like(self.rnn_cell.bias_hh)   # (5,)
        d_fc_w = torch.zeros_like(self.fc.weight)         # (1, 5)
        d_fc_b = torch.zeros_like(self.fc.bias)           # (1,)
        
        # Error flowing from step t+1 to t
        dh_next = torch.zeros(1, self.hidden_size)        # (1, 5)

        # 3. Backward Loop
        for t in reversed(range(len(x))):
            # dy = (y_hat - y) -> shape: (1, 1)
            dy = y_hats[t] - y[t].view_as(y_hats[t])
            
            # d_fc_w = dy^T @ h_t -> (1, 1) @ (1, 5) = (1, 5)
            d_fc_w += dy.t() @ h_states[t+1]
            d_fc_b += dy.squeeze(0)

            # dh = dy @ W_fc + dh_next -> (1, 1) @ (1, 5) + (1, 5) = (1, 5)
            dh = (dy @ self.fc.weight) + dh_next
            
            # dtanh = dh * (1 - h^2) -> (1, 5) element-wise
            dtanh = dh * (1 - h_states[t+1]**2)

            # d_w_ih = dtanh^T @ x_t -> (5, 1) @ (1, 1) = (5, 1)
            d_w_ih += dtanh.t() @ x[t].unsqueeze(0)
            
            # d_w_hh = dtanh^T @ h_{t-1} -> (5, 1) @ (1, 5) = (5, 5)
            d_w_hh += dtanh.t() @ h_states[t]
            
            d_b_ih += dtanh.squeeze(0)
            d_b_hh += dtanh.squeeze(0)

            # dh_next = dtanh @ W_hh -> (1, 5) @ (5, 5) = (1, 5)
            dh_next = dtanh @ self.rnn_cell.weight_hh

        # 4. Manual Update
        with torch.no_grad():
            self.rnn_cell.weight_ih -= learning_rate * d_w_ih
            self.rnn_cell.weight_hh -= learning_rate * d_w_hh
            self.rnn_cell.bias_ih -= learning_rate * d_b_ih
            self.rnn_cell.bias_hh -= learning_rate * d_b_hh
            self.fc.weight -= learning_rate * d_fc_w
            self.fc.bias -= learning_rate * d_fc_b
 
import torch 
torch.manual_seed(42) 
input_sequence = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32) 
expected_output = torch.tensor([[2.0], [3.0], [4.0], [5.0]], dtype=torch.float32) 

rnn = SimpleRNN(input_size=1, hidden_size=5, output_size=1) 
for epoch in range(100): 
    rnn.backward(input_sequence, expected_output, learning_rate=0.01) 
    output = rnn.forward(input_sequence) 

print(output.detach().numpy().tolist())