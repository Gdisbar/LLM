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
 
import torch 
torch.manual_seed(42) 
input_sequence = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32) 
expected_output = torch.tensor([[2.0], [3.0], [4.0], [5.0]], dtype=torch.float32) 

rnn = SimpleRNN(input_size=1, hidden_size=5, output_size=1) 
for epoch in range(100): 
    rnn.backward(input_sequence, expected_output, learning_rate=0.01) 
    output = rnn.forward(input_sequence) 

print(output.detach().numpy().tolist())