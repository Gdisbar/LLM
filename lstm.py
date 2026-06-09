import torch
import torch.nn as nn
import numpy as np
# import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --------------------------
# 1. Generate synthetic time series data (sine wave)
# --------------------------
def create_sequence_data(seq_len=50, n_samples=1000):
    """Generate x, y pairs: x is a window of sine values, y is the next value."""
    x = np.linspace(0, 100, n_samples + seq_len)
    y_full = np.sin(x)  # pure sine wave

    X, Y = [], []
    for i in range(len(y_full) - seq_len):
        X.append(y_full[i:i+seq_len])          # input sequence
        Y.append(y_full[i+seq_len])            # target: next value
    X = np.array(X, dtype=np.float32)          # shape: (n_samples, seq_len)
    Y = np.array(Y, dtype=np.float32)          # shape: (n_samples,)
    # Add feature dimension: (n_samples, seq_len, input_size)
    X = X[..., np.newaxis]                     # input_size = 1
    return X, Y

seq_len = 50
X, Y = create_sequence_data(seq_len=seq_len, n_samples=1000)

# Train-test split
split = int(0.8 * len(X))
X_train, Y_train = X[:split], Y[:split]
X_test,  Y_test  = X[split:], Y[split:]

# Convert to PyTorch tensors and create DataLoaders
train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
test_dataset  = TensorDataset(torch.from_numpy(X_test),  torch.from_numpy(Y_test))
train_loader  = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader   = DataLoader(test_dataset,  batch_size=32, shuffle=False)

# --------------------------
# 2. Define the LSTM model
# --------------------------
class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # LSTM layer: batch_first=True so input is (batch, seq_len, input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        # Fully connected layer to map last hidden state to output
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM forward pass
        out, (hn, cn) = self.lstm(x, (h0, c0))   # out: (batch, seq_len, hidden_size)
        # We take the output at the last time step (many-to-one)
        out = out[:, -1, :]                       # shape: (batch, hidden_size)
        out = self.fc(out)                        # shape: (batch, 1)
        return out.squeeze(-1)                    # (batch,)

model = LSTMForecaster(input_size=1, hidden_size=64, num_layers=2, dropout=0.2).to(device)

# --------------------------
# 3. Training setup
# --------------------------
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 30
train_losses = []
test_losses = []

print("Starting training...")
for epoch in range(1, num_epochs + 1):
    model.train()
    epoch_train_loss = 0.0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        preds = model(batch_x)
        loss = criterion(preds, batch_y)
        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item() * batch_x.size(0)

    # Validation
    model.eval()
    epoch_test_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            epoch_test_loss += loss.item() * batch_x.size(0)

    train_loss = epoch_train_loss / len(train_loader.dataset)
    test_loss  = epoch_test_loss  / len(test_loader.dataset)
    train_losses.append(train_loss)
    test_losses.append(test_loss)

    if epoch % 5 == 0:
        print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f}")

# --------------------------
# 4. Plot predictions vs actual (first 100 test points)
# --------------------------
model.eval()
with torch.no_grad():
    X_test_tensor = torch.from_numpy(X_test[:200]).to(device)
    preds_test = model(X_test_tensor).cpu().numpy()

# plt.figure(figsize=(12,4))
# plt.plot(Y_test[:200], label='Actual')
# plt.plot(preds_test,   label='Predicted')
# plt.legend()
# plt.title("LSTM Time Series Forecasting - Test Set")
# plt.show()

# # Loss curve
# plt.figure(figsize=(8,4))
# plt.plot(train_losses, label='Train')
# plt.plot(test_losses,  label='Test')
# plt.xlabel('Epoch')
# plt.ylabel('MSE Loss')
# plt.legend()
# plt.show()