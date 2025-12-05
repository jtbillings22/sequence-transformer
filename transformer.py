import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# ============================================
# 1. Normalization: mean/std per sequence
# ============================================

def normalize_sequences(data):
    """Normalize each sequence to zero mean, unit std"""
    mn = data.mean(axis=1, keepdims=True)
    std = data.std(axis=1, keepdims=True)
    std[std == 0] = 1.0  # avoid division by zero
    norm_data = (data - mn) / std
    return norm_data.astype(np.float32), mn.flatten(), std.flatten()

def denormalize(value, mn, std):
    return value * std + mn

# ============================================
# 2. Load sequences from dataset file
# ============================================

def load_sequences_from_txt(filename: str):
    """Load all sequences as a 2D numpy array and normalize"""
    data = np.loadtxt(filename, delimiter=",", dtype=np.float32)
    norm_data, mn, std = normalize_sequences(data)
    return norm_data, mn, std

# ============================================
# 3. Build training windows (predict deltas)
# ============================================

def create_training_samples(sequences, seq_len=30):
    """
    Converts sequences into sliding windows of length seq_len
    Targets are differences: delta = x[t+1] - x[t]
    """
    X_list, Y_list = [], []

    for seq in sequences:
        if len(seq) <= seq_len:
            continue
        windows = np.lib.stride_tricks.sliding_window_view(seq, window_shape=seq_len)
        X_list.append(windows[:-1])
        deltas = seq[seq_len:] - seq[seq_len - 1:-1]  # delta between next value and last window element
        Y_list.append(deltas)

    X = np.concatenate(X_list, axis=0)
    Y = np.concatenate(Y_list, axis=0)

    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

# ============================================
# 4. Transformer model
# ============================================

class TransformerRegressor(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, 500, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, 1)  # Linear output, no sigmoid

    def forward(self, x):
        x = x.unsqueeze(-1)  # (batch, seq_len, 1)
        x = self.input_proj(x)
        x = x + self.pos_embedding[:, :x.size(1), :]
        h = self.transformer(x)
        last = h[:, -1, :]
        return self.fc_out(last).squeeze(-1)

# ============================================
# 5. Training
# ============================================

def train_model(X, Y, batch_size=2048, epochs=15, lr=1e-4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    dataset = TensorDataset(X, Y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)

    model = TransformerRegressor().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            optimizer.zero_grad()
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)

        avg = total_loss / len(dataset)
        print(f"Epoch {epoch}/{epochs} - Loss: {avg:.6f}")

    return model

# ============================================
# 6. Multi-step prediction (autoregressive using deltas)
# ============================================

def predict_future(model, initial_seq, steps, mn, std, device):
    model.eval()
    seq = initial_seq.copy()
    preds = []

    with torch.no_grad():
        seq_len = len(initial_seq)
        for _ in range(steps):
            x = torch.tensor(seq[-seq_len:], dtype=torch.float32).unsqueeze(0).to(device)
            delta = model(x).cpu().item()
            next_val = seq[-1] + delta
            preds.append(denormalize(next_val, mn, std))
            seq = np.append(seq, next_val)  # autoregressive step in normalized space

    return preds

# ============================================
# 7. Main
# ============================================

if __name__ == "__main__":
    sequences, mn_list, std_list = load_sequences_from_txt("seq_dataset.txt")
    print(f"Loaded {len(sequences)} sequences")

    seq_len = 30
    X, Y = create_training_samples(sequences, seq_len=seq_len)
    print(f"Training samples: {len(X)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = train_model(X, Y, batch_size=2048, epochs=15, lr=1e-4)

    print("\n==================== PREDICTIONS ====================\n")

    for idx, seq in enumerate(sequences):
        mn, std = mn_list[idx], std_list[idx]
        initial_seq = seq[:seq_len]
        future_steps = 5
        preds = predict_future(model, initial_seq, future_steps, mn, std, device)

        print(f"Sequence {idx + 1}:")
        print(f"Initial sequence: {denormalize(initial_seq, mn, std)}")
        print(f"Predicted next {future_steps} steps: {np.round(preds, 4)}")
        print("-" * 50)
