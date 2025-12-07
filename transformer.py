import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

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

def load_sequences_from_txt(filename: str, types=None, return_types: bool = False):
    """
    Load sequences as a 2D numpy array and normalize.
    If `types` is provided, only lines whose leading label is in that list are used, allows us to test train on smaller subsets.
    """
    selected = set(types) if types else None
    sequences = []
    seq_types = []

    with open(filename, "r") as f:
        for line in f:
            if not line:
                continue
            seq_type, sep, numeric_part = line.partition(",")
            if not sep:
                continue
            if selected and seq_type not in selected:
                continue
            values = np.fromstring(numeric_part.strip(), sep=",", dtype=np.float32)
            if values.size:
                sequences.append(values)
                seq_types.append(seq_type)

    if not sequences:
        empty = (
            np.empty((0, 0), dtype=np.float32),
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
        )
        if return_types:
            return (*empty, [])
        return empty

    data = np.vstack(sequences)
    norm_data, mn, std = normalize_sequences(data)
    if return_types:
        return norm_data, mn, std, seq_types
    return norm_data, mn, std

# ============================================
# 3. Build training windows (predict deltas)
# ============================================

def create_training_samples(sequences, seq_len=30, types=None, type_to_idx=None):
    """
    Converts sequences into sliding windows of length seq_len
    Targets are differences: delta = x[t+1] - x[t]
    """
    X_list, Y_list, T_list = [], [], []

    for seq_idx, seq in enumerate(sequences):
        if len(seq) <= seq_len:
            continue
        windows = np.lib.stride_tricks.sliding_window_view(seq, window_shape=seq_len)
        X_list.append(windows[:-1])
        deltas = seq[seq_len:] - seq[seq_len - 1:-1]  # delta between next value and last window element
        Y_list.append(deltas)
        if types is not None:
            if type_to_idx is None:
                raise ValueError("type_to_idx mapping is required when types are provided.")
            type_id = type_to_idx[types[seq_idx]]
            T_list.append(np.full_like(deltas, fill_value=type_id, dtype=np.int64))

    X = np.concatenate(X_list, axis=0)
    Y = np.concatenate(Y_list, axis=0)

    type_tensor = None
    if T_list:
        type_tensor = torch.tensor(np.concatenate(T_list, axis=0), dtype=torch.long)

    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32), type_tensor

# ============================================
# 4. Transformer model
# ============================================

class Transformer(nn.Module):
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

def train_model(X, Y, type_ids, type_names, batch_size=2048, epochs=15, lr=1e-4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    dataset = TensorDataset(X, Y, type_ids)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)

    model = Transformer().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    type_history = {name: [] for name in type_names} # used to build error per type graph
    num_types = len(type_names)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        per_type_sum = [0.0] * num_types
        per_type_count = [0] * num_types

        for batch_x, batch_y, batch_type in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            batch_type = batch_type.to(device, non_blocking=True)

            optimizer.zero_grad()
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)

            # accumulate per-type squared error
            squared_error = (pred - batch_y) ** 2
            for t_idx in range(num_types):
                mask = batch_type == t_idx
                count = mask.sum().item()
                if count:
                    per_type_sum[t_idx] += squared_error[mask].sum().item()
                    per_type_count[t_idx] += count

        avg = total_loss / len(dataset)
        per_type_mse = []
        for t_idx in range(num_types):
            if per_type_count[t_idx]:
                per_type_mse.append(per_type_sum[t_idx] / per_type_count[t_idx])
            else:
                per_type_mse.append(float("nan"))
        for name, mse in zip(type_names, per_type_mse):
            type_history[name].append(mse)

        per_type_str = " | ".join(f"{name}: {mse:.6f}" if not np.isnan(mse) else f"{name}: nan" for name, mse in zip(type_names, per_type_mse))
        print(f"Epoch {epoch}/{epochs} - Loss: {avg:.6f} | Per-type MSE -> {per_type_str}")

    return model, type_history


def plot_type_error_history(history, output_path="error_history.png"):
    if plt is None:
        print("matplotlib not available; skipping per-type error plot.")
        return
    plt.figure(figsize=(8, 6))
    epochs = range(1, len(next(iter(history.values()))) + 1)
    for type_name, values in history.items():
        plt.plot(epochs, values, label=type_name)
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Per-type Training Error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved per-type error plot to: {output_path}")

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

#TODO: so i updated the input so we can select a subset of sequences from the seq_dataset, update this main so we can choose which sequences to run as input

if __name__ == "__main__":

    # types included to train on
    selected_types = [
        "constant",
        "linear",
        "logarithmic",
        "prime",
        "collatz",
        "arithmetic",
        "fibonacci",
        "noisy_sinusoidal",
        "quadratic",
    ]

    # (normalized sequences, mean, standard deviation)
    sequences, mn_list, std_list, seq_types = load_sequences_from_txt("seq_dataset.txt", selected_types, return_types=True) 
    print(f"Loaded {len(sequences)} sequences")
    type_names = list(selected_types)
    type_to_idx = {name: idx for idx, name in enumerate(type_names)}

    seq_len = 30
    X, Y, type_ids = create_training_samples(sequences, seq_len=seq_len, types=seq_types, type_to_idx=type_to_idx)
    print(f"Training samples: {len(X)}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, type_error_history = train_model(X, Y, type_ids, type_names, batch_size=64, epochs=10, lr=1e-4)

    plot_type_error_history(type_error_history)

    print("\n==================== PREDICTIONS ====================\n")

    max_predictions = 50
    for idx, seq in enumerate(sequences[:max_predictions]):
        mn, std = mn_list[idx], std_list[idx]
        initial_seq = seq[:seq_len]
        future_steps = 5
        preds = predict_future(model, initial_seq, future_steps, mn, std, device)

        print(f"Sequence {idx + 1}:")
        print(f"Initial sequence: {denormalize(initial_seq, mn, std)}")
        print(f"Predicted next {future_steps} steps: {np.round(preds, 4)}")
        print("-" * 50)
    
