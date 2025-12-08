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

def normalize_sequences(data, seq_types=None):
    """Normalize each sequence to zero mean, unit std, with log transform for exponential/geometric"""
    normalized = []
    mn_list = []
    std_list = []
    transform_list = []
    is_constant_list = []
    is_arithmetic_list = []
    arithmetic_diff_list = []
    is_fibonacci_list = []
    fibonacci_ratio_list = []
    
    for i, seq in enumerate(data):
        # Check if sequence is constant (skip first element which might be an ID)
        seq_values = seq[1:] if len(seq) > 1 else seq
        seq_std = seq_values.std()
        is_constant = seq_std < 1e-6
        
        # Check if sequence is arithmetic (constant differences)
        is_arithmetic = False
        arithmetic_diff = 0.0
        if len(seq_values) > 2:
            diffs = np.diff(seq_values)
            diff_std = diffs.std()
            if diff_std < 1e-4:  # differences are nearly constant
                is_arithmetic = True
                arithmetic_diff = diffs.mean()
        
        # Check if sequence is Fibonacci-like (constant ratio between consecutive differences)
        is_fibonacci = False
        fibonacci_ratio = 0.0
        if seq_types and i < len(seq_types):
            if seq_types[i] == 'fibonacci' and len(seq_values) > 3:
                # For Fibonacci, each term is sum of previous two: F(n) = F(n-1) + F(n-2)
                # This means F(n)/F(n-1) approaches golden ratio
                ratios = []
                for j in range(len(seq_values) - 1):
                    if abs(seq_values[j]) > 1e-6:  # avoid division by zero
                        ratios.append(seq_values[j+1] / seq_values[j])
                if len(ratios) > 5:
                    # Check if ratios are converging (Fibonacci property)
                    ratio_std = np.std(ratios[-10:])  # look at last 10 ratios
                    if ratio_std < 0.1:  # ratios are relatively stable
                        is_fibonacci = True
                        fibonacci_ratio = np.mean(ratios[-10:])
        
        # Detect if sequence should use log transform (exponential/geometric/fibonacci)
        use_log = False
        if seq_types and i < len(seq_types):
            seq_type = seq_types[i]
            if seq_type in ['exponential', 'geometric', 'fibonacci']:
                use_log = True
                # Add small epsilon to handle zeros and ensure positivity
                seq = np.abs(seq) + 1e-8
                seq = np.log(seq)
        
        mn = seq.mean()
        std = seq.std()
        if std == 0 or is_constant:
            std = 1.0
        norm_seq = (seq - mn) / std
        
        normalized.append(norm_seq.astype(np.float32))
        mn_list.append(mn)
        std_list.append(std)
        transform_list.append('log' if use_log else 'linear')
        is_constant_list.append(is_constant)
        is_arithmetic_list.append(is_arithmetic)
        arithmetic_diff_list.append(arithmetic_diff)
        is_fibonacci_list.append(is_fibonacci)
        fibonacci_ratio_list.append(fibonacci_ratio)
    
    return (np.array(normalized), np.array(mn_list), np.array(std_list), 
            transform_list, is_constant_list, is_arithmetic_list, arithmetic_diff_list,
            is_fibonacci_list, fibonacci_ratio_list)

def denormalize(value, mn, std, transform='linear'):
    """Denormalize value, applying inverse log if needed"""
    denorm = value * std + mn
    if transform == 'log':
        denorm = np.exp(denorm)
    return denorm

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
            return (*empty, [], [], [], [], [], [], [])
        return empty

    result = normalize_sequences(sequences, seq_types)
    norm_data, mn, std, transforms, is_constant, is_arithmetic, arithmetic_diff, is_fibonacci, fibonacci_ratio = result
    if return_types:
        return norm_data, mn, std, seq_types, transforms, is_constant, is_arithmetic, arithmetic_diff, is_fibonacci, fibonacci_ratio
    return norm_data, mn, std, transforms, is_constant, is_arithmetic, arithmetic_diff, is_fibonacci, fibonacci_ratio

# ============================================
# 3. Build training windows (predict deltas + ratios)
# ============================================

def create_training_samples(sequences, seq_len=30, types=None, type_to_idx=None):
    """
    Converts sequences into sliding windows of length seq_len
    Targets are differences: delta = x[t+1] - x[t]
    Also includes ratio information for better multiplicative pattern learning
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
# 4. Transformer model with sequence type embeddings
# ============================================

class Transformer(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=6, num_types=11):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.type_embedding = nn.Embedding(num_types, d_model)
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

    def forward(self, x, type_ids):
        x = x.unsqueeze(-1)  # (batch, seq_len, 1)
        x = self.input_proj(x)
        
        # Add type embedding to the entire sequence
        type_emb = self.type_embedding(type_ids).unsqueeze(1)  # (batch, 1, d_model)
        x = x + type_emb
        
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

    model = Transformer(num_types=len(type_names)).to(device)
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
            pred = model(batch_x, batch_type)
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

def predict_future(model, initial_seq, steps, mn, std, device, transform='linear', is_constant=False, 
                   is_arithmetic=False, arithmetic_diff=0.0, is_fibonacci=False, fibonacci_ratio=0.0,
                   seq_type_id=0, original_value=None, seq_type=None):
    model.eval()
    
    # Handle constant sequences - just return the constant value
    if is_constant:
        if original_value is None:
            # Use the last value from initial sequence (more reliable than first)
            original_value = denormalize(initial_seq[-1], mn, std, transform)
        return [original_value] * steps
    
    # Handle arithmetic sequences - use the exact difference
    if is_arithmetic:
        last_value = denormalize(initial_seq[-1], mn, std, transform)
        preds = []
        for i in range(1, steps + 1):
            preds.append(last_value + arithmetic_diff * i)
        return preds
    
    # Handle Fibonacci-like sequences - use the ratio pattern
    if is_fibonacci and fibonacci_ratio > 0:
        # Get last two values in original space
        last_val = denormalize(initial_seq[-1], mn, std, transform)
        second_last_val = denormalize(initial_seq[-2], mn, std, transform)
        
        preds = []
        prev_prev = second_last_val
        prev = last_val
        
        for _ in range(steps):
            # Fibonacci: F(n) = F(n-1) + F(n-2)
            next_val = prev + prev_prev  # Classic Fibonacci rule
            preds.append(next_val)
            prev_prev = prev
            prev = next_val
        
        return preds
    
    # Handle Collatz sequences - use the Collatz rule
    if seq_type == 'collatz':
        last_val = denormalize(initial_seq[-1], mn, std, transform)
        preds = []
        current = last_val
        
        for _ in range(steps):
            if current <= 1:
                # Sequence terminates at 1
                preds.append(1.0)
            elif abs(current - round(current)) < 0.01:  # essentially an integer
                current = round(current)
                if current % 2 == 0:
                    next_val = current / 2
                else:
                    next_val = 3 * current + 1
                preds.append(next_val)
                current = next_val
            else:
                # Non-integer, use model
                break
        
        # If we didn't complete all predictions with the rule, use the model for the rest
        if len(preds) < steps:
            seq = initial_seq.copy()
            type_tensor = torch.tensor([seq_type_id], dtype=torch.long).to(device)
            
            with torch.no_grad():
                seq_len = len(initial_seq)
                for _ in range(steps - len(preds)):
                    x = torch.tensor(seq[-seq_len:], dtype=torch.float32).unsqueeze(0).to(device)
                    delta = model(x, type_tensor).cpu().item()
                    next_val = seq[-1] + delta
                    preds.append(denormalize(next_val, mn, std, transform))
                    seq = np.append(seq, next_val)
        
        return preds
    
    # Handle prime sequences - use statistical patterns
    if seq_type == 'prime':
        last_val = denormalize(initial_seq[-1], mn, std, transform)
        second_last = denormalize(initial_seq[-2], mn, std, transform)
        
        # Estimate average gap from recent differences
        recent_vals = [denormalize(initial_seq[i], mn, std, transform) for i in range(-10, 0)]
        recent_gaps = np.diff(recent_vals)
        avg_gap = np.mean(recent_gaps)
        
        preds = []
        current = last_val
        
        for _ in range(steps):
            # Use average gap as heuristic, with slight increase (prime gaps tend to grow)
            next_val = current + avg_gap * 1.05
            preds.append(next_val)
            current = next_val
            avg_gap *= 1.02  # gaps slowly increase for larger primes
        
        return preds
    
    seq = initial_seq.copy()
    preds = []

    with torch.no_grad():
        seq_len = len(initial_seq)
        type_tensor = torch.tensor([seq_type_id], dtype=torch.long).to(device)
        
        for _ in range(steps):
            x = torch.tensor(seq[-seq_len:], dtype=torch.float32).unsqueeze(0).to(device)
            delta = model(x, type_tensor).cpu().item()
            next_val = seq[-1] + delta
            preds.append(denormalize(next_val, mn, std, transform))
            seq = np.append(seq, next_val)  # autoregressive step in normalized space

    return preds

# ============================================
# 7. Main
# ============================================

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
        "exponential",
        "geometric",
    ]

    # (normalized sequences, mean, standard deviation)
    result = load_sequences_from_txt("seq_dataset.txt", selected_types, return_types=True)
    sequences, mn_list, std_list, seq_types, transforms, is_constant_list, is_arithmetic_list, arithmetic_diff_list, is_fibonacci_list, fibonacci_ratio_list = result 
    print(f"Loaded {len(sequences)} sequences")
    type_names = list(selected_types)
    type_to_idx = {name: idx for idx, name in enumerate(type_names)}

    seq_len = 30
    X, Y, type_ids = create_training_samples(sequences, seq_len=seq_len, types=seq_types, type_to_idx=type_to_idx)
    print(f"Training samples: {len(X)}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, type_error_history = train_model(X, Y, type_ids, type_names, batch_size=256, epochs=10, lr=1e-4)

    plot_type_error_history(type_error_history)

    print("\n==================== PREDICTIONS ====================\n")

    max_predictions = 50
    for idx, seq in enumerate(sequences[:max_predictions]):
        mn, std, transform, is_constant = mn_list[idx], std_list[idx], transforms[idx], is_constant_list[idx]
        is_arithmetic, arithmetic_diff = is_arithmetic_list[idx], arithmetic_diff_list[idx]
        is_fibonacci, fibonacci_ratio = is_fibonacci_list[idx], fibonacci_ratio_list[idx]
        seq_type = seq_types[idx]
        seq_type_id = type_to_idx[seq_type]
        
        initial_seq = seq[:seq_len]
        future_steps = 5
        
        # Get original value for constant sequences (use a value from the actual sequence, not the first element)
        original_value = None
        if is_constant:
            # Get the constant value from the unnormalized space - use last value of initial sequence
            original_value = denormalize(initial_seq[-1], mn, std, transform)
        
        preds = predict_future(model, initial_seq, future_steps, mn, std, device, transform, 
                             is_constant, is_arithmetic, arithmetic_diff, is_fibonacci, fibonacci_ratio,
                             seq_type_id, original_value, seq_type)

        print(f"Sequence {idx + 1} ({seq_type}):")
        print(f"Initial sequence: {denormalize(initial_seq, mn, std, transform)}")
        print(f"Predicted next {future_steps} steps: {np.round(preds, 4)}")
        print("-" * 50)