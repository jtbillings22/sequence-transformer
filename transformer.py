import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
import os

# Normalize the dataset to zero mean

def normalize_sequences(data):
    """Normalize each sequence to zero mean, unit std"""
    mn = data.mean(axis=1, keepdims=True)
    std = data.std(axis=1, keepdims=True)
    std[std == 0] = 1.0  # avoid division by zero
    norm_data = (data - mn) / std
    return norm_data.astype(np.float32), mn.flatten(), std.flatten()

def denormalize(value, mn, std):
    return value * std + mn

# load generated sequences 

def load_sequences_from_txt(filename: str, types=None, return_types: bool = False):
    
    selected = set(types) if types else None
    sequences = []
    seq_types = []

    # load seqeuences as np array, only use selected types if types is provided
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


    data = np.vstack(sequences)
    norm_data, mean, standard_dev = normalize_sequences(data)
    if return_types:
        return norm_data, mean, standard_dev, seq_types
    return norm_data, mean, standard_dev

# Creates training window examples with fixed length 30

def create_training_samples(sequences, seq_len=30, types=None, type_to_idx=None):
    # converts inputed sequences into sliding windows with various prediction targets,
    # delta (x_t+1 - x_t), ratio (x_t+1 / x_t) and direct next value (x_t+1). 
    
    X_list, Y_deltaList, Y_ratioList, Y_directList, T_list = [], [], [], [], []
    
    for seq_idx, seq in enumerate(sequences):
        if len(seq) <= seq_len:
            continue
        windows = np.lib.stride_tricks.sliding_window_view(seq, window_shape=seq_len)
        X_list.append(windows[:-1])
        
        # multiple targets for better learning
        last_vals = seq[seq_len - 1:-1]
        next_vals = seq[seq_len:]
        
        # Delta
        deltas = next_vals - last_vals
        Y_deltaList.append(deltas)
        
        # ratios = np.where(np.abs(last_vals) > 1e-8, next_vals / last_vals, 1.0)
        eps = 1e-8 # epsilon in case of 0
        ratios = next_vals / (last_vals + eps)

        Y_ratioList.append(ratios)
        
        # direct value
        Y_directList.append(next_vals)
        
        if types is not None:
            if type_to_idx is None:
                raise ValueError("type_to_idx mapping is required when types are provided.")
            type_id = type_to_idx[types[seq_idx]]
            T_list.append(np.full_like(deltas, fill_value=type_id, dtype=np.int64))

    X = np.concatenate(X_list, axis=0)
    Y_delta = np.concatenate(Y_deltaList, axis=0)
    Y_ratio = np.concatenate(Y_ratioList, axis=0)
    Y_direct = np.concatenate(Y_directList, axis=0)

    type_tensor = None
    if T_list:
        type_tensor = torch.tensor(np.concatenate(T_list, axis=0), dtype=torch.long)

    return (torch.tensor(X, dtype=torch.float32), 
            torch.tensor(Y_delta, dtype=torch.float32),
            torch.tensor(Y_ratio, dtype=torch.float32),
            torch.tensor(Y_direct, dtype=torch.float32),
            type_tensor)

# Transformer network with type embeddings as an attempt to enhance predictions

class Transformer(nn.Module):
    def __init__(self, dim_model=256, nhead=8, num_layers=6, num_types=10, dropout=0.1):
        super().__init__()
        self.d_model = dim_model
        
        # Input projection 
        self.input_proj = nn.Linear(1, dim_model)
        
        # Type embeddings
        self.type_embedding = nn.Embedding(num_types, dim_model)
        
        # Learnable positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, 500, dim_model) * 0.02)
        
        #  Transformer config
        layer = nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=nhead,
            dim_feedforward=dim_model * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        
        # 3-head prediction: delta, ratio, and direct value

        self.delta_head = nn.Sequential(
            nn.Linear(dim_model, dim_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_model // 2, 1)
        )
        
        self.ratio_head = nn.Sequential(
            nn.Linear(dim_model, dim_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_model // 2, 1)
        )
        
        self.direct_head = nn.Sequential(
            nn.Linear(dim_model, dim_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_model // 2, 1)
        )
        
        # 3 dim vector prediction weights used for gate
        self.gate = nn.Sequential(
            nn.Linear(dim_model, 3),
            nn.Softmax(dim=-1) # w softmax, our ml glorious king
        )
    
    # feed forward function
    def forward(self, x, type_ids):
        batch_size, seq_len = x.shape
        
        # input projection
        x = x.unsqueeze(-1)  # (batch, seq_len, 1)
        x = self.input_proj(x)
        
        # type embedding
        type_emb = self.type_embedding(type_ids).unsqueeze(1)  # (batch, 1, d_model)
        x = x + type_emb
        
        # positional encoding
        x = x + self.pos_embedding[:, :seq_len, :]
        
        h = self.transformer(x)
        last_token = h[:, -1, :]
        
        delta = self.delta_head(last_token).squeeze(-1)
        ratio = self.ratio_head(last_token).squeeze(-1)
        direct = self.direct_head(last_token).squeeze(-1)
        
        # gate weights (combines 3 prediction heads and outputs 3dim weighted vector)
        gates = self.gate(last_token)  # (batch, 3)
        
        return delta, ratio, direct, gates

# training loop, base config as; 256 batches, 10 epochs, and 1e-4 learning rate

def train_model(X, Y_delta, Y_ratio, Y_direct, type_ids, type_names, 
                batch_size=256, epochs=10, lr=1e-4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    dataset = TensorDataset(X, Y_delta, Y_ratio, Y_direct, type_ids)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                       pin_memory=True, num_workers=0)

    model = Transformer(num_types=len(type_names)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Scheduler to improve convergence, using cosinseAnealing
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    loss_function = nn.MSELoss()
    type_history = {name: [] for name in type_names}
    num_types = len(type_names)

    # Core training loop
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        per_type_sum = [0.0] * num_types
        per_type_count = [0] * num_types

        for batch_x, batch_delta, batch_ratio, batch_direct, batch_type in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_delta = batch_delta.to(device, non_blocking=True)
            batch_ratio = batch_ratio.to(device, non_blocking=True)
            batch_direct = batch_direct.to(device, non_blocking=True)
            batch_type = batch_type.to(device, non_blocking=True)

            optimizer.zero_grad()
            pred_delta, pred_ratio, pred_direct, gates = model(batch_x, batch_type)
            
            # delta, ratio and direct objective losses
            loss_delta = loss_function(pred_delta, batch_delta)
            loss_ratio = loss_function(pred_ratio, batch_ratio)
            loss_direct = loss_function(pred_direct, batch_direct)
            
            # sum losses
            loss = loss_delta + loss_ratio + loss_direct
            
            # backpropagation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item() * batch_x.size(0)

            # per type error tracking so we can plot loss error graph
            squared_error = (pred_delta - batch_delta) ** 2
            for t_idx in range(num_types):
                mask = batch_type == t_idx
                count = mask.sum().item()
                if count:
                    per_type_sum[t_idx] += squared_error[mask].sum().item()
                    per_type_count[t_idx] += count

        scheduler.step()
        
        avg = total_loss / len(dataset)
        per_type_mse = []
        for t_idx in range(num_types):
            if per_type_count[t_idx]:
                per_type_mse.append(per_type_sum[t_idx] / per_type_count[t_idx])
            else:
                per_type_mse.append(float("nan"))
        for name, mse in zip(type_names, per_type_mse):
            type_history[name].append(mse)

        per_type_str = " | ".join(
            f"{name}: {mse:.6f}" if not np.isnan(mse) else f"{name}: nan" 
            for name, mse in zip(type_names, per_type_mse)
        )
        print(f"Epoch {epoch}/{epochs} - Loss: {avg:.6f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        print(f"  Per-type MSE -> {per_type_str}")

    return model, type_history

# plots graph of error history

def plot_type_error_history(history, output_path="error_history.png"):
    if plt is None:
        print("matplotlib not available; skipping")
        return
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(next(iter(history.values()))) + 1)
    for type_name, values in history.items():
        plt.plot(epochs, values, label=type_name, marker='o', markersize=3)
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.yscale('log')  # Log scale to see all sequences
    plt.title("Per-type Training Error")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved per-type error plot to: {output_path}")

# multi-step prediction
 
def predict_future(model, initial_seq, steps, mn, std, device, type_id):
    model.eval()
    seq = initial_seq.copy()
    preds = []
    
    type_tensor = torch.tensor([type_id], dtype=torch.long).to(device)

    with torch.no_grad():
        seq_len = len(initial_seq)
        for _ in range(steps):
            x = torch.tensor(seq[-seq_len:], dtype=torch.float32).unsqueeze(0).to(device)
            pred_delta, pred_ratio, pred_direct, gates = model(x, type_tensor)
            
            # use gated combination of predictions
            last_val = seq[-1]
            
            # three prediction strategies
            pred_from_delta = last_val + pred_delta.cpu().item()
            pred_from_ratio = last_val * pred_ratio.cpu().item()
            pred_from_direct = pred_direct.cpu().item()
            
            # weighted combination
            gate_weights = gates.cpu().numpy()[0]
            next_val = (gate_weights[0] * pred_from_delta + 
                       gate_weights[1] * pred_from_ratio + 
                       gate_weights[2] * pred_from_direct)
            
            preds.append(denormalize(next_val, mn, std))
            seq = np.append(seq, next_val)

    return preds

# draw sample from each type, make predictions and plot then save to results

def predictions(model_path="results/transformer_model.pt", dataset_path="seq_dataset.txt", context_ratio=0.8, selected_types=None):
    if plt is None:
        print("matplotlib not available; skipping prediction plots.")
        return

    sequences, mean_list, standard_dev_list, seq_types = load_sequences_from_txt(dataset_path, selected_types, return_types=True)
    if len(sequences) == 0:
        print("No sequences available for predictions.")
        return

    type_to_first_idx = {}
    for idx, name in enumerate(seq_types):
        if name not in type_to_first_idx:
            type_to_first_idx[name] = idx

    type_names = list(type_to_first_idx.keys())
    type_to_idx = {name: i for i, name in enumerate(type_names)}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Transformer(num_types=len(type_names)).to(device)
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return
    model.load_state_dict(torch.load(model_path, map_location=device))

    os.makedirs("results", exist_ok=True)
    for type_name, seq_idx in type_to_first_idx.items():
        seq = sequences[seq_idx]
        if len(seq) < 2:
            continue
        mean, standard_dev = mean_list[seq_idx], standard_dev_list[seq_idx]
        split = int(len(seq) * context_ratio)
        split = min(max(split, 1), len(seq) - 1)  # ensure at least one future step

        context = seq[:split]
        steps = len(seq) - split
        type_id = type_to_idx[type_name]
        preds = predict_future(model, context, steps, mean, standard_dev, device, type_id)

        true_vals = denormalize(seq, mean, standard_dev)
        pred_series = np.concatenate([true_vals[:split], np.array(preds, dtype=np.float32)])

        plt.figure(figsize=(8, 4))
        plt.plot(true_vals, label="True")
        plt.plot(pred_series, linestyle="--", label="Predicted")
        plt.axvline(split - 1, color="gray", linestyle=":", label="Context end")
        plt.title(f"{type_name} predictions")
        plt.xlabel("Step")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()

        output_path = os.path.join("results", f"{type_name}_predictions.png")
        plt.savefig(output_path)
        plt.close()
        print(f"Saved prediction plot for {type_name} to: {output_path}")


if __name__ == "__main__":

    # Types included to train on
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
        "geometric"
    ]

    # Load sequences
    sequences, mean_list, standard_dev_list, seq_types = load_sequences_from_txt(
        "seq_dataset.txt", selected_types, return_types=True
    ) 
    print(f"Loaded {len(sequences)} sequences")
    type_names = list(selected_types)
    type_to_idx = {name: idx for idx, name in enumerate(type_names)}

    seq_len = 30
    X, Y_delta, Y_ratio, Y_direct, type_ids = create_training_samples(
        sequences, seq_len=seq_len, types=seq_types, type_to_idx=type_to_idx
    )
    print(f"Training samples: {len(X)}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, type_error_history = train_model(
        X, Y_delta, Y_ratio, Y_direct, type_ids, type_names, 
        batch_size=256, epochs=10, lr=1e-4
    )

    model_path = os.path.join("results", "transformer_model.pt")
    torch.save(model.state_dict(), model_path)

    plot_type_error_history(type_error_history)
    predictions(model_path, selected_types=selected_types)

    print("\n==================== PREDICTIONS ====================\n")

    max_predictions = 50
    for idx, seq in enumerate(sequences[:max_predictions]):
        mean, standard_dev = mean_list[idx], standard_dev_list[idx]
        seq_type = seq_types[idx]
        type_id = type_to_idx[seq_type]
        
        initial_seq = seq[:seq_len]
        future_steps = 5
        preds = predict_future(model, initial_seq, future_steps, mean, standard_dev, device, type_id)

        print(f"Sequence {idx + 1} ({seq_type}):")
        print(f"Initial sequence: {denormalize(initial_seq, mean, standard_dev)}")
        print(f"Predicted next {future_steps} steps: {np.round(preds, 4)}")
        print("-" * 50)
