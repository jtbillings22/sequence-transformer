import math
import random
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim


# ============================================
# 1. Utility: make sliding-window dataset
# ============================================

def make_sliding_windows(
    sequence: List[float],
    window_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Given a 1D list of numbers [x1, x2, ..., xN],
    construct training examples of the form:

      input:  [x_t, x_{t+1}, ..., x_{t+window_size-1}]
      target: x_{t+window_size}

    for all valid t.

    Returns:
      X: tensor of shape (num_samples, window_size)
      y: tensor of shape (num_samples, 1)
    """
    X = []
    y = []
    for i in range(len(sequence) - window_size):
        window = sequence[i : i + window_size]         # length = window_size
        target = sequence[i + window_size]             # the next value
        X.append(window)
        y.append([target])  # wrap target in list so shape is (1,), then -> (num_samples, 1)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return X, y



# ============================================
# 3. Model: simple MLP to predict next number
# ============================================

class NextNumberMLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64):
        """
        input_size: window_size (how many past values we feed in)
        hidden_size: size of hidden layer
        """
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)  # predict a single scalar

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch_size, input_size)
        returns: (batch_size, 1)
        """
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ============================================
# 4. Training loop
# ============================================

def train_model(
    X: torch.Tensor,
    y: torch.Tensor,
    window_size: int,
    num_epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 32
) -> NextNumberMLP:
    """
    Train NextNumberMLP on the given dataset (X, y).

    X: (num_samples, window_size)
    y: (num_samples, 1)
    """
    model = NextNumberMLP(input_size=window_size, hidden_size=64)

    # Mean Squared Error is standard for regression.
    criterion = nn.MSELoss()

    # Adam optimizer is a good default choice.
    optimizer = optim.Adam(model.parameters(), lr=lr)

    num_samples = X.shape[0]

    for epoch in range(num_epochs):
        # simple mini-batch loop
        permutation = torch.randperm(num_samples)

        epoch_loss = 0.0
        for i in range(0, num_samples, batch_size):
            indices = permutation[i : i + batch_size]
            batch_X = X[indices]
            batch_y = y[indices]

            # 1) Zero old gradients
            optimizer.zero_grad()

            # 2) Forward pass
            preds = model(batch_X)

            # 3) Compute loss
            loss = criterion(preds, batch_y)

            # 4) Backprop: compute gradients
            loss.backward()

            # 5) Step: update parameters
            optimizer.step()

            epoch_loss += loss.item() * batch_X.size(0)

        epoch_loss /= num_samples

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs}  |  Loss: {epoch_loss:.6f}")

    return model


# ============================================
# 5. Putting it all together
# ============================================

def main():
    # ----- Step 1: generate or load your numeric sequence -----
    full_sequence = [28,38,48,58,68,78,88,98,108,118,128,138,148,158,168,178,188,198,208,218,228,238,248,258,268,278,288,298,308,318,328,338,348,358,368,378,388,398,408,418,428,438,448,458,468,478,488,498,508,518,528,538,548,558,568,578,588,598,608,618,628,638,648,658,668,678,688,698,708,718,728,738,748,758,768,778,788,798,808,818,828,838,848,858,868,878,888,898,908,918,928,938,948,958,968,978,988,998,1008,1018,1028,1038,1048,1058,1068,1078,1088,1098,1108,1118,1128,1138,1148,1158,1168,1178,1188,1198,1208,1218,1228,1238,1248,1258,1268,1278,1288,1298,1308,1318,1328,1338,1348,1358,1368,1378,1388,1398,1408,1418,1428,1438,1448,1458,1468,1478,1488,1498,1508,1518,1528,1538,1548,1558,1568,1578,1588,1598,1608,1618,1628,1638,1648,1658,1668,1678,1688,1698,1708,1718,1728,1738,1748,1758,1768,1778,1788,1798,1808,1818,1828,1838,1848,1858,1868,1878,1888,1898,1908,1918,1928,1938,1948,1958,1968,1978,1988,1998,2008,2018,2028,2038,2048,2058,2068,2078,2088,2098,2108,2118,2128,2138,2148,2158,2168,2178,2188,2198,2208,2218,2228,2238,2248,2258,2268,2278,2288,2298,2308,2318,2328,2338,2348,2358,2368,2378,2388,2398,2408,2418,2428,2438,2448,2458,2468,2478,2488,2498,2508,2518,2528,2538,2548,2558,2568,2578,2588,2598,2608,2618,2628,2638,2648,2658,2668,2678,2688,2698,2708,2718,2728,2738,2748,2758,2768,2778,2788,2798,2808,2818,2828,2838,2848,2858,2868,2878,2888,2898,2908,2918,2928,2938,2948,2958,2968,2978,2988,2998,3008,3018,3028,3038,3048,3058,3068,3078,3088,3098,3108,3118,3128,3138,3148,3158,3168,3178,3188,3198,3208,3218,3228,3238,3248,3258,3268,3278,3288,3298,3308,3318,3328,3338,3348,3358,3368,3378,3388,3398,3408,3418,3428,3438,3448,3458,3468,3478,3488,3498,3508,3518,3528,3538,3548,3558,3568,3578,3588,3598,3608,3618,3628,3638,3648,3658,3668,3678,3688,3698,3708,3718,3728,3738,3748,3758,3768,3778,3788,3798,3808,3818,3828,3838,3848,3858,3868,3878,3888,3898,3908,3918,3928,3938,3948,3958,3968,3978,3988,3998,4008,4018,4028,4038,4048,4058,4068,4078,4088,4098,4108,4118,4128,4138,4148,4158,4168,4178,4188,4198,4208,4218,4228,4238,4248,4258,4268,4278,4288,4298,4308,4318,4328,4338,4348,4358,4368,4378,4388,4398,4408,4418,4428,4438,4448,4458,4468,4478,4488,4498,4508,4518,4528,4538,4548,4558,4568,4578,4588,4598,4608,4618,4628,4638,4648,4658,4668,4678,4688,4698,4708,4718,4728,4738,4748,4758,4768,4778,4788,4798,4808,4818,4828,4838,4848,4858,4868,4878,4888,4898,4908,4918,4928,4938,4948,4958,4968,4978,4988,4998,5008,5018
]

    # Split into train / test so we can see if it generalizes
    train_sequence = full_sequence[:25]
    test_sequence  = full_sequence[25:]  # last 200 points

    # ----- Step 2: build sliding windows -----
    window_size = 5  # how many past points we look at

    X_train, y_train = make_sliding_windows(train_sequence, window_size)
    X_test,  y_test  = make_sliding_windows(test_sequence, window_size)

    print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

    # ----- Step 3: train the model -----
    model = train_model(
        X_train,
        y_train,
        window_size=window_size,
        num_epochs=5000,
        lr=1e-3,
        batch_size=64,
    )

    # ----- Step 4: evaluate quickly on test set -----
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test)
        test_loss = nn.MSELoss()(test_preds, y_test).item()

    print(f"\nTest MSE: {test_loss:.6f}")

    # ----- Step 5: show a few example predictions -----
    print("\nSome predictions (true vs predicted):")
    for i in range(5):
        true_val = y_test[i].item()
        pred_val = test_preds[i].item()
        window = X_test[i].tolist()
        predicted_sequence = [*window, pred_val]
        print(f"  Sample {i}: true = {true_val:+.4f},  pred = {pred_val:+.4f}, predicted sequence = {predicted_sequence}")

if __name__ == "__main__":
    main()
