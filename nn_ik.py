import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Μήκη συνδέσμων (για FK + error σε πραγματικές μονάδες)
l1 = 1.0
l2 = 0.8

def forward_kinematics_torch(theta1, theta2):
    """Torch FK: theta1, theta2 tensors -> x,y tensors"""
    x = l1 * torch.cos(theta1) + l2 * torch.cos(theta1 + theta2)
    y = l1 * torch.sin(theta1) + l2 * torch.sin(theta1 + theta2)
    return x, y

def wrap_to_pi_torch(a):
    """Wrap angle to [-pi, pi) in torch."""
    return (a + np.pi) % (2*np.pi) - np.pi

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # cos t1, sin t1, cos t2, sin t2
        )

    def forward(self, x):
        return self.net(x)

def angles_to_sincos(theta):
    """
    theta shape: (N,2) -> returns (N,4): [cos t1, sin t1, cos t2, sin t2]
    """
    t1 = theta[:, 0]
    t2 = theta[:, 1]
    return np.stack([np.cos(t1), np.sin(t1), np.cos(t2), np.sin(t2)], axis=1)

def sincos_to_angles(y_pred):
    """
    y_pred torch shape: (N,4) -> theta1, theta2 torch
    theta = atan2(sin, cos)
    """
    c1 = y_pred[:, 0]
    s1 = y_pred[:, 1]
    c2 = y_pred[:, 2]
    s2 = y_pred[:, 3]
    t1 = torch.atan2(s1, c1)
    t2 = torch.atan2(s2, c2)
    return t1, t2

def plot_robot(theta1, theta2, target_xy=None, title="2R Robot (NN IK)"):
    """Matplotlib plot robot in 2D"""
    x1 = l1 * np.cos(theta1)
    y1 = l1 * np.sin(theta1)
    x2 = x1 + l2 * np.cos(theta1 + theta2)
    y2 = y1 + l2 * np.sin(theta1 + theta2)

    plt.figure()
    plt.plot([0, x1, x2], [0, y1, y2], "-o", label="robot")
    if target_xy is not None:
        plt.plot(target_xy[0], target_xy[1], "x", markersize=10, label="target")
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # --- 1) Load dataset ---
    data = np.load("dataset_2r.npz")
    X = data["X"]  # normalized (x,y) in roughly [-1,1]
    Y = data["Y"]  # angles (theta1, theta2) in [-pi, pi)

    # Convert labels to sin/cos (για να μην μπερδευτεί το NN με wrap)
    Y_sc = angles_to_sincos(Y)

    # --- 2) Train/test split ---
    N = X.shape[0]
    idx = np.random.permutation(N)
    split = int(0.85 * N)
    train_idx, test_idx = idx[:split], idx[split:]

    X_train, Y_train = X[train_idx], Y_sc[train_idx]
    X_test,  Y_test  = X[test_idx],  Y_sc[test_idx]

    # Torch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32)
    X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
    Y_test_t  = torch.tensor(Y_test,  dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train_t, Y_train_t),
                              batch_size=256, shuffle=True)

    # --- 3) Model / loss / optimizer ---
    device = "cpu"  # απλό, μην μπλέκουμε με GPU
    model = MLP().to(device)
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # --- 4) Training loop ---
    epochs = 30
    train_losses = []
    test_losses = []

    for ep in range(epochs):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb)
            loss = loss_fn(pred, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item() * xb.shape[0]

        train_loss = total / len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        with torch.no_grad():
            pred_test = model(X_test_t.to(device))
            test_loss = loss_fn(pred_test, Y_test_t.to(device)).item()
            test_losses.append(test_loss)

        print(f"Epoch {ep+1:02d}/{epochs} | train_loss={train_loss:.6f} | test_loss={test_loss:.6f}")

    # --- 5) Plot loss curves ---
    plt.figure()
    plt.plot(train_losses, label="train loss")
    plt.plot(test_losses, label="test loss")
    plt.grid(True)
    plt.xlabel("epoch")
    plt.ylabel("MSE loss (sincos)")
    plt.title("Training curves")
    plt.legend()
    plt.show()

    # --- 6) Evaluate in Cartesian space (με FK) ---
    # Θέλουμε να δούμε πόσο κοντά πάει στο (x,y) που δίνουμε.
    # Προσοχή: το X είναι normalized, άρα απο-normalize:
    reach = l1 + l2
    X_test_real = X_test_t * reach  # torch

    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_t.to(device))
        t1_pred, t2_pred = sincos_to_angles(y_pred)

        x_pred, y_pred_xy = forward_kinematics_torch(t1_pred, t2_pred)

        # true (x,y) είναι τα input test σε real units:
        x_true = X_test_real[:, 0]
        y_true = X_test_real[:, 1]

        pos_err = torch.sqrt((x_true - x_pred.cpu())**2 + (y_true - y_pred_xy.cpu())**2)
        print("\n--- Cartesian error on TEST set ---")
        print("mean error:", float(pos_err.mean()))
        print("median error:", float(pos_err.median()))
        print("95th percentile:", float(torch.quantile(pos_err, 0.95)))

    # --- 7) Quick demo: pick one test sample and draw robot ---
    i = np.random.randint(0, X_test.shape[0])
    target_norm = X_test[i]
    target_real = target_norm * reach

    with torch.no_grad():
        inp = torch.tensor(target_norm[None, :], dtype=torch.float32)
        out = model(inp)
        t1, t2 = sincos_to_angles(out)
        t1 = float(t1.item())
        t2 = float(t2.item())

    print("\nDemo target (x,y) =", target_real)
    print("Predicted angles (rad) =", (t1, t2))

    plot_robot(t1, t2, target_xy=target_real, title="2R Robot using NN IK (one demo)")

    # --- 8) Save model ---
    torch.save(model.state_dict(), "nn_ik_2r.pt")
    print("\nSaved model to nn_ik_2r.pt")
