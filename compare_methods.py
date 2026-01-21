import time
import numpy as np
import torch
import torch.nn as nn

# ίδια μήκη
l1 = 1.0
l2 = 0.8
reach = l1 + l2

# ---------- FK & J (numpy, για Newton) ----------
def fk_np(theta):
    t1, t2 = theta
    x = l1*np.cos(t1) + l2*np.cos(t1+t2)
    y = l1*np.sin(t1) + l2*np.sin(t1+t2)
    return np.array([x,y], dtype=float)

def jacobian_np(theta):
    t1, t2 = theta
    s1 = np.sin(t1); c1 = np.cos(t1)
    s12 = np.sin(t1+t2); c12 = np.cos(t1+t2)
    j11 = -l1*s1 - l2*s12
    j12 = -l2*s12
    j21 =  l1*c1 + l2*c12
    j22 =  l2*c12
    return np.array([[j11,j12],[j21,j22]], dtype=float)

def ik_newton(target_xy, theta0, max_iters=60, tol=1e-6, step_scale=0.8):
    theta = np.array(theta0, dtype=float)
    for _ in range(max_iters):
        cur = fk_np(theta)
        e = target_xy - cur
        if np.linalg.norm(e) < tol:
            return theta, True
        J = jacobian_np(theta)
        dtheta = step_scale * (np.linalg.pinv(J) @ e)
        theta = theta + dtheta
    return theta, False

# ---------- NN model (ίδιο architecture με πριν) ----------
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4)   # cos t1, sin t1, cos t2, sin t2
        )
    def forward(self, x):
        return self.net(x)

def sincos_to_angles(y_pred):
    c1 = y_pred[:,0]; s1 = y_pred[:,1]
    c2 = y_pred[:,2]; s2 = y_pred[:,3]
    t1 = torch.atan2(s1, c1)
    t2 = torch.atan2(s2, c2)
    return t1, t2

def fk_torch(t1, t2):
    x = l1*torch.cos(t1) + l2*torch.cos(t1+t2)
    y = l1*torch.sin(t1) + l2*torch.sin(t1+t2)
    return x, y

if __name__ == "__main__":
    # Load trained NN
    model = MLP()
    model.load_state_dict(torch.load("nn_ik_2r.pt", map_location="cpu"))
    model.eval()

    # Generate random targets by sampling angles and using FK (guaranteed reachable)
    rng = np.random.default_rng(0)
    N = 2000

    thetas = np.stack([
        rng.uniform(-np.pi, np.pi, size=N),
        rng.uniform(0.0, np.pi, size=N)  # theta2 >= 0, same as dataset rule
    ], axis=1)

    targets = np.array([fk_np(th) for th in thetas], dtype=float)

    # --- Compare ---
    nn_errors = []
    newton_errors = []
    newton_fail = 0

    t_nn_start = time.time()

    # NN inference batch
    X_norm = (targets / reach).astype(np.float32)
    with torch.no_grad():
        inp = torch.tensor(X_norm)
        out = model(inp)
        t1p, t2p = sincos_to_angles(out)
        x_pred, y_pred = fk_torch(t1p, t2p)

    t_nn = time.time() - t_nn_start

    pred_xy = torch.stack([x_pred, y_pred], dim=1).cpu().numpy()
    nn_errors = np.linalg.norm(targets - pred_xy, axis=1)

    # Newton (loop)
    t_newton_start = time.time()
    for i in range(N):
        target = targets[i]
        theta0 = np.array([0.2, 0.2])  # fixed initial guess
        theta_sol, ok = ik_newton(target, theta0)
        if not ok:
            newton_fail += 1
        err = np.linalg.norm(target - fk_np(theta_sol))
        newton_errors.append(err)

    t_newton = time.time() - t_newton_start
    newton_errors = np.array(newton_errors)

    print("----- Results (N =", N, ") -----")
    print("NN  : mean err =", nn_errors.mean(), "median =", np.median(nn_errors), "95% =", np.quantile(nn_errors, 0.95))
    print("Newton: mean err =", newton_errors.mean(), "median =", np.median(newton_errors), "95% =", np.quantile(newton_errors, 0.95))
    print("Newton fails:", newton_fail, "/", N)
    print("\nTiming (rough, on your PC):")
    print("NN total time (batch):", t_nn, "seconds")
    print("Newton total time:", t_newton, "seconds")
