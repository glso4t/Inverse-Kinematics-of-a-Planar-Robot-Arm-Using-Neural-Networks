import numpy as np
import matplotlib.pyplot as plt

# Μήκη συνδέσμων
l1 = 1.0
l2 = 0.8

def forward_kinematics(theta1, theta2):
    """(theta1, theta2) -> (x, y)"""
    x = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2)
    y = l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2)
    return x, y

def wrap_to_pi(angle):
    """
    Τυλίγει γωνία στο [-pi, pi)
    """
    return (angle + np.pi) % (2*np.pi) - np.pi

def make_dataset(n_samples=50000, seed=0, enforce_theta2_positive=True):
    """
    Φτιάχνει dataset:
      input:  (x, y)
      output: (theta1, theta2)

    enforce_theta2_positive:
      - αν True, κρατάμε μόνο δείγματα με theta2 >= 0
        για να είναι το mapping (x,y)->(theta1,theta2) μονοσήμαντο.
    """
    rng = np.random.default_rng(seed)

    X = []
    Y = []

    # Δειγματοληψία γωνιών (τυχαία)
    # Επιλέγουμε range [-pi, pi) για να καλύπτουμε όλο το χώρο
    for _ in range(n_samples * 2):  # *2 για να έχουμε περιθώριο αν φιλτράρουμε
        theta1 = rng.uniform(-np.pi, np.pi)
        theta2 = rng.uniform(-np.pi, np.pi)

        if enforce_theta2_positive and theta2 < 0:
            continue

        x, y = forward_kinematics(theta1, theta2)

        X.append([x, y])
        Y.append([wrap_to_pi(theta1), wrap_to_pi(theta2)])

        if len(X) >= n_samples:
            break

    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)
    return X, Y

def normalize_inputs(X):
    """
    Normalization για inputs (x,y)
    Καλό για NN: να είναι περίπου στο [-1,1]
    Εδώ διαιρούμε με max reach (l1+l2).
    """
    reach = l1 + l2
    return X / reach

def save_dataset(X, Y, filename="dataset_2r.npz"):
    """
    Αποθήκευση σε .npz (εύκολο να το φορτώσεις μετά)
    """
    np.savez(filename, X=X, Y=Y, l1=l1, l2=l2)
    print(f"Saved dataset to {filename}")
    print("X shape:", X.shape, "Y shape:", Y.shape)

if __name__ == "__main__":
    # 1) Φτιάχνουμε dataset
    X, Y = make_dataset(n_samples=50000, seed=42, enforce_theta2_positive=True)

    # 2) Normalization inputs (θα το χρησιμοποιήσουμε στο NN)
    Xn = normalize_inputs(X)

    # 3) Αποθήκευση
    save_dataset(Xn, Y, filename="dataset_2r.npz")

    # 4) Γρήγορο check: scatter plot των (x,y)
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], s=1, alpha=0.3)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title("Workspace samples (x,y) from generated dataset")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.show()
