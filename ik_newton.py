import numpy as np
import matplotlib.pyplot as plt

# Μήκη συνδέσμων (ίδια παντού)
l1 = 1.0
l2 = 0.8

def forward_kinematics(theta):
    """theta = [theta1, theta2] (rad) -> (x, y)"""
    t1, t2 = theta
    x = l1 * np.cos(t1) + l2 * np.cos(t1 + t2)
    y = l1 * np.sin(t1) + l2 * np.sin(t1 + t2)
    return np.array([x, y], dtype=float)

def jacobian(theta):
    """Jacobian 2x2"""
    t1, t2 = theta
    s1 = np.sin(t1)
    c1 = np.cos(t1)
    s12 = np.sin(t1 + t2)
    c12 = np.cos(t1 + t2)

    j11 = -l1 * s1 - l2 * s12
    j12 = -l2 * s12
    j21 =  l1 * c1 + l2 * c12
    j22 =  l2 * c12

    return np.array([[j11, j12],
                     [j21, j22]], dtype=float)

def ik_newton(target_xy, theta0, max_iters=50, tol=1e-6, step_scale=1.0):
    """
    Newton / Resolved-Rate IK:
    target_xy: desired [x_d, y_d]
    theta0: initial guess [t1, t2]
    step_scale: 1.0 = κανονικό Newton. Μικρότερο (π.χ. 0.5) πιο σταθερό.
    """
    theta = np.array(theta0, dtype=float)

    history = []  # κρατάμε τροχιά του end-effector για plot

    for k in range(max_iters):
        current = forward_kinematics(theta)
        error = target_xy - current
        err_norm = np.linalg.norm(error)

        history.append(current)

        # Αν το σφάλμα είναι αρκετά μικρό, σταματάμε
        if err_norm < tol:
            return theta, np.array(history), True, k

        J = jacobian(theta)

        # Pseudoinverse (σταθερότερο από inverse)
        J_pinv = np.linalg.pinv(J)

        delta_theta = step_scale * (J_pinv @ error)

        theta = theta + delta_theta

    return theta, np.array(history), False, max_iters

def plot_robot(theta, target=None, path=None):
    """Σχεδιάζει το 2R ρομπότ + στόχο + path end-effector"""
    t1, t2 = theta

    # joint1
    x1 = l1 * np.cos(t1)
    y1 = l1 * np.sin(t1)

    # end-effector
    x2 = x1 + l2 * np.cos(t1 + t2)
    y2 = y1 + l2 * np.sin(t1 + t2)

    plt.figure()
    plt.plot([0, x1, x2], [0, y1, y2], "-o", label="robot")

    if target is not None:
        plt.plot(target[0], target[1], "x", markersize=10, label="target")

    if path is not None and len(path) > 0:
        plt.plot(path[:, 0], path[:, 1], "--", label="end-effector path")

    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid()
    plt.legend()
    plt.title("Newton / Resolved-Rate IK (2R)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

if __name__ == "__main__":
    # --- Διάλεξε έναν στόχο ---
    # Προσοχή: πρέπει να είναι μέσα στο workspace.
    # Το μέγιστο reach είναι l1 + l2 = 1.8
    target = np.array([1.2, 0.5], dtype=float)

    # Αρχική εικασία γωνιών (rad)
    theta0 = np.array([0.2, 0.2], dtype=float)

    theta_sol, path, converged, iters = ik_newton(
        target_xy=target,
        theta0=theta0,
        max_iters=60,
        tol=1e-6,
        step_scale=0.8   # λίγο πιο σταθερό από 1.0
    )

    print("Converged:", converged, "iters:", iters)
    print("theta_sol (rad):", theta_sol)

    final_xy = forward_kinematics(theta_sol)
    print("final xy:", final_xy)
    print("target xy:", target)
    print("final error norm:", np.linalg.norm(target - final_xy))

    plot_robot(theta_sol, target=target, path=path)
