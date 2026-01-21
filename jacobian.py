import numpy as np

# Μήκη συνδέσμων (ίδια με fk.py)
l1 = 1.0
l2 = 0.8

def forward_kinematics(theta1, theta2):
    """Επιστρέφει x, y του end-effector."""
    x = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2)
    y = l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2)
    return x, y

def jacobian(theta1, theta2):
    """Επιστρέφει τον Jacobian 2x2 για το 2R ρομπότ."""
    s1 = np.sin(theta1)
    c1 = np.cos(theta1)
    s12 = np.sin(theta1 + theta2)
    c12 = np.cos(theta1 + theta2)

    j11 = -l1 * s1 - l2 * s12
    j12 = -l2 * s12
    j21 =  l1 * c1 + l2 * c12
    j22 =  l2 * c12

    J = np.array([[j11, j12], [j21, j22]], dtype=float)
    return J

if __name__ == "__main__":
    theta1 = 0.5
    theta2 = 0.3

    x, y = forward_kinematics(theta1, theta2)
    J = jacobian(theta1, theta2)

    print("FK:")
    print("x =", x, "y =", y)
    print("\nJacobian J:")
    print(J)

    # Έλεγχος: αν δώσουμε μικρές ταχύτητες στις αρθρώσεις,
    # το J μας δίνει περίπου τις ταχύτητες στο (x, y)
    theta_dot = np.array([0.1, -0.05])  # rad/s
    xy_dot = J @ theta_dot

    print("\nExample velocity mapping:")
    print("theta_dot =", theta_dot)
    print("xy_dot = J @ theta_dot =", xy_dot)
