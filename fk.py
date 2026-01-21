import numpy as np
import matplotlib.pyplot as plt

# Μήκη συνδέσμων
l1 = 1.0
l2 = 0.8

def forward_kinematics(theta1, theta2):
    """
    Υπολογίζει τη θέση (x, y) του end-effector
    για δοσμένες γωνίες theta1, theta2 (σε rad).
    """
    x = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2)
    y = l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2)
    return x, y

def plot_robot(theta1, theta2):
    """
    Σχεδιάζει το 2R ρομπότ για δοσμένες γωνίες.
    """
    # Πρώτος σύνδεσμος
    x1 = l1 * np.cos(theta1)
    y1 = l1 * np.sin(theta1)

    # End-effector
    x2 = x1 + l2 * np.cos(theta1 + theta2)
    y2 = y1 + l2 * np.sin(theta1 + theta2)

    # Σημεία άρθρωσης
    X = [0, x1, x2]
    Y = [0, y1, y2]

    plt.figure()
    plt.plot(X, Y, '-o')
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid()
    plt.title("2R Planar Robot")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

if __name__ == "__main__":
    # Παράδειγμα δοκιμής
    theta1 = 0.5   # rad
    theta2 = 0.3   # rad

    x, y = forward_kinematics(theta1, theta2)

    print("End-effector position:")
    print("x =", x)
    print("y =", y)

    plot_robot(theta1, theta2)
