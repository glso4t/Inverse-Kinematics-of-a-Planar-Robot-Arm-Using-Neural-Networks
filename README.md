# Inverse Kinematics of a Planar Robot Arm using Neural Networks

## Project Description
This project was developed for the **AI in Robotics** course.  
It focuses on solving the **inverse kinematics (IK)** problem of a planar **2R robotic manipulator** using two different approaches:

1. A **classical numerical method** (Newton–Raphson / resolved-rate control)
2. A **machine learning approach** using a **neural network**

The two methods are implemented, tested, and **quantitatively compared** in terms of accuracy, robustness, and execution time.

---

## Robot Model
- **Type**: Planar articulated robot (2 revolute joints – 2R)
- **Link lengths**:
  - l₁ = 1.0
  - l₂ = 0.8

### Forward Kinematics
The end-effector position is given by:
- `x = l1 * cos(θ1) + l2 * cos(θ1 + θ2)`
- `y = l1 * sin(θ1) + l2 * sin(θ1 + θ2)`

These equations are implemented in `fk.py` and are used throughout the project
for dataset generation, numerical inverse kinematics, and neural network evaluation.

---

## Implemented Methods

### 1. Newton–Raphson Inverse Kinematics
- Iterative numerical solution
- Uses the Jacobian matrix and its pseudoinverse
- Very high accuracy when convergence is achieved
- May fail depending on:
  - initial guess
  - proximity to singular configurations

### 2. Neural Network Inverse Kinematics
- Multi-layer perceptron (MLP)
- Input: Cartesian position (x, y)
- Output: sin/cos representation of joint angles
- Joint angles recovered using `atan2`
- Fast inference and robust behavior

---

## Project Structure

project/
│
├── fk.py # Forward kinematics
├── jacobian.py # Jacobian computation
├── ik_newton.py # Newton–Raphson inverse kinematics
├── dataset.py # Dataset generation
├── nn_ik.py # Neural network training and demo
├── compare_methods.py # Quantitative comparison (NN vs Newton)
├── README.md
└── report.pdf

---

## Requirements
- Python 3.9 or newer
- NumPy
- Matplotlib
- PyTorch

---

## Installation
Install the required packages with:

pip install numpy matplotlib torch

---

## HOT TO RUN
Forward Kinematics Demo
python fk.py

## Jacobian Computation
python jacobian.py

## Newton–Raphson Inverse Kinematics
python ik_newton.py

## Dataset Generation
python dataset.py
- This creates the file:
 - dataset_2r.npz

## Neural Network Training and Demo
python nn_ik.py
- This will:
 - train the neural network
 - display training curves
 - run a demo inverse kinematics example
 - save the trained model as nn_ik_2r.pt

## Method Comparison
python compare_methods.py
- This script compares:
 - Cartesian position error
 - convergence behavior
 - execution time of both methods

---

## Results Summary
- Newton–Raphson IK
 - Extremely accurate when convergent
 - Can fail in some cases
 - Computationally expensive

- Neural Network IK
 - Approximate solution (centimeter-level accuracy)
 - No convergence failures
 - Significantly faster than the numerical method

---

## Conclusion
This project demonstrates how artificial intelligence can complement classical robotics algorithms.
Neural networks provide fast and robust inverse kinematics approximations, while numerical methods remain preferable when maximum accuracy is required.

---
