import numpy as np
from scipy.optimize import minimize

# Define the loss function
def loss_function(z):
    x, y = z
    return x**2 + y**2 - 6*x*y - 4*x - 5*y

# Define the original constraints
def constraint1(z):
    x, y = z
    return -(x-2)**2 + 4 - y  # First constraint

def constraint2(z):
    x, y = z
    return y - (-x + 1)  # Second constraint

# Define the new first constraint
def new_constraint1(z):
    x, y = z
    return -(x-2)**2 + 4.1 - y

# Initial guess
x0 = np.array([0.0, 0.0])

# Set up the original constraints
cons = [{'type': 'ineq', 'fun': constraint1},
        {'type': 'ineq', 'fun': constraint2}]

# Optimize with original constraints and print the result
result = minimize(loss_function, x0, constraints=cons)
print("Original constraints solution:", result.x)
print("Original constraints optimal loss:", result.fun)

# Set up the new constraints
new_cons = [{'type': 'ineq', 'fun': new_constraint1},
            {'type': 'ineq', 'fun': constraint2}]

# Optimize with new constraints and print the result
new_result = minimize(loss_function, x0, constraints=new_cons)
print("New constraints solution:", new_result.x)
print("New constraints optimal loss:", new_result.fun)
