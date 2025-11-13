"""
Fit unknown parameters (θ, M, X) in:
x = t*cos(θ) - e^(M|t|)*sin(0.3t)*sin(θ) + X
y = 42 + t*sin(θ) + e^(M|t|)*sin(0.3t)*cos(θ)

Author: Hiya Nachnani
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ============ STEP 1: Load your data ============
# Replace 'xy_data.csv' with your CSV filename (must have columns x,y)
df = pd.read_csv("xy_data.csv")   # <-- change this line
x_vals = df['x'].values
y_vals = df['y'].values
n = len(x_vals)

# If you have t values, read them here instead of generating linearly
t_vals = np.linspace(6, 60, n)   # assumed uniform

# ============ STEP 2: Define model ============
def predict(params, t):
    theta_deg, M, X = params
    theta = np.deg2rad(theta_deg)
    term = np.exp(M * np.abs(t)) * np.sin(0.3 * t)
    x_pred = t * np.cos(theta) - term * np.sin(theta) + X
    y_pred = 42 + t * np.sin(theta) + term * np.cos(theta)
    return x_pred, y_pred

# ============ STEP 3: Loss function ============
def loss(params):
    x_pred, y_pred = predict(params, t_vals)
    return np.sum((x_vals - x_pred)**2 + (y_vals - y_pred)**2)

# ============ STEP 4: Initial estimates ============
# Estimate θ from linear trend ignoring oscillation
coeffs = np.polyfit(x_vals, y_vals - 42, 1)  # (y-42) ≈ tan(θ)*x
theta_init_deg = np.rad2deg(np.arctan(coeffs[0]))
theta_init_deg = np.clip(theta_init_deg, 0.1, 49.9)
X_init = np.mean(x_vals - t_vals * np.cos(np.deg2rad(theta_init_deg)))
M_init = 0.0

initial = [theta_init_deg, M_init, X_init]
bounds = [(0.0001, 50.0), (-0.05, 0.05), (0.0, 100.0)]

# ============ STEP 5: Optimization ============
best_res = minimize(loss, initial, bounds=bounds, method='L-BFGS-B', options={'maxiter':10000})
theta_opt_deg, M_opt, X_opt = best_res.x

# ============ STEP 6: Evaluation ============
x_pred, y_pred = predict((theta_opt_deg, M_opt, X_opt), t_vals)
dists = np.sqrt((x_vals - x_pred)**2 + (y_vals - y_pred)**2)
L1_sum = np.sum(dists)
L1_mean = np.mean(dists)

# ============ STEP 7: Print results ============
print("\n===== PARAMETER ESTIMATES =====")
print(f"Theta (degrees): {theta_opt_deg:.6f}")
print(f"M: {M_opt:.6f}")
print(f"X: {X_opt:.6f}")
print(f"Sum of squared error: {best_res.fun:.4f}")
print(f"L1 distance sum: {L1_sum:.4f}, mean: {L1_mean:.4f}")

latex = rf"\left(t\cos({theta_opt_deg:.6f}) - e^{{{M_opt:.6f}|t|}}\sin(0.3t)\sin({theta_opt_deg:.6f}) + {X_opt:.6f},\ 42 + t\sin({theta_opt_deg:.6f}) + e^{{{M_opt:.6f}|t|}}\sin(0.3t)\cos({theta_opt_deg:.6f})\right)"
print("\n===== SUBMISSION STRING =====")
print(latex)

# ============ STEP 8: Save results ============
# Save predictions
df['x_pred'] = x_pred
df['y_pred'] = y_pred
df['residual'] = dists
df.to_csv("param_fit_results.csv", index=False)
print("\nSaved detailed fit results to param_fit_results.csv")

# Save equation to text file
with open("fitted_equation.txt", "w") as f:
    f.write("===== PARAMETER ESTIMATES =====\n")
    f.write(f"Theta (degrees): {theta_opt_deg:.6f}\n")
    f.write(f"M: {M_opt:.6f}\n")
    f.write(f"X: {X_opt:.6f}\n")
    f.write(f"Sum of squared error: {best_res.fun:.4f}\n")
    f.write(f"L1 distance sum: {L1_sum:.4f}, mean: {L1_mean:.4f}\n\n")
    f.write("===== FITTED EQUATION =====\n")
    f.write(f"x = t*cos({theta_opt_deg:.6f}) - exp({M_opt:.6f}*|t|)*sin(0.3t)*sin({theta_opt_deg:.6f}) + {X_opt:.6f}\n")
    f.write(f"y = 42 + t*sin({theta_opt_deg:.6f}) + exp({M_opt:.6f}*|t|)*sin(0.3t)*cos({theta_opt_deg:.6f})\n\n")
    f.write("===== LATEX SUBMISSION STRING =====\n")
    f.write(latex)
print("Saved fitted equation to fitted_equation.txt")

# ============ STEP 9: Visualization ============
plt.figure(figsize=(8,6))
plt.scatter(x_vals, y_vals, s=15, label="Observed", alpha=0.7)
plt.plot(x_pred, y_pred, color='red', label="Predicted Curve", linewidth=2)
plt.title("Fitted Parametric Curve")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
