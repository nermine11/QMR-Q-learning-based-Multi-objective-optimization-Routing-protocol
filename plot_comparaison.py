import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

# Load both CSV files
fixed = pd.read_csv('experiment_full_fixed.csv')
dynamic = pd.read_csv('experiment_full_dynamic.csv')

nodes_fixed = fixed['nodes'].values
lifetime_fixed = fixed['avg_lifetime_min'].values
nodes_dyn = dynamic['nodes'].values
lifetime_dyn = dynamic['avg_lifetime_min'].values

# Smoothing factor : controls the trade‑off between closeness to data and smoothness.
# We can increase s for a smoother curve ( 1.0 to 5.0)
s = 1.0

# Create smoothing splines (piecewise cubic, Bézier‑like)
spline_fixed = UnivariateSpline(nodes_fixed, lifetime_fixed, s=s)
spline_dyn   = UnivariateSpline(nodes_dyn, lifetime_dyn, s=s)

# Dense x‑axis for a perfectly smooth line
x_dense = np.linspace(nodes_fixed.min(), nodes_fixed.max(), 300)

plt.figure(figsize=(8,5))



# Smooth Bézier‑like curves
plt.plot(x_dense, spline_fixed(x_dense), '-', color='blue', linewidth=2, label='Fixed ω')
plt.plot(x_dense, spline_dyn(x_dense), '--', color='red', linewidth=2, label='Dynamic ω')

plt.xlabel('Number of drones')
plt.ylabel('Average bottleneck lifetime (min)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('comparaison1000.png', dpi=150)
plt.show()