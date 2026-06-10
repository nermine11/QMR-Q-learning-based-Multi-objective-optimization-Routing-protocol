import numpy as np
import matplotlib.pyplot as plt

# Load the two CSV files (adjust filenames if needed)
fixed = np.loadtxt('experiment_full_fixed.csv', delimiter=',', skiprows=1)
dynamic = np.loadtxt('experiment_full_dynamic.csv', delimiter=',', skiprows=1)

# Extract columns
nodes_fixed, avg_fixed, std_fixed = fixed[:,0], fixed[:,1], fixed[:,2]
nodes_dyn, avg_dyn, std_dyn = dynamic[:,0], dynamic[:,1], dynamic[:,2]

# Plot
plt.figure(figsize=(8, 5))
plt.errorbar(nodes_fixed, avg_fixed, yerr=None, marker='o', capsize=5,
             label='Fixed ω = 0.8', color='blue')
plt.errorbar(nodes_dyn, avg_dyn, yerr=None, marker='s', capsize=5,
             label='Dynamic ω', color='red')
plt.xlabel('Number of nodes')
plt.ylabel('Average path lifetime (min)')
plt.title('QMR lifetime: fixed vs dynamic ω')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('comparison_lifetime.png', dpi=150)
plt.show()