import numpy as np
import matplotlib.pyplot as plt


# 1. Load saved results

data = np.load("Seed_Results.npy", allow_pickle=True)

# 2. Extract columns
drones = data[:, 0].astype(int)          # number of drones
pdr    = data[:, 2].astype(float)        # packet delivery ratio (PDR)
delay  = data[:, 3].astype(float)        # average end‑to‑end delay (s)
energy = data[:, 4].astype(float)        # total energy consumed

# 4. Plot (three separate figures)

unique_drones = np.unique(drones)

#Compute stadanrd deviation for each metric
def avg_std(values):
    means, stds = [], []
    for nd in unique_drones:
        mask = (drones == nd)
        means.append(np.mean(values[mask]))
        stds.append(np.std(values[mask]))
    return means, stds

pdr_avg, pdr_std       = avg_std(pdr)
delay_avg, delay_std   = avg_std(delay)
energy_avg, energy_std = avg_std(energy)

# --- PDR ---
plt.figure(figsize=(7,5))
plt.errorbar(unique_drones, pdr_avg, yerr=pdr_std, marker='o', capsize=5,
             color='blue', label='Baseline (fixed ω=0.8)')
plt.xlabel('Number of Drones')
plt.ylabel('Packet Delivery Ratio')
plt.title('PDR vs Number of Drones')
plt.ylim(0, 1.0)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.savefig('pdr_vs_drones.png', dpi=150)
plt.show()

