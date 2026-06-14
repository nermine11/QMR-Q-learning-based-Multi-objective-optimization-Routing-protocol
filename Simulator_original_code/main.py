import numpy as np
import multiprocessing
from src.simulation.simulator import Simulator
from src.utilities.policies import *


def main():
    """ the place where to run simulations and experiments. """

    drones = range(5,35,5)

    #grid_search(drones, alphas, gammas, divs, epsilons, Epsilon(), negRewards)
    seed_results = []
    try:
        for nb_drones in drones:
            # run the experiment 30 times with a different sed
            for seed in range(1, 31):
                sim = Simulator(nb_drones, seed, simulation_name=f"test_drone{nb_drones}_seed{seed}")
                sim.run()
                # -- Packet Delivery Ratio (PDR) --
                pdr = len(sim.metrics.drones_packets_to_depot) / sim.metrics.all_data_packets_in_simulation
                # -- Average End‑to‑End Delay --
                sim.metrics.other_metrics()
                avg_delay = sim.metrics.packet_mean_delivery_time
                # -- Total Energy Consumed (all drones) --
                energy_consumed = sum(d.initial_energy - d.residual_energy for d in sim.drones)
                # Store the results
                seed_results.append((nb_drones, seed, pdr, avg_delay, energy_consumed))
                sim.close()
    finally:
        # This runs even if you press Ctrl+C
        np.save("./Seed_Results.npy", np.array(seed_results))
        print(seed_results)


if __name__ == "__main__":
    main()
