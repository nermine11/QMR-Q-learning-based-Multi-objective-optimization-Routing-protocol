#!/usr/bin/env python3
"""
Batch experiment: average path lifetime vs. number of nodes.
Runs QMR for different node counts (with multiple random seeds),
collects the average bottleneck energy (lifetime) per run,
and plots the results.
Usage:
  python batch_experiment.py --min-nodes 5 --max-nodes 30 --step 5 --mode dynamic
"""

import argparse
import math
import random
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')          # non-interactive backend – no popup windows
import matplotlib.pyplot as plt

# ================== 1. QMAR class ==================
class QMAR:
    def __init__(self, drone, simulator, mode='fixed'):
        self.drone = drone
        self.simulator = simulator
        self.mode = mode
        self.maxReward = 5
        self.minReward = -5
        self.max_delay = 1.0

    def feedback(self, outcome, id_j, Q_value_best_action):
        alpha = self.drone.neighbor_table[id_j, 10]
        gamma = self.drone.neighbor_table[id_j, 7]
        Q_value_i_j = self.drone.neighbor_table[id_j, 9]

        if outcome == 1:
            self.drone.neighbor_table[id_j, 9] = Q_value_i_j + alpha * self.maxReward
        elif outcome == 0:
            delay = self.drone.neighbor_table[id_j, 8] + self.drone.neighbor_table[id_j, 11]
            reward = self.computeReward(delay, id_j)
            self.drone.neighbor_table[id_j, 9] = Q_value_i_j + alpha * (
                reward + gamma * Q_value_best_action - Q_value_i_j)
        else:
            self.drone.neighbor_table[id_j, 9] = Q_value_i_j + alpha * (
                self.minReward + gamma * Q_value_best_action - Q_value_i_j)

    def computeReward(self, delay, neighbor_id):
        if delay > self.max_delay:
            self.max_delay = delay
        neighbor = next(d for d in self.simulator.drones if d.identifier == neighbor_id)
        e_neighbor = neighbor.residual_energy / neighbor.initial_energy
        norm_delay = delay / self.max_delay
        exp_delay = math.exp(-norm_delay)

        if self.mode == 'dynamic':
            my_energy_ratio = self.drone.residual_energy / self.drone.initial_energy
            w = 0.9 if my_energy_ratio < 0.5 else 0.3
        else:
            w = 0.8
        return w * exp_delay + (1 - w) * e_neighbor

    def relay_selection(self, opt_neighbors, data):
        packet = data[0]
        candidates = []
        candidates2 = []

        for node_j in self.simulator.drones:
            if node_j not in opt_neighbors:
                continue
            j = node_j.identifier
            deadline = 2001 - (self.simulator.cur_step - packet.time_step_creation)
            if deadline <= 0:
                deadline = 1
            dist_i = math.dist(self.drone.coords, self.simulator.depot_coordinates)
            req_v = dist_i / deadline
            actual_v, dist_ij = self.computeActualVel(j, node_j, dist_i)

            if actual_v >= req_v:
                LQ = self.drone.neighbor_table[j, 12]
                R = self.drone.communication_range
                M = 0 if dist_ij > R else 1 - (dist_ij / R)
                k = M * LQ
                candidates.append((node_j, k))
            elif actual_v > 0:
                candidates2.append((node_j, actual_v))

        if not candidates:
            if candidates2:
                chosen = max(candidates2, key=lambda x: x[1])[0]
            else:
                return "RHP"
        else:
            best_candidates = []
            maxx = -float('inf')
            for cand, k in candidates:
                Q_val = self.drone.neighbor_table[cand.identifier, 9]
                weighted = Q_val * k
                if weighted > maxx:
                    maxx = weighted
                    best_candidates = [cand]
                elif weighted == maxx:
                    best_candidates.append(cand)
            chosen = random.choice(best_candidates)

        if random.random() < 0.1 and opt_neighbors:
            chosen = random.choice(opt_neighbors)
        return chosen

    def computeActualVel(self, j, node_j, distance_i):
        x2 = self.drone.neighbor_table[j, 4]
        y2 = self.drone.neighbor_table[j, 5]
        x1 = self.drone.neighbor_table[j, 0]
        y1 = self.drone.neighbor_table[j, 1]
        angle_j = math.atan((y2 - y1) / (x2 - x1)) if (x2 - x1) != 0 else 0.0
        delay = self.drone.neighbor_table[j, 8] + self.drone.neighbor_table[j, 11]
        if delay == 0:
            delay = 0.01
        t1 = self.drone.neighbor_table[j, 6]
        t3 = self.simulator.cur_step + delay
        x = x1 + node_j.speed * math.cos(angle_j) * (t3 - t1)
        y = y1 + node_j.speed * math.sin(angle_j) * (t3 - t1)
        distance_j = math.dist((x, y), self.simulator.depot_coordinates)
        distance_ij = math.dist(self.drone.coords, (x, y))
        return (distance_i - distance_j) / delay, distance_ij


# ================== 2. Mock environment ==================
class MockDrone:
    def __init__(self, identifier, coords, energy, max_drones, speed=0.0, comm_range=200):
        self.identifier = identifier
        self.coords = coords
        self.residual_energy = energy
        self.initial_energy = 100.0
        self.speed = speed
        self.neighbor_table = np.zeros((max_drones, 13))
        self.neighbor_table[:, 9] = 0.5
        self.communication_range = comm_range

class MockSimulator:
    def __init__(self, drones, depot_coords):
        self.drones = drones
        self.depot_coordinates = depot_coords
        self.cur_step = 0

class MockPacket:
    def __init__(self, dest, creation_step=0):
        self.time_step_creation = creation_step


# ================== 3. Random geometric graph ==================
def build_random_graph(n_nodes, width, length, comm_range, seed):
    random.seed(seed)
    G = nx.DiGraph()
    for i in range(n_nodes):
        x = random.uniform(0, width)
        y = random.uniform(0, length)
        energy = random.randint(0, 100)
        G.add_node(i, coords=(x, y), energy=energy)

    for i in range(n_nodes):
        xi, yi = G.nodes[i]['coords']
        for j in range(n_nodes):
            if i == j:
                continue
            xj, yj = G.nodes[j]['coords']
            dist = math.dist((xi, yi), (xj, yj))
            if dist < comm_range:
                delay = dist * 0.1
                G.add_edge(i, j, delay=delay)
    return G


# ================== 4. Convert graph to mock drones ==================
def graph_to_mock_2d(G, depot_coords=None, comm_range=200):
    nodes = list(G.nodes())
    n = len(nodes)
    max_drones = n

    drones = []
    drone_dict = {}
    for node in nodes:
        energy = G.nodes[node]['energy']
        pos = G.nodes[node]['coords']
        drone = MockDrone(node, pos, energy, max_drones, speed=0.0, comm_range=comm_range)
        drone.residual_energy = energy
        drones.append(drone)
        drone_dict[node] = drone

    for u, v, data in G.edges(data=True):
        delay = data['delay']
        # u -> v
        src = drone_dict[u]
        dst = drone_dict[v]
        tbl = src.neighbor_table
        tbl[v, 0] = dst.coords[0] - 10
        tbl[v, 1] = dst.coords[1] - 10
        tbl[v, 4] = dst.coords[0]
        tbl[v, 5] = dst.coords[1]
        tbl[v, 6] = 0
        tbl[v, 7] = 0.9
        tbl[v, 8] = delay
        tbl[v, 9] = 0.5
        tbl[v,10] = 0.3
        tbl[v,11] = 0
        tbl[v,12] = 1.0

        # v -> u
        src = drone_dict[v]
        dst = drone_dict[u]
        tbl = src.neighbor_table
        tbl[u, 0] = dst.coords[0] - 10
        tbl[u, 1] = dst.coords[1] - 10
        tbl[u, 4] = dst.coords[0]
        tbl[u, 5] = dst.coords[1]
        tbl[u, 6] = 0
        tbl[u, 7] = 0.9
        tbl[u, 8] = delay
        tbl[u, 9] = 0.5
        tbl[u,10] = 0.3
        tbl[u,11] = 0
        tbl[u,12] = 1.0

    if depot_coords is None:
        depot_coords = (0, 0)
    sim = MockSimulator(drones, depot_coords)
    return sim, drone_dict


# ================== 5. Forward packet ==================
def forward_packet(start_id, dest_id, drone_dict, qmar_dict, sim, mode):
    current_id = start_id
    path = [current_id]
    prev_id = None

    for _ in range(20):
        if current_id == dest_id:
            break
        if current_id not in qmar_dict:
            qmar_dict[current_id] = QMAR(drone_dict[current_id], sim, mode)
        qmar = qmar_dict[current_id]
        current_drone = drone_dict[current_id]

        neighbor_rows = np.where(current_drone.neighbor_table[:, 8] != 0)[0]
        opt_neighbors = [drone_dict[i] for i in neighbor_rows
                         if i != current_id and i != prev_id]

        if not opt_neighbors:
            break

        chosen = qmar.relay_selection(opt_neighbors, (MockPacket(dest_id),))
        if chosen == "RHP":
            break

        next_id = chosen.identifier
        path.append(next_id)
        chosen_drone = drone_dict[next_id]
        max_q = np.max(chosen_drone.neighbor_table[:, 9]) if np.any(chosen_drone.neighbor_table[:, 9]) else 0.5
        qmar.feedback(0, next_id, max_q)
        prev_id = current_id
        current_id = next_id

    return path


# ================== 6. Single simulation run ==================
def run_single(n_nodes, width, length, comm_range, packets, seed, mode):
    """Return the average bottleneck energy (lifetime) across packets."""
    G = build_random_graph(n_nodes, width, length, comm_range, seed)

    # Choose random source and destination (different nodes)
    nodes_list = list(G.nodes())
    src, dst = random.sample(nodes_list, 2)

    depot_coords = G.nodes[dst]['coords']
    sim, drone_dict = graph_to_mock_2d(G, depot_coords, comm_range)

    qmar_dict = {}
    lifetimes = []

    for _ in range(packets):
        path_ids = forward_packet(src, dst, drone_dict, qmar_dict, sim, mode)
        if len(path_ids) < 2:   # no route or only source
            lifetimes.append(0.0)   # worst case: path dead
        else:
            energies = [G.nodes[n]['energy'] for n in path_ids]
            bottleneck = min(energies)
            lifetimes.append(bottleneck)

    return np.mean(lifetimes) if lifetimes else 0.0


# ================== 7. Batch experiment ==================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Batch QMR lifetime vs nodes')
    parser.add_argument('-m', '--mode', choices=['fixed', 'dynamic'], default='dynamic')
    parser.add_argument('--min-nodes', type=int, default=5)
    parser.add_argument('--max-nodes', type=int, default=30)
    parser.add_argument('--step', type=int, default=5)
    parser.add_argument('-W', '--width', type=float, default=500.0)
    parser.add_argument('-L', '--length', type=float, default=500.0)
    parser.add_argument('-r', '--range', type=float, default=200.0)
    parser.add_argument('-p', '--packets', type=int, default=50)
    parser.add_argument('--seeds', type=int, default=10, help='Number of random seeds per node count')
    args = parser.parse_args()

    node_counts = list(range(args.min_nodes, args.max_nodes + 1, args.step))
    avg_lifetimes = []
    std_lifetimes = []

    print(f"Running batch experiment: mode={args.mode}, nodes={args.min_nodes}..{args.max_nodes} step {args.step}, seeds={args.seeds}")
    print(f"Area {args.width}x{args.length}, range {args.range}, packets {args.packets}")

    for n in node_counts:
        values = []
        for s in range(args.seeds):
            seed = s * 100 + n   # spread seeds
            avg_life = run_single(n, args.width, args.length, args.range, args.packets, seed, args.mode)
            values.append(avg_life)
        mean_val = np.mean(values)
        std_val = np.std(values)
        avg_lifetimes.append(mean_val)
        std_lifetimes.append(std_val)
        print(f"  n={n:3d}: avg lifetime {mean_val:.1f} min (±{std_val:.1f})")

    # Save data
    data = np.column_stack((node_counts, avg_lifetimes, std_lifetimes))
    # Save data to CSV
    csv_filename = f'batch_results_{args.mode}.csv'
    np.savetxt(csv_filename, data, header='nodes,avg_lifetime,std_lifetime', delimiter=',', comments='')
    print(f"Saved {csv_filename}")
