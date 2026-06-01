#!/usr/bin/env python3
"""
Random geometric graph QMR simulation (fixed or dynamic ω).
Visualization in a separate function.
Usage:
  python random_qmr.py --mode dynamic --width 600 --length 600 --nodes 15 --range 250
"""

import argparse
import math
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# ================== 1. QMAR class ==================
class QMAR:
    def __init__(self, drone, simulator, mode='fixed'):
        self.drone = drone
        self.simulator = simulator
        self.mode = mode          # 'fixed' or 'dynamic'
        self.maxReward = 5
        self.minReward = -5
        self.max_delay = 1.0      # for delay normalisation

    def feedback(self, outcome, id_j, Q_value_best_action):
        alpha = self.drone.neighbor_table[id_j, 10]
        gamma = self.drone.neighbor_table[id_j, 7]
        Q_value_i_j = self.drone.neighbor_table[id_j, 9]

        if outcome == 1:          # reached destination
            self.drone.neighbor_table[id_j, 9] = Q_value_i_j + alpha * self.maxReward
        elif outcome == 0:        # normal forward
            delay = self.drone.neighbor_table[id_j, 8] + self.drone.neighbor_table[id_j, 11]
            reward = self.computeReward(delay, id_j)
            self.drone.neighbor_table[id_j, 9] = Q_value_i_j + alpha * (
                reward + gamma * Q_value_best_action - Q_value_i_j)
        else:                     # failure
            self.drone.neighbor_table[id_j, 9] = Q_value_i_j + alpha * (
                self.minReward + gamma * Q_value_best_action - Q_value_i_j)

    def computeReward(self, delay, neighbor_id):
        # update max delay
        if delay > self.max_delay:
            self.max_delay = delay

        # neighbour energy (0..1)
        neighbor = next(d for d in self.simulator.drones if d.identifier == neighbor_id)
        e_neighbor = neighbor.residual_energy / neighbor.initial_energy

        # delay score (normalised)
        norm_delay = delay / self.max_delay
        exp_delay = math.exp(-norm_delay)

        # weight
        if self.mode == 'dynamic':
            my_energy_ratio = self.drone.residual_energy / self.drone.initial_energy
            if my_energy_ratio < 0.5:
                w = 0.9          # low own battery → speed
            else:
                w = 0.3          # high own battery → energy
        else:
            w = 0.8               # fixed ω

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
                if dist_ij > R:
                    M = 0
                else:
                    M = 1 - (dist_ij / R)
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
            # random tie-breaking among best κ‑weighted Q values
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

        # ε‑greedy exploration (10 %)
        if random.random() < 0.1 and opt_neighbors:
            chosen = random.choice(opt_neighbors)

        return chosen

    def computeActualVel(self, j, node_j, distance_i):
        # positions from neighbour table
        x2 = self.drone.neighbor_table[j, 4]
        y2 = self.drone.neighbor_table[j, 5]
        x1 = self.drone.neighbor_table[j, 0]
        y1 = self.drone.neighbor_table[j, 1]

        if (x2 - x1) != 0:
            angle_j = math.atan((y2 - y1) / (x2 - x1))
        else:
            angle_j = math.atan(0)

        delay = self.drone.neighbor_table[j, 8] + self.drone.neighbor_table[j, 11]
        if delay == 0:
            delay = 0.01

        t1 = self.drone.neighbor_table[j, 6]   # timestamp
        t3 = self.simulator.cur_step + delay
        # predicted position of j at arrival
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
        self.neighbor_table[:, 9] = 0.5          # initial Q-values
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
def build_random_graph(n_nodes, width, length, comm_range, seed=42):
    random.seed(seed)
    G = nx.DiGraph()

    for i in range(n_nodes):
        x = random.uniform(0, width)
        y = random.uniform(0, length)
        energy = random.randint(10, 100)   # %
        G.add_node(i, coords=(x, y), energy=energy)

    for i in range(n_nodes):
        xi, yi = G.nodes[i]['coords']
        for j in range(n_nodes):
            if i == j:
                continue
            xj, yj = G.nodes[j]['coords']
            dist = math.dist((xi, yi), (xj, yj))
            if dist < comm_range:
                delay = dist * 0.1   # ms/m
                G.add_edge(i, j, delay=delay)
    return G


# ================== 4. Visualization function ==================
def visualize_graph(G):
    """Draw the graph with node IDs, energy and edge delays."""
    pos = {i: G.nodes[i]['coords'] for i in G.nodes()}
    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)
    node_labels = {i: f"{i}\n({G.nodes[i]['energy']}%)" for i in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=10,
                           connectionstyle='arc3, rad=0.1')
    edge_labels = {(u, v): f"{d['delay']:.1f} ms" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=7)
    plt.title("Random Geometric Graph")
    plt.axis('off')
    plt.tight_layout()

    try:
        plt.show()
    except:
        plt.savefig("random_graph.png", dpi=150)
        print("Graph saved as 'random_graph.png' – open it to see the topology.")


# ================== 5. Convert graph to mock drones ==================
def graph_to_mock_2d(G, depot_coords=None, comm_range=200):
    nodes = list(G.nodes())
    n = len(nodes)
    name_to_id = {i: i for i in nodes}   # IDs are integers
    id_to_name = name_to_id
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

    # Fill neighbour tables from edges (both directions)
    for u, v, data in G.edges(data=True):
        delay = data['delay']
        # u -> v
        src = drone_dict[u]
        dst = drone_dict[v]
        tbl = src.neighbor_table
        tbl[v, 0] = dst.coords[0] - 10   # fake previous x
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
    return sim, drone_dict, name_to_id, id_to_name


# ================== 6. Forward packet ==================
def forward_packet(start_id, dest_id, drone_dict, qmar_dict, sim):
    current_id = start_id
    path = [current_id]
    prev_id = None

    for _ in range(20):          # max hops
        if current_id == dest_id:
            break

        if current_id not in qmar_dict:
            qmar_dict[current_id] = QMAR(drone_dict[current_id], sim, mode=args.mode)

        qmar = qmar_dict[current_id]
        current_drone = drone_dict[current_id]

        # neighbors (non-zero delay, exclude self and previous)
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

        # feedback
        chosen_drone = drone_dict[next_id]
        max_q = np.max(chosen_drone.neighbor_table[:, 9]) if np.any(chosen_drone.neighbor_table[:, 9]) else 0.5
        qmar.feedback(0, next_id, max_q)

        prev_id = current_id
        current_id = next_id

    return path


# ================== 7. Main ==================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='QMR on random geometric graph')
    parser.add_argument('-m', '--mode', choices=['fixed', 'dynamic'], default='fixed')
    parser.add_argument('-W', '--width', type=float, default=500.0)
    parser.add_argument('-L', '--length', type=float, default=500.0)
    parser.add_argument('-n', '--nodes', type=int, default=10)
    parser.add_argument('-r', '--range', type=float, default=200.0)
    parser.add_argument('-p', '--packets', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print(f"Mode: {args.mode.upper()} ω")
    print("Assumption: 1% battery = 1 minute of activity\n")

    # Build random graph
    G = build_random_graph(args.nodes, args.width, args.length, args.range, seed=args.seed)

    # Show the graph
    visualize_graph(G)

    # ---------- User chooses source / destination ----------
    default_source = 0
    default_dest = args.nodes - 1
    try:
        src_input = input(f"Source node (default {default_source}): ").strip()
        source = int(src_input) if src_input else default_source
        dst_input = input(f"Destination node (default {default_dest}): ").strip()
        dest = int(dst_input) if dst_input else default_dest
    except (ValueError, EOFError):
        source, dest = default_source, default_dest

    if source not in G.nodes() or dest not in G.nodes():
        print("Invalid node IDs, using defaults.")
        source, dest = default_source, default_dest

    print(f"Source: {source}, Destination: {dest}\n")

    # ---------- Convert to mock environment ----------
    depot_coords = G.nodes[dest]['coords']
    sim, drone_dict, name_to_id, id_to_name = graph_to_mock_2d(G, depot_coords, comm_range=args.range)

    # ---------- Run simulation ----------
    qmar_dict = {}
    lifetimes = []                              # <-- new list

    for pkt in range(1, args.packets + 1):
        path_ids = forward_packet(source, dest, drone_dict, qmar_dict, sim)
        path_str = ' → '.join(str(n) for n in path_ids)
        print(f"Packet {pkt}: {path_str}")

        energies = [G.nodes[n]['energy'] for n in path_ids]
        bottleneck = min(energies)
        lifetimes.append(bottleneck)            # <-- store lifetime
        print(f"         Bottleneck energy: {bottleneck}% → lifetime {bottleneck} min")

    # After all packets, compute and print the average lifetime
    if lifetimes:
        avg_lifetime = sum(lifetimes) / len(lifetimes)
        print(f"\nAverage path lifetime: {avg_lifetime:.1f} min")
    else:
        print("\nNo packets delivered, cannot compute average lifetime.")