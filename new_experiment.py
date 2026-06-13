"""

Experiment using QMR
Setup:
- Same destintation not a node but a ground station (no energy, can't relay packets)
- the number of nodes nb_nodes in the experiment goes from 1 to 49 
- For each n nb_nodes we create 1-50 topologies
- For each topology, each node becomes a source and sends 1000 packets to the destination
- We count the average of lifetimes of each path: whichi is the bottlenck energy of the path ( the least energy of a drone in the path)
We do these two experiments for w_dynamic and w_fixed

"""

import argparse
import math
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import sys

IDLE_ENERGY = 10.0
TRANSMISSION_ENERGY = 15.0
TOPOLOGIES_PER_COUNT = 49
NB_ATTEMPTS = 500
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
            print(my_energy_ratio)
            if my_energy_ratio < 0.5:          # low battery → prioritise speed
                w = 0.8
            else:    
                                         # high battery → prioritise neighbour energy
                w = 0.3
        else:
            w = 0.8
        return w * exp_delay + (1 - w) * e_neighbor

    def relay_selection(self, opt_neighbors, data):
        packet = data[0]
        candidates = []
        candidates2 = []

        # Always print debug info for any drone that runs relay_selection
        #print(f"\n[DEBUG] Drone {self.drone.identifier} selecting next hop")
        #print(f"  My residual energy: {self.drone.residual_energy:.2f}%")
        #print(f"  My position: {self.drone.coords}, Depot: {self.simulator.depot_coordinates}")

        for node_j in self.simulator.drones:
            if node_j not in opt_neighbors:
                continue
            j = node_j.identifier
            deadline = 2001 - (self.simulator.cur_step - packet.time_step_creation)
            if deadline <= 0:
                deadline = 1
            dist_i = math.dist(self.drone.coords, self.simulator.depot_coordinates)
            req_v = dist_i / deadline # required velocity to reach destination
            actual_v, dist_ij = self.computeActualVel(j, node_j, dist_i)

            #print(f"  Neighbor {j}: residual_energy={node_j.residual_energy:.2f}%, dist_ij={dist_ij:.1f}, actual_v={actual_v:.3f}, req_v={req_v:.3f}")

            if actual_v >= req_v:
                LQ = self.drone.neighbor_table[j, 12] # always 1
                R = self.drone.communication_range
                M = 0 if dist_ij > R else 1 - (dist_ij / R)
                k= M * LQ
                candidates.append((node_j, k))
                #print(f"    -> candidate: M={M:.3f}, LQ={LQ:.3f}, k={k:.3f}")
            elif actual_v > 0:
                candidates2.append((node_j, actual_v))
                #print(f"    -> secondary list (actual_v>0 but < req_v)")
            else:
                pass
                #print(f"    -> discarded (actual_v <= 0)")

        if not candidates:
            if candidates2:
                chosen = max(candidates2, key=lambda x: x[1])[0]
            else:
                #print(f"  No candidates at all -> RHP")
                return "RHP"
        else:
                maxx = -100000
                chosen = None
                #print("  Candidate scores:")
                for i in range(len(candidates)):
                    candidate = candidates[i][0]
                    k = candidates[i][1]
                    Q_val = self.drone.neighbor_table[candidate.identifier, 9]
                    weighted = Q_val * k
                    #print(f"    {candidate.identifier}: Q={Q_val:.3f}, k={k:.3f}, weighted={weighted:.3f}")
                    if (Q_val * k > maxx):
                        chosen = candidate
                        maxx = Q_val * k

        if random.random() < 0.1 and opt_neighbors:
            prev = chosen
            chosen = random.choice(opt_neighbors)
            #print(f"  ε-greedy override: was {prev.identifier if prev else 'None'}, now randomly picked {chosen.identifier}")

        #print(f"  Final choice: {chosen.identifier if chosen != 'RHP' else 'RHP'}\n")
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


class MockDrone:
    def __init__(self, identifier, coords, energy, max_drones, speed=0.0, comm_range=200):
        self.identifier = identifier
        self.coords = coords
        self.initial_energy = energy
        self.residual_energy = energy
        self.speed = speed
        self.neighbor_table = np.zeros((max_drones, 13))
        self.neighbor_table[:, 9] = 0.5
        self.communication_range = comm_range


    def consume_energy_idle(self, time_minutes):
        """Idle drain: 10% per minute. We use transmission duration to define a minute"""
        loss = IDLE_ENERGY * time_minutes
        self.residual_energy = max(0.0, self.residual_energy - loss)

    def consume_energy_tx(self):
        """Transmission cost: 15 percentage points."""
        self.residual_energy = max(0.0, self.residual_energy - TRANSMISSION_ENERGY)


class MockSimulator:
    def __init__(self, drones, depot_coords):
        self.drones = drones
        self.depot_coordinates = depot_coords
        self.cur_step = 0


class MockPacket:
    def __init__(self, dest, creation_step=0):
        self.time_step_creation = creation_step



# =============================================================================
# 3. Random geometric graph with fixed ground station
# distance between drones >= 10 meters
# graph is connected
# =============================================================================
def build_connected_graph(n_nodes, width, length, comm_range, seed, min_dist=10.0):
    """
    Drones have random initial energy 10‑100%. Ground station at centre, infinite energy.
    Returns (G, station_id) or None.
    """
    random.seed(seed)
    station_id = n_nodes
    station_coords = (width / 2.0, length / 2.0)

    for attempt in range(200):
        G = nx.DiGraph()
        coords = []

        # Place drones with min_dist
        for i in range(n_nodes):
            placed = False
            for _ in range(100):
                x = random.uniform(0, width)
                y = random.uniform(0, length)
                if all(math.dist((x, y), p) >= min_dist for p in coords):
                    coords.append((x, y))
                    placed = True
                    break
            if not placed:
                break
        if len(coords) < n_nodes:
            continue

        # Add drones with random energy (10‑100%)
        for i, (x, y) in enumerate(coords):
            energy = random.randint(10, 100)   # percent
            G.add_node(i, coords=(x, y), energy=energy)

        # Add station (infinite energy, ignored in metrics)
        G.add_node(station_id, coords=station_coords, energy=float('inf'))

        # Edges between drones only (both directions)
        for i in range(n_nodes):
            xi, yi = G.nodes[i]['coords']
            for j in range(i + 1, n_nodes):
                xj, yj = G.nodes[j]['coords']
                dist = math.dist((xi, yi), (xj, yj))
                if dist < comm_range:
                    delay = dist * 0.1   # ms
                    G.add_edge(i, j, delay=delay)
                    G.add_edge(j, i, delay=delay)

        # Temporarily add direct edges to station for connectivity check
        temp_edges = []
        for i in range(n_nodes):
            xi, yi = G.nodes[i]['coords']
            if math.dist((xi, yi), station_coords) < comm_range:
                G.add_edge(i, station_id, delay=0)
                temp_edges.append(i)

        ok = all(nx.has_path(G, i, station_id) for i in range(n_nodes))

        # Remove temporary edges
        for i in temp_edges:
            G.remove_edge(i, station_id)

        if ok:
            return G, station_id
    return None



# =============================================================================
# 4. Convert graph to mock drones (station is NOT a drone)
# =============================================================================
def graph_to_mock_2d(G, station_id, mode, comm_range=200):
    drones = []
    drone_dict = {}
    max_drones = len(G.nodes())

    for node in G.nodes():
        if node == station_id:
            continue
        energy = G.nodes[node]['energy']
        pos = G.nodes[node]['coords']
        drone = MockDrone(node, pos, energy, max_drones, speed=0.0, comm_range=comm_range)
        drone.residual_energy = energy
        drones.append(drone)
        drone_dict[node] = drone

    # Fill neighbour tables (only drone‑to‑drone edges)
    for u, v, data in G.edges(data=True):
        if u == station_id or v == station_id:
            continue
        delay = data['delay']
        # u -> v
        src_drone = drone_dict[u]
        dst_drone = drone_dict[v]
        tbl = src_drone.neighbor_table
        tbl[v, 0] = dst_drone.coords[0] - 10
        tbl[v, 1] = dst_drone.coords[1] - 10
        tbl[v, 4] = dst_drone.coords[0]
        tbl[v, 5] = dst_drone.coords[1]
        tbl[v, 6] = 0
        tbl[v, 7] = 0.9
        tbl[v, 8] = delay
        tbl[v, 9] = 0.5
        tbl[v,10] = 0.3
        tbl[v,11] = 0
        tbl[v,12] = 1.0

        # v -> u
        src_drone = drone_dict[v]
        dst_drone = drone_dict[u]
        tbl = src_drone.neighbor_table
        tbl[u, 0] = dst_drone.coords[0] - 10
        tbl[u, 1] = dst_drone.coords[1] - 10
        tbl[u, 4] = dst_drone.coords[0]
        tbl[u, 5] = dst_drone.coords[1]
        tbl[u, 6] = 0
        tbl[u, 7] = 0.9
        tbl[u, 8] = delay
        tbl[u, 9] = 0.5
        tbl[u,10] = 0.3
        tbl[u,11] = 0
        tbl[u,12] = 1.0
    if mode =='dynamic':
        for drone in drones:
            for nbr in range(max_drones):
                if drone.neighbor_table[nbr, 8] != 0:   # neighbour exists (delay > 0)
                    if nbr != station_id:                # skip the ground station
                        # Column 9 = Q‑value; set it to energy ratio (0…1)
                        drone.neighbor_table[nbr, 9] = drone_dict[nbr].residual_energy / 100.0
    depot_coords = G.nodes[station_id]['coords']
    sim = MockSimulator(drones, depot_coords)
    return sim, drone_dict



# =============================================================================
# 5. Forward a packet with energy consumption
# =============================================================================
def forward_packet_with_energy(start_id, dest_id, drone_dict, qmar_dict, sim, mode):
    current_id = start_id
    path = [current_id]
    prev_id = None

    for _ in range(20):
        if current_id == dest_id:
            break
        current_drone = drone_dict[current_id]

        # Direct delivery check
        if math.dist(current_drone.coords, sim.depot_coordinates) < current_drone.communication_range:
            current_drone.consume_energy_tx()
            if current_drone.residual_energy <= 0:
                return None
            path.append(dest_id)
            return path

        # Relay transmission
        current_drone.consume_energy_tx()
        if current_drone.residual_energy <= 0:
            return None

        if current_id not in qmar_dict:
            qmar_dict[current_id] = QMAR(current_drone, sim, mode)
        qmar = qmar_dict[current_id]

        neighbor_rows = np.where(current_drone.neighbor_table[:, 8] != 0)[0]
        opt_neighbors = [drone_dict[i] for i in neighbor_rows
                         if i != current_id and i != prev_id]

        if not opt_neighbors:
            return None

        # choose the next hop using qmr
        chosen = qmar.relay_selection(opt_neighbors, (MockPacket(dest_id),))
        if chosen == "RHP":
            return None

        next_id = chosen.identifier
        path.append(next_id)

        # Feedback from the chosen neighbour
        chosen_drone = drone_dict[next_id]
        max_q = np.max(chosen_drone.neighbor_table[:, 9]) if np.any(chosen_drone.neighbor_table[:, 9]) else 0.5
        qmar.feedback(0, next_id, max_q)

        prev_id = current_id
        current_id = next_id

    return path if path[-1] == dest_id else None


# =============================================================================
# 6. Simulate one source – average bottleneck lifetime (minutes)
# =============================================================================
def simulate_source(G, station_id, src_id, comm_range, mode, packets=1000, cbr_interval_ms=200):
    """
    Run CBR traffic until source dies or all packets sent.
    For each successfully delivered packet, compute bottleneck lifetime (min % / 10).
    Return average lifetime in minutes over all delivered packets.
    """
    time_per_packet_min = (cbr_interval_ms / 1000.0) / 60.0
    sim, drone_dict = graph_to_mock_2d(G, station_id, mode, comm_range)
    src_drone = drone_dict[src_id]

    qmar_dict = {}
    lifetimes = []

    for _ in range(packets):
        # Apply idle drain to all drones for the interval
        for d in drone_dict.values():
            d.consume_energy_idle(time_per_packet_min)
        if src_drone.residual_energy <= 0:
            break

        path = forward_packet_with_energy(src_id, station_id, drone_dict, qmar_dict, sim, mode)
        if path is None or path[-1] != station_id:
            continue   # packet failed, skip

        # Get energies of all drones on the path (exclude station)
        # Get the current energy of the drone
        energies = [drone_dict[n].residual_energy for n in path if n != station_id]        
        if not energies:
            continue
        min_energy = min(energies)
        # Convert to minutes: 10% = 1 minute
        lifetime_min = min_energy / 10.0
        lifetimes.append(lifetime_min)

    if not lifetimes:
        return 0.0   # no successful deliveries
    return np.mean(lifetimes)


# =============================================================================
# 7. Run experiment for a single node count
"""
For each n_nodes number of drones, generate 50 connected topologies
For each topology, let every drone be the source and send 1000 packets to the destination
average bottlenck is the average lifetime accross all 50 topologies for each n_nodes nb of drones
"""
# =============================================================================
def run_for_node_count(n_nodes, width, length, comm_range, mode, packets, cbr_interval, seed_base):
    all_lifetimes = []
    topo_count = 0
    attempt = 0

    while topo_count < TOPOLOGIES_PER_COUNT and attempt < NB_ATTEMPTS:
        seed = seed_base + attempt * 1000
        result = build_connected_graph(n_nodes, width, length, comm_range, seed, min_dist=10.0)
        attempt += 1
        if result is None:
            continue

        G, station_id = result
        for src in range(n_nodes):
            avg_life = simulate_source(G, station_id, src, comm_range, mode, packets, cbr_interval)
            all_lifetimes.append(avg_life)
        topo_count += 1

    if topo_count < TOPOLOGIES_PER_COUNT:
        print(f"Warning: n={n_nodes}: only generated {topo_count} topologies")

    if not all_lifetimes:
        return 0.0, 0.0
    return np.mean(all_lifetimes), np.std(all_lifetimes)




def visualize_source_paths(G, station_id, src_id, comm_range, mode,
                           packets=1000, cbr_interval_ms=200, max_paths=10):
    """
    Simulate a single source and collect distinct paths.
    Draw the network with the station in red and each distinct path in a different colour.
    """
    time_per_packet_min = (cbr_interval_ms / 1000.0) / 60.0
    sim, drone_dict = graph_to_mock_2d(G, station_id, mode ,comm_range)
    src_drone = drone_dict[src_id]
    #src_drone.residual_energy = 90.0
    #src_drone.initial_energy = 100.0
    qmar_dict = {}
    distinct_paths = set()          # use tuples of node IDs to store unique paths
    all_paths_ordered = []          # store paths in order of first occurrence

    # send 1000 packets and see the paths taken
    for _ in range(packets):
        # Apply idle drain (same as simulate_source)
        for d in drone_dict.values():
            d.consume_energy_idle(time_per_packet_min)
        if src_drone.residual_energy <= 0:
            break

        path = forward_packet_with_energy(src_id, station_id, drone_dict, qmar_dict, sim, mode)
        if path is None or path[-1] != station_id:
            continue

        # Convert to tuple (hashable) and check if it's new
        path_tuple = tuple(path)
        if path_tuple not in distinct_paths:
            distinct_paths.add(path_tuple)
            all_paths_ordered.append(list(path_tuple))   # store as list for later
            if len(distinct_paths) >= max_paths:
                break

    if not all_paths_ordered:
        print("No successful paths found – cannot visualise.")
        return
    else:
        print(all_paths_ordered)
    # ---------- Drawing ----------
    pos = {node: G.nodes[node]['coords'] for node in G.nodes()}
    plt.figure(figsize=(10, 8))

    # Draw all edges in light grey
    nx.draw_networkx_edges(G, pos, edge_color='lightgrey', arrows=True, arrowsize=10)

    # Draw all drones (exclude station) in light blue
    drone_nodes = [n for n in G.nodes() if n != station_id]
    nx.draw_networkx_nodes(G, pos, nodelist=drone_nodes, node_color='lightblue', node_size=400)

    # Draw the ground station in red (larger size)
    nx.draw_networkx_nodes(G, pos, nodelist=[station_id], node_color='red', node_size=600,
                           node_shape='s')   # square for distinction

    # Label all nodes with ID and energy (for drones) / "Station" for station
    node_labels = {}
    for n in G.nodes():
        if n == station_id:
            node_labels[n] = "Station"
        else:
            node_labels[n] = f"{n}\n({G.nodes[n]['energy']}%)"
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=7)

    # Colour map for paths
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_paths_ordered)))

    # Draw each path with a different colour
    for idx, path in enumerate(all_paths_ordered):
        # edges of the path
        path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color=[colors[idx]],
                               width=2.5, arrows=True, arrowsize=15)

        # highlight nodes on the path (except station, which is already red)
        path_nodes = [n for n in path if n != station_id]
        nx.draw_networkx_nodes(G, pos, nodelist=path_nodes, node_color=[colors[idx]],
                               node_size=400, alpha=0.6)

    # Create a simple legend for the paths
    legend_handles = [Line2D([0], [0], color=colors[i], lw=2, label=f'Path {i+1}')
                      for i in range(len(all_paths_ordered))]
    plt.legend(handles=legend_handles, loc='upper left', fontsize='small')

    plt.title(f"QMR paths from {src_id} to station (mode={mode})")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'path_visualization_{mode}.png', dpi=150)
    plt.show()

def build_illustrative_graph():
    G = nx.DiGraph()
    
    # Use integer IDs with a name map
    # A=0, B=1, C=2, D=3, E=4
    id_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    
    G.add_node(0, coords=(500, 0), energy=80)
    G.add_node(1, coords=(350, 0), energy=30)
    G.add_node(2, coords=(400, 150), energy=100)
    G.add_node(3, coords=(0, 0), energy=float('inf'))   # station
    G.add_node(4, coords=(200, 150), energy=100)
    
    G.add_edge(0, 1, delay=5)
    G.add_edge(0, 2, delay=25)
    G.add_edge(1, 3, delay=25)
    G.add_edge(2, 4, delay=25)
    G.add_edge(4, 3, delay=25)
    
    return G, 3   # station ID = 3


# =============================================================================
# 8. Main batch experiment
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Full energy‑aware QMR with bottleneck lifetime')
    parser.add_argument('-m', '--mode', choices=['fixed', 'dynamic'], default='dynamic')
    parser.add_argument('--min-nodes', type=int, default=1)
    parser.add_argument('--max-nodes', type=int, default=199)
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('-W', '--width', type=float, default=500.0)
    parser.add_argument('-L', '--length', type=float, default=500.0)
    parser.add_argument('-r', '--range', type=float, default=200.0)
    parser.add_argument('-p', '--packets', type=int, default=1000)
    parser.add_argument('--cbr-interval', type=float, default=200.0, help='CBR interval (ms)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize paths for a single source instead of running the batch experiment')
    parser.add_argument('--vis-nodes', type=int, default=8,
                        help='Number of drones for the visualisation (used with --visualize)')
    parser.add_argument('--vis-source', type=int, default=0,
                        help='Source node for the visualisation')
    parser.add_argument('--vis-seed', type=int, default=42,
                        help='Random seed for the topology in visualisation')
    parser.add_argument('--vis-range', type=float, default=None,
                    help='Communication range for visualization (default: same as --range)')
    args = parser.parse_args()

    """
    python3 new_experiment.py --visualize 
    --mode dynamic --vis-nodes 10 --vis-source 3 --vis-seed 42 --vis-range 120
    """
    if args.visualize:
        # ---- Visualisation mode ----
        print(f"Visualisation: nodes={args.vis_nodes}, source={args.vis_source}, seed={args.vis_seed}")
        vis_range = args.vis_range if args.vis_range is not None else args.range
        result = build_connected_graph(args.vis_nodes, args.width, args.length,
                                    vis_range, args.vis_seed, min_dist=10.0)
        if result is None:
            print("Could not generate a connected topology with the given parameters.")
            sys.exit(1)
        G, station_id = result
        # Use the hand‑crafted topology instead of a random one
        G, station_id = build_illustrative_graph()
    
        # The source is always "A" in this example
        src = 0
        # ---- Fixed ω ----
        print("\n--- Fixed ω ---")
        visualize_source_paths(G, station_id, args.vis_source, vis_range, 'fixed',
                            packets=args.packets, cbr_interval_ms=args.cbr_interval,
                            max_paths=10)

        # ---- Dynamic ω ----
        print("\n--- Dynamic ω ---")
        visualize_source_paths(G, station_id, args.vis_source, vis_range, 'dynamic',
                            packets=args.packets, cbr_interval_ms=args.cbr_interval,
                            max_paths=10)

        sys.exit(0)
    node_counts = list(range(args.min_nodes, args.max_nodes + 1, args.step))
    avg_lifetimes = []
    std_lifetimes = []

    print(f"Experiment: mode={args.mode} (energy evolves, bottleneck lifetime, 10% = 1 min)")
    print(f"Drones {args.min_nodes} to {args.max_nodes} (step {args.step})")
    print(f"Area {args.width}x{args.length} m, range {args.range} m")
    print(f"Packets per source: {args.packets}, CBR interval: {args.cbr_interval} ms\n")

    for n in node_counts:
        print(f"Processing n={n} drones...")
        mean_life, std_life = run_for_node_count(
            n, args.width, args.length, args.range, args.mode,
            args.packets, args.cbr_interval, args.seed
        )
        avg_lifetimes.append(mean_life)
        std_lifetimes.append(std_life)
        print(f"  n={n:3d}: avg bottleneck lifetime {mean_life:.2f} min (±{std_life:.2f})")

    data = np.column_stack((node_counts, avg_lifetimes, std_lifetimes))
    csv_filename = f'experiment_full_{args.mode}.csv'
    np.savetxt(csv_filename, data, header='nodes,avg_lifetime_min,std_lifetime_min', delimiter=',', comments='')
    print(f"\nSaved {csv_filename}")