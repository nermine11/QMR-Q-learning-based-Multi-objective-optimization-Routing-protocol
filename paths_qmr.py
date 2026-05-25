import math, random
import numpy as np
import networkx as nx
from qmr_fixed_w import QMAR          # your corrected QMAR class (unchanged except for util replacement)

# ================== 1. Mock environment ==================
class MockDrone:
    def __init__(self, identifier, coords, energy, max_drones, speed=8.0):
        self.identifier = identifier          # integer
        self.coords = coords
        self.residual_energy = energy
        self.initial_energy = 100.0
        self.speed = speed
        self.neighbor_table = np.zeros((max_drones, 13))
        self.neighbor_table[:, 9] = 0.5        # initial Q-values
        self.communication_range = 200

class MockSimulator:
    def __init__(self, drones, depot_coords):
        self.drones = drones
        self.depot_coordinates = depot_coords
        self.cur_step = 0

class MockPacket:
    def __init__(self, dest, creation_step=0):
        self.time_step_creation = creation_step

# ================== 2. Graph building ==================
def build_graph():
    G = nx.DiGraph()
    G.add_node("A", energy=30)
    G.add_node("B", energy=10)
    G.add_node("C", energy=100)
    G.add_node("D", energy=30)
    G.add_node("E", energy=100)
    G.add_edge("A", "B", delay=5)
    G.add_edge("A", "C", delay=25)
    G.add_edge("B", "D", delay=25)
    G.add_edge("C", "E", delay=25)
    G.add_edge("E", "D", delay=25)
    return G

# ================== 3. Convert graph to mocks with 2D table ==================
def graph_to_mock_2d(G, depot_coords=(500,500)):
    nodes = list(G.nodes())
    max_drones = len(nodes)
    name_to_id = {name: i for i, name in enumerate(nodes)}

    drones = []
    drone_dict = {}
    for name in nodes:
        drone_id = name_to_id[name]
        pos = (random.randint(0,500), random.randint(0,500))
        energy = G.nodes[name]['energy']
        drone = MockDrone(drone_id, pos, energy, max_drones)
        drone.residual_energy = energy
        drones.append(drone)
        drone_dict[drone_id] = drone

    # fill neighbour tables
    for u, v, data in G.edges(data=True):
        uid = name_to_id[u]
        vid = name_to_id[v]
        delay = data['delay']

        # u -> v
        table_u = drone_dict[uid].neighbor_table
        v_drone = drone_dict[vid]
        table_u[vid, 0] = v_drone.coords[0] - 10 # previous x coordinate of neighbor
        table_u[vid, 1] = v_drone.coords[1] - 10 # previous y coordinate of neighbor
        table_u[vid, 4] = v_drone.coords[0]      # current X coordinate of neighbor
        table_u[vid, 5] = v_drone.coords[1]      # current Y coordinate of neighbor
        table_u[vid, 6] = 0                      # timestamp of last positio update
        table_u[vid, 7] = 0.9                    # discount factor  
        table_u[vid, 8] = delay                   
        table_u[vid, 9] = 0.5                    # intial q  value for the neighbor
        table_u[vid,10] = 0.3                    # learning rate
        table_u[vid,11] = 0                       # queue delay
        table_u[vid,12] = 1.0                     # link quality

        # v -> u (same way)
        table_v = drone_dict[vid].neighbor_table
        u_drone = drone_dict[uid]
        table_v[uid, 0] = u_drone.coords[0] - 10
        table_v[uid, 1] = u_drone.coords[1] - 10
        table_v[uid, 4] = u_drone.coords[0]
        table_v[uid, 5] = u_drone.coords[1]
        table_v[uid, 6] = 0
        table_v[uid, 7] = 0.9
        table_v[uid, 8] = delay
        table_v[uid, 9] = 0.5
        table_v[uid,10] = 0.3
        table_v[uid,11] = 0
        table_v[uid,12] = 1.0

    sim = MockSimulator(drones, depot_coords)
    id_to_name = {v: k for k, v in name_to_id.items()}   # <-- ADD THIS
    return sim, drone_dict, name_to_id, id_to_name

# ================== 4. Forward packet ==================
def forward_packet(start_id, dest_id, drone_dict, qmar_dict, sim):
    current_id = start_id
    path = [current_id]
    prev_id = None

    for _ in range(20):          # max 20 hops
        if current_id == dest_id:
            break

        # Get QMAR instance for current drone
        if current_id not in qmar_dict:
            qmar_dict[current_id] = QMAR(drone_dict[current_id], sim)

        qmar = qmar_dict[current_id]
        current_drone = drone_dict[current_id]

        # Find neighbours (any row with delay > 0, excluding self and previous)
        neighbor_rows = np.where(current_drone.neighbor_table[:, 8] != 0)[0]
        opt_neighbors = [drone_dict[i] for i in neighbor_rows
                         if i != current_id and i != prev_id]

        if not opt_neighbors:
            break

        # Relay selection
        chosen = qmar.relay_selection(opt_neighbors, (MockPacket(dest_id),))
        if chosen == "RHP":
            break
        else:
            next_id = chosen.identifier
            path.append(next_id)
            # Feedback: best future Q from chosen node
            chosen_drone = drone_dict[next_id]
            max_q = np.max(chosen_drone.neighbor_table[:, 9]) if np.any(chosen_drone.neighbor_table[:, 9]) else 0.5
            qmar.feedback(0, next_id, max_q)

            # Move to next hop 
            prev_id = current_id
            current_id = next_id

    return path     # only return after loop finishes

# ================== 5. Main ==================
if __name__ == "__main__":
    G = build_graph()
    sim, drone_dict, name_to_id, id_to_name = graph_to_mock_2d(G)

    start_name = "A"
    dest_name = "D"
    start_id = name_to_id[start_name]
    dest_id = name_to_id[dest_name]

    qmar_dict = {}
    print("Fixed W: 0.8\n")
    for pkt in range(1,6):
        path_ids = forward_packet(start_id, dest_id, drone_dict, qmar_dict, sim)
        path_names = [id_to_name[i] for i in path_ids]
        print(f"Packet {pkt}: {' → '.join(path_names)}")
        qa = drone_dict[start_id].neighbor_table
        print(f"         Q(A,B)={qa[name_to_id['B'],9]:.3f}  Q(A,C)={qa[name_to_id['C'],9]:.3f}")