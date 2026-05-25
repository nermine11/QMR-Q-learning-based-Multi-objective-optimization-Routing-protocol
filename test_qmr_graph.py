import math, random
from collections import defaultdict
import networkx as nx
import numpy as np

from qmr_fixed_w import QMAR

# ================== 1. Mock environment ==================

class MockDrone:
    """Minimal drone that stores just what QMAR needs."""
    def __init__(self, identifier, coords, energy, max_drones, speed=0.0):
        self.identifier = identifier
        self.coords = coords
        self.residual_energy = energy
        self.initial_energy = energy   # assume started with this energy
        self.speed = speed
        # neighbor_table is a dict of dicts, columns 0..12
        self.neighbor_table = np.zeros((max_drones, 13))
        self.neighbor_table[:, 9] = 0.5        # initial Q-values
        self.communication_range = 200


class MockSimulator:
    """Minimal simulator – holds drones and depot coordinates."""
    def __init__(self, drones, depot_coords):
        self.drones = drones
        self.depot_coordinates = depot_coords
        self.cur_step = 0


class MockPacket:
    def __init__(self, dest, creation_step=0):
        self.time_step_creation = creation_step


# ================== 2. Build the graph with NetworkX ==================
def build_graph_from_nx():
    # Directed graph (you can change to nx.Graph for undirected)
    G = nx.DiGraph()

    # Add nodes with energy attribute (percentage 0-100)
    G.add_node("A", energy=30)
    G.add_node("B", energy=10)
    G.add_node("C", energy=100)
    G.add_node("D", energy=30)   # destination
    G.add_node("E", energy=100)

    # Add edges with delay attribute (ms)
    G.add_edge("A", "B", delay=5)
    G.add_edge("A", "C", delay=25)
    G.add_edge("B", "D", delay=25)
    G.add_edge("C", "E", delay=25)
    G.add_edge("E", "D", delay=25)

    return G


def draw_graph(G):
    """ Draw the graph with node energy and edge delay labels."""
    pos = {
        "A": (0, 0),
        "B": (1, -0.5),
        "C": (1, 0.5),
        "D": (3, -0.5),
        "E": (2, 0.5)
    }
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color='lightblue',
            node_size=2000, font_size=12, arrows=True, arrowsize=20)
    edge_labels = {(u, v): f"{d['delay']} ms" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
    # Add energy labels
    for node, (x, y) in pos.items():
        energy = G.nodes[node]['energy']
        plt.text(x, y - 0.15, f"{energy}%", ha='center', va='top',
                 fontsize=9, color='darkblue')
    plt.title("FANET Test Topology")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("graph.png")
    plt.show()

# ================== 3. Convert NetworkX graph to mock environment ==================
def nx_to_mock(G, depot_coords=(500,500)):
    """Create MockDrones and MockSimulator from a NetworkX graph."""
    drones = []
    drone_dict = {}

    # Create a drone for each node
    for node in G.nodes():
        energy = G.nodes[node]['energy']
        # Give a random position 
        pos = (random.randint(0,500), random.randint(0,500))
        drone = MockDrone(node, pos, energy)
        drone.initial_energy = 100.0      # maximum battery assumed
        drone.residual_energy = energy    # current residual
        drones.append(drone)
        drone_dict[node] = drone

    # Fill neighbour tables from edges
    for u, v, data in G.edges(data=True):
        delay = data['delay']
        src_drone = drone_dict[u]
        dst_drone = drone_dict[v]
        # Set up neighbour table entries for both directions (if you want undirected)
        for src, dst, delay in [(u, v, delay), (v, u, delay)]:
            source = drone_dict[u]
            destination = drone_dict[v]
            table = src_drone.neighbor_table
            table[dst] = {
                0: dst_drone.coords[0] - 10,   # previous x (fake)
                1: dst_drone.coords[1] - 10,   # previous y
                4: dst_drone.coords[0],        # current x
                5: dst_drone.coords[1],        # current y
                6: 0,                          # last position update timestamp
                7: 0.9,                        # gamma (discount)
                8: delay,                      # MAC delay
                9: 0.5,                        # initial Q-value
                10: 0.3,                       # alpha (learning rate)
                11: 0,                         # queuing delay
                12: 1.0                        # link quality (perfect)
            }

    sim = MockSimulator(drones, depot_coords)
    return sim, drone_dict




# ================== 4. Forward Packet Through the Network ==================
def forward_packet_to_destination(start_drone, dest_id, drone_dict, qmar_dict):
    """
    Forward a packet from start_drone to dest_id using QMAR at each hop.
    Returns the full path as a list of node identifiers.
    """
    current_drone = start_drone
    path = [current_drone.identifier]
    max_hops = 10  # prevent infinite loops

    for hop in range(max_hops):
        # Stop if we reached the destination
        if current_drone.identifier == dest_id:
            break
        # Get QMAR instance for the current drone (create if not exists)
        if current_drone.identifier not in qmar_dict:
            # We need a simulator reference : use the one from start
            sim = MockSimulator(list(drone_dict.values()), (500, 500))
            qmar_dict[current_drone.identifier] = QMAR(current_drone, sim)
        qmar = qmar_dict[current_drone.identifier]
        # Find all neighbors of the current drone
        opt_neighbors = []
        for nbr_id in current_drone.neighbor_table:
            if nbr_id != current_drone.identifier:
                opt_neighbors.append(drone_dict[nbr_id])
        if not opt_neighbors:
            print(f"    [No neighbors at {current_drone.identifier}]")
            break
        # Create packet
        packet = MockPacket(dest_id=dest_id)
        data = (packet,)
        # Relay selection (same as original QMAR)
        chosen = qmar.relay_selection(opt_neighbors, data)
        if chosen == "RHP":
            print(f"    [Routing Hole at {current_drone.identifier}]")
            break
        else:
            # Add to path
            path.append(chosen.identifier)
            # Feedback: update Q-value of the chosen neighbor
            chosen_drone = chosen
            max_q = 0.5
            for nbr_id, entry in chosen_drone.neighbor_table.items():
                if entry[9] > max_q:
                    max_q = entry[9]
            best_next_q = max_q
            # Normal forward (outcome=0 means reached neighbor but not destination)
            qmar.feedback(0, chosen.identifier, best_next_q)
            # Move to the next drone
            current_drone = chosen_drone
            return path
        


# ================== 5. Main ==================
if __name__ == "__main__":
    G = build_graph()
    sim, drone_dict = graph_to_mock(G)

    drone_A = drone_dict["A"]
    drone_B = drone_dict["B"]
    drone_C = drone_dict["C"]
    drone_D = drone_dict["D"]

    # Create QMAR instance for drone A
    qmar_A = QMAR(drone_A, sim)
    qmar_dict = {"A": qmar_A}

    print("=" * 65)
    print("  QMAR ROUTING – FULL PATH FROM A TO D")
    print("=" * 65)
    print(f"\n  Source:      A (energy={drone_A.residual_energy}%)")
    print(f"  Neighbors:   B (energy={drone_B.residual_energy}%, delay=5ms)")
    print(f"               C (energy={drone_C.residual_energy}%, delay=25ms)")
    print(f"  Destination: D")

    print("\n--- Forwarding 5 packets from A to D ---")
    for pkt_num in range(1, 6):
        # Reset A's Q-values each packet for a fresh test (optional)
        # drone_A.neighbor_table["B"][9] = 0.5
        # drone_A.neighbor_table["C"][9] = 0.5

        path = forward_packet_to_destination(drone_A, "D", drone_dict, qmar_dict)
        path_str = " → ".join(path)
        print(f"  Packet {pkt_num}:  {path_str}")

        # Show Q-values after this packet
        qa = drone_A.neighbor_table
        print(f"           Q(A,B)={qa['B'][9]:.3f}  Q(A,C)={qa['C'][9]:.3f}")

    # Show reward comparison
    print("\n--- Reward Comparison (computeReward) ---")
    rB = qmar_A.computeReward(0, 5, "B")
    rC = qmar_A.computeReward(0, 25, "C")
    print(f"  ω=0.8:  R(A,B) = {rB:.3f}     R(A,C) = {rC:.3f}")

    # Show the graph structure
    print("\n--- Graph Structure ---")
    for u, v, d in G.edges(data=True):
        print(f"  {u} → {v}  (delay={d['delay']}ms)")

    # Show which paths are possible
    print("\n--- All Possible Paths from A to D ---")
    for path in nx.all_simple_paths(G, source="A", target="D"):
        total_delay = sum(G[path[i]][path[i+1]]['delay'] for i in range(len(path)-1))
        path_str = " → ".join(path)
        print(f"  {path_str}  (total delay: {total_delay}ms)")