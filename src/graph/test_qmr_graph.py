import math, random
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
from src.routing_algorithms.q_learning_routing2 import QMAR          y


# ================== 1. Mock environment ==================

class MockDrone:
    """Minimal drone that stores just what QMAR needs."""
    def __init__(self, identifier, coords, energy, speed=0.0):
        self.identifier = identifier
        self.coords = coords
        self.residual_energy = energy
        self.initial_energy = energy   # assume started with this energy
        self.speed = speed
        # neighbor_table is a dict of dicts, columns 0..12
        self.neighbor_table = defaultdict(lambda: defaultdict(float))
        self.communication_range = 200

    def __repr__(self):
        return f"Drone({self.identifier})"


class MockSimulator:
    """Minimal simulator – holds drones and depot coordinates."""
    def __init__(self, drones, depot_coords):
        self.drones = drones
        self.depot_coordinates = depot_coords
        self.cur_step = 0


class MockPacket:
    """Fake data packet with only the destination id."""
    def __init__(self, dest):
        self.time_step_creation = 0   # assume just created



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
                10: 0.7,                       # alpha (learning rate)
                11: 0,                         # queuing delay
                12: 1.0                        # link quality (perfect)
            }

    sim = MockSimulator(drones, depot_coords)
    return sim, drone_dict


