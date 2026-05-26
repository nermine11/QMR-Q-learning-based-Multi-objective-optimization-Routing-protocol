import networkx as nx
import matplotlib.pyplot as plt

# Create a directed graph (important if you want one-way links)
G = nx.DiGraph()

# --- Add Nodes with energy attribute (fraction of max battery) ---
G.add_node("A", energy=0.30)  # 30% battery
G.add_node("B", energy=0.10)  # 10% battery
G.add_node("C", energy=1.00)  # 100% battery
G.add_node("D", energy=0.30)  # 30% Destination 
G.add_node("E", energy=1.00)  # 100% battery

# --- Add Edges with delay attribute (in ms) ---
G.add_edge("A", "B", delay=5)
G.add_edge("A", "C", delay=25)
G.add_edge("B", "D", delay=25)
G.add_edge("C", "E", delay=25)
G.add_edge("E", "D", delay=25)

print("Graph created!")

# Choose a layout for the nodes
pos = {
    "A": (0, 0),
    "B": (1, -0.5),
    "C": (1, 0.5),
    "D": (3, -0.5),
    "E": (2, 0.5)
}

plt.figure(figsize=(8, 6))
# Draw nodes and labels
nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=2000)
nx.draw_networkx_labels(G, pos, font_size=12)

# Draw directed edges
nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=20, connectionstyle='arc3, rad=0.1')

# Draw edge labels showing delay
edge_labels = {(u, v): f"{d['delay']} ms" for u, v, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=10)

# Draw node labels showing energy
node_labels = {n: f"{n}\n({G.nodes[n]['energy']*100:.0f}%)" for n in G.nodes}
for node, (x, y) in pos.items():
    plt.text(x, y-0.15, node_labels[node], ha='center', va='top', fontsize=9, color='darkblue')

plt.title(" Test Topology")
plt.axis('off')
plt.tight_layout()
plt.show()