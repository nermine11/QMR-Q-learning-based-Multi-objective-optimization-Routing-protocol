# Energy-Aware Q-Learning Routing for Flying Ad Hoc Networks

This repository contains an energy-aware extension of QMR (Q-learning based Multi-objective optimization Routing) for Flying Ad Hoc Networks (FANETs).

Flying Ad Hoc Networks are composed of cooperating UAVs that communicate without relying on fixed network infrastructure. 
Because UAVs can move rapidly and the network topology can change frequently, finding reliable and energy-efficient routes is a challenging problem.
Reinforcement-learning-based routing protocols such as QMR can adapt routing decisions according to network conditions.
However, the original QMR formulation uses a fixed trade-off between transmission delay and residual energy, as well as a uniform initialization of Q-values.
This project investigates whether incorporating additional energy information into these decisions can improve network lifetime.

This work was carried out during "the Introduction to Research" course in my second year at ESIEE Paris.
The project is based on: [QMR Energy-Aware Q-Learning Routing for Flying Ad Hoc Networks](https://hal.science/hal-02970649v1/document)

## QMR

QMR uses reinforcement learning to select forwarding UAVs while optimizing multiple routing objectives, particularly:
* end-to-end delay;
* residual energy of forwarding UAVs.
The original QMR reward can be represented conceptually as:
``` * R = ω · R_delay + (1 - ω) · R_energy```

where ω determines the relative importance of delay and energy. In the original approach, this weight remains fixed.

## Motivation

Two aspects of the original approach were investigated.
1. Fixed reward weighting
The original implementation uses a fixed reward weight that strongly favors delay.
This means that routing decisions use the same delay/energy trade-off regardless of the battery level of the UAV currently forwarding the packet.
2. Uniform Q-table initialization
Neighbor Q-values are initially assigned the same value.
As a result, the protocol has no initial preference for neighbors with more available energy and must discover this information progressively through reinforcement learning.

## Proposed Energy-Aware QMR
I introduced two modifications.
1. Dynamic reward weight
Instead of using the same weight throughout the simulation, the reward weight depends on the forwarding UAV's residual battery level.
When the UAV still has sufficient energy, the routing policy gives greater importance to selecting energy-rich neighbors.
When its battery becomes low, the policy prioritizes delay more strongly in order to deliver packets efficiently.
In the current implementation:

| Forwarding UAV battery | Delay weight ω |
| ---------------------- | -------------- |
| Battery ≥ 50%          | 0.3            |
| Battery < 50%          | 0.8            |

The reward therefore adapts during packet forwarding rather than using a constant optimization objective.

2- Energy-aware Q-table initialization

For the modified protocol, initial Q-values are derived from the residual energy of neighboring UAVs:

``` Q_initial(i, j) = E_j / E_max```

This gives high-energy neighbors a higher initial routing value before learning begins.

The original/fixed version initializes neighbor Q-values uniformly to 0.5.

### Experimental Setup

The modified protocol is evaluated on randomly generated static FANET topologies.

The main experiments compare:

* Fixed QMR : fixed reward weight and uniform Q-value initialization;
* Dynamic QMR : battery-dependent reward weight and energy-aware Q-value initialization.

The experiment configuration used for the study includes:

* up to 49 UAVs;
* a 500 m × 500 m simulation area;
* 200 m communication range;
* a fixed ground station located at the center;
* randomly assigned UAV battery levels;
* connected random network topologies;
* multiple source UAVs per topology;
* 1000 packets per source;
* 200 ms CBR packet interval.
 
For each network size, multiple random connected topologies are evaluated.

## Evaluation Metric

The main metric is the average bottleneck path lifetime.
For each successfully delivered packet, the residual energy of all UAVs participating in the selected route is examined.
The bottleneck corresponds to the UAV with the lowest remaining energy:
```E_bottleneck = min(E_u)```, for all UAVs u in the path
This provides an indication of how long the weakest UAV along a routing path can continue operating.
A higher average bottleneck lifetime indicates that the routing algorithm distributes traffic more effectively across the network instead of repeatedly exhausting the same UAVs.

## Results

The simulations show that the energy-aware version of QMR improves the average bottleneck path lifetime compared with the original fixed strategy.
The improvement becomes particularly visible for networks containing more than approximately five UAVs.
These results suggest that incorporating residual-energy information both:
* before learning, through Q-table initialization; and
* during learning, through adaptive reward weighting
can help reinforcement-learning routing protocols preserve the energy of critical forwarding UAVs and extend FANET network lifetime.

The experiments are intentionally performed on simplified static topologies, so the results should be interpreted as an investigation of the proposed routing modifications 
rather than as a complete real-world FANET evaluation.

## Repository Structure

```text
.
├── ESIEE_paper_code/
│   ├── new_experiment.py
│   ├── new_run.sh
│   ├── plot_comparaison.py
│   └── paper_graphs/
│
├── Simulator_original_code/
│   ├── main.py
│   ├── src/
│   ├── README.md
│   └── LICENSE.txt
│
├── requirements.txt
└── README.md
```

### `ESIEE_paper_code/`

Contains the code developed for the energy-aware QMR experiments:

- **`new_experiment.py`** — implements the experiments comparing fixed and dynamic QMR.
- **`new_run.sh`** — runs both configurations and generates the comparison results.
- **`plot_comparaison.py`** — plots the experimental results.
- **`paper_graphs/`** — contains figures and graphical resources used for the project.

### `Simulator_original_code/`

Contains the original QMR/DroNet-based simulator used as the starting point for this work:

- **`main.py`** — original simulator entry point.
- **`src/`** — simulator source code.
- **`README.md`** — documentation for the original implementation.
- **`LICENSE.txt`** — license of the original simulator.

### `requirements.txt`

Lists the Python dependencies required to run the project.

## Installation

* Clone the repository:

```
git clone https://github.com/nermine11/QMR-Q-learning-based-Multi-objective-optimization-Routing-protocol.git
cd QMR-Q-learning-based-Multi-objective-optimization-Routing-protocol
```

* Create a virtual environment:

```
python3 -m venv .venv
source .venv/bin/activate
```

* Install the dependencies:
```
pip install -r requirements.txt
```

*Running the Experiments

Run the full comparison
The provided script runs both the fixed and dynamic QMR experiments from 1 to 49 UAVs and then generates the comparison plot:
```
cd ESIEE_paper_code
chmod +x new_run.sh
./new_run.sh
```

The default experiment uses:

Nodes:                1–49
Area:                 500 × 500 m
Communication range:  200 m
Packets/source:       1000
CBR interval:         200 ms
Seed:                 42
Run one configuration

Dynamic QMR:
```
python3 new_experiment.py \
    --mode dynamic \
    --min-nodes 1 \
    --max-nodes 49 \
    --packets 1000
```
Original/fixed QMR:
```
python3 new_experiment.py \
    --mode fixed \
    --min-nodes 1 \
    --max-nodes 49 \
    --packets 1000
Visualize routing paths
```
The simulator can also visualize the different paths selected from a particular source UAV to the ground station.

For example:
```
python3 new_experiment.py \
    --visualize \
    --vis-nodes 25 \
    --vis-source 3 \
    --vis-seed 46 \
    --vis-range 200 \
    --packets 20
```
The visualization runs both the fixed and dynamic strategies on the same generated topology, making their routing behavior easier to compare.

### Main Experiment Parameters

Useful command-line arguments include:

* --mode              fixed or dynamic
* --min-nodes         minimum number of UAVs
* --max-nodes         maximum number of UAVs
* --step              increment in UAV count
* --width             simulation-area width
* --length            simulation-area length
* --range             UAV communication range
* --packets           packets sent per source
* --cbr-interval      packet interval in milliseconds
* --seed              random seed
* --visualize         enable path visualization
* --vis-nodes         UAV count for visualization
* --vis-source        source UAV
* --vis-seed          visualization topology seed
* --vis-range         visualization communication range
