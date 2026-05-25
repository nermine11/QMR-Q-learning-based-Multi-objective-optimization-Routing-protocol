import random
import numpy as np
import math

def euclidean_distance(p1, p2):
    """Return the straight‑line distance between two (x, y) points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

class BASE_routing:
    """Minimal base class (same as original, just no DroNet dependencies)."""
    def __init__(self, drone, simulator):
        self.drone = drone
        self.simulator = simulator

class QMAR(BASE_routing):

    def __init__(self, drone, simulator):
        BASE_routing.__init__(self, drone, simulator)
        self.maxReward = 5      # The biggest possoble reward (when the packet reaches the depot)
        self.minReward = -5     # The worst possible reward (when the packet is lost or reaches a dead end).
        self.max_delay = 1.0    # Start with 1 to avoid dividing by zero

    def feedback(self, outcome, id_j, Q_value_best_action):
        """
        Feedback returned when the packet arrives at the depot or not
        outcome: what happened to the packet
        id_j: id of the neighbor that received or failed to receive the packet
        Q_value_best_action: the best Q_value that neighbor j itslef has for reaching 
        the destination

        """
        alpha = self.drone.neighbor_table[id_j, 10]
        gamma = self.drone.neighbor_table[id_j, 7]
        Q_value_i_j = self.drone.neighbor_table[id_j, 9]

        #gives the max reward when the packet arrives to the depot
        if (outcome == 1):
            self.drone.neighbor_table[id_j, 9] = Q_value_i_j + alpha * (self.maxReward)

        #the packet is arrived to the node j, but it isn't the depot
        elif (outcome == 0):
            delay = self.drone.neighbor_table[id_j, 8] + self.drone.neighbor_table[id_j, 11]
            reward = self.computeReward(outcome, delay, id_j)
            #Update Q table
            self.drone.neighbor_table[id_j, 9] = Q_value_i_j + alpha * (reward + gamma * Q_value_best_action - Q_value_i_j)
        # failure to reach j
        else:
            self.drone.neighbor_table[id_j, 9] = Q_value_i_j + alpha * (self.minReward + gamma * Q_value_best_action - Q_value_i_j)


    def computeReward(self, outcome, delay, neighbor_id):
        """
        Calculate the immediate reward
        """
       # 1 Keep track of the largest delay we've seen (like C's orignal code nb_maxdelay)
        if delay > self.max_delay:
            self.max_delay = delay

        # 2 Find the neighbor drone and get its energy (normalised between 0 and 1)
        neighbor = next(d for d in self.simulator.drones if d.identifier == neighbor_id)
        e_neighbor = neighbor.residual_energy / neighbor.initial_energy

        # 3 Normalise the delay and apply negative exponential
        norm_delay = delay / self.max_delay
        exp_delay = math.exp(-norm_delay)   

        # 4. DYNAMIC WEIGHT – based on THIS drone's own battery
        my_energy_ratio = self.drone.residual_energy / self.drone.initial_energy
        if my_energy_ratio < 0.5:          # low battery → prioritise speed
            dynamic_w = 0.8
        else:                              # high battery → prioritise neighbour energy
            dynamic_w = 0.3

        # 5 Combine into the reward
        reward = dynamic_w * exp_delay + (1 - dynamic_w) * e_neighbor
        return reward

    def relay_selection(self, opt_neighbors, data):
        """
        Decides which neighbor to send the packet to
        """

        packet = data[0]
        candidates = []
        candidates2 = []
        for node_j in self.simulator.drones:
            if (node_j in opt_neighbors):
                j = node_j.identifier
                #first of all we need to compute the requested velocity not to expire the packet
                deadline = 2001 - (self.simulator.cur_step - packet.time_step_creation)
                if (deadline == 0):
                    print()
                distance_i = euclidean_distance(self.drone.coords, self.simulator.depot_coordinates)      #distance from node i to depot
                req_v = distance_i / deadline

                #we compute the actual velocity from node i to node j
                actual_v, distance_i_j = self.computeActualVel(j, node_j, distance_i)
                if (actual_v >= req_v):
                    #node_j is a possible candidate!!! Now we need to compute the weight k
                    LQ = self.drone.neighbor_table[j, 12]
                    #Computing relationship coefficient
                    R = self.drone.communication_range
                    if (distance_i_j > R):
                        M = 0
                    else:
                        M = 1 - (distance_i_j/R)
                    #k = M * LQ
                    k = 1 # dont use M because the drones are not moving
                    candidates.append((node_j, k))

                elif(actual_v > 0):
                    '''
                    we append in the secondary array of candidates the neighbors
                    whose actual velocities are greater than 0, so the neighbor associated
                    with the maximum actual velocity will be selected as the next hop
                    '''
                    candidates2.append((node_j, actual_v))

        if len(candidates) == 0:
            maxx = -3414212
            if (len(candidates2) > 0):
                for i in range(len(candidates2)):
                    if (candidates2[i][1] > maxx):
                        maxx = candidates2[i][1]
                        chosen = candidates2[i][0]
            else:
                #we've encountered the routing hole problem so we give to the previous hop node 𝑖 the minimum reward
                return "RHP"
        else:
            # find all candidates with the maximum κ-weighted Q-value
            best_candidates = []
            maxx = -100000
            for i in range(len(candidates)):
                candidate = candidates[i][0]
                k = candidates[i][1]
                Q_val = self.drone.neighbor_table[candidate.identifier, 9]
                weighted = Q_val * k
                if weighted > maxx:
                    maxx = weighted
                    best_candidates = [candidate]
                elif weighted == maxx:
                    best_candidates.append(candidate)
            chosen = random.choice(best_candidates)

        #Select the id of the chosen drone
        if (chosen == None):
            id = self.drone.identifier
        else:
            id = chosen.identifier
        if random.random() < 0.1:   # 10% exploration like epl in C
            # pick a random neighbor from the list of possible neighbors
            if opt_neighbors:
                chosen = random.choice(opt_neighbors)
        return chosen

    def computeActualVel(self, j, node_j, distance_i):
        """
        how much closer the packet will get to the depot if it’s sent to neighbour j
        """
        
        #we try to estimate the position of node j at time t3, so when the packet should arrive
        x2 = self.drone.neighbor_table[j, 4]
        y2 = self.drone.neighbor_table[j, 5]
        x1 = self.drone.neighbor_table[j, 0]
        y1 = self.drone.neighbor_table[j, 1]


        if (x2 - x1 != 0):
            angle_j = math.atan((y2 - y1) / (x2 - x1))
        else:
            #it's possible that the hello packet from node_j isn't arrived yet
            angle_j = math.atan(0)

        delay = self.drone.neighbor_table[j, 8] + self.drone.neighbor_table[j, 11]
        if (delay == 0):
            delay = 0.01
        t1 = self.drone.neighbor_table[j, 6]    #timestamp of the last update of the node j in the neighbor table of node i (=self.drone)
        t3 = self.simulator.cur_step + delay
        x = x1 + node_j.speed * math.cos(angle_j) * (t3 - t1)
        y = node_j.coords[1] + node_j.speed * math.sin(angle_j) * (t3 - t1)
        distance_j = euclidean_distance((x, y), self.simulator.depot_coordinates)
        distance_i_j = euclidean_distance(self.drone.coords, (x, y))
        return (distance_i - distance_j) / delay, distance_i_j
