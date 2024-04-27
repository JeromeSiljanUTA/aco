import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ITERS = 20
DEBUG = 0
PHEROMONE_INITIAL_VALUE = 0.1
EVAPORATION_CONSTANT = 0.5
Q = 1
MAX_DIST = 100000

np.set_printoptions(precision=4)

node_dict = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
    4: "E",
    5: "F",
    6: "G",
    7: "H",
}


class Ant:
    def __init__(self, index):
        self.index = index
        self.current_node = index
        self.starting_node = index
        self.previously_visited = []

    def announce(self):
        print(
            f"ant at {node_dict[self.current_node]} started at node {node_dict[self.starting_node]} has visited {self.previously_visited}"
        )

    def choose(self, target_node):
        if self.current_node not in self.previously_visited:
            self.previously_visited.append(self.current_node)

        if target_node not in self.previously_visited:
            self.previously_visited.append(target_node)
            if DEBUG:
                print(f"ant at {ant.current_node} chooses node {node_dict[choice]}")
            self.current_node = target_node

    def calc_desires(self, num_nodes, pheromones, distances):
        desires = np.zeros(num_nodes)
        for node in range(num_nodes):
            if node not in self.previously_visited and node != self.starting_node:
                print(f"Comparing {self.current_node} and {node}")
                desires[node] = pheromones[self.current_node][node] * (
                    1 / distances[self.current_node][node]
                )
            else:
                desires[node] = 0
        return desires


def initialize_matrices():
    df = pd.read_csv("aco_locations.csv", index_col="Point")

    n = len(df)
    distances = np.zeros((n, n))
    pheromones = np.full((n, n), PHEROMONE_INITIAL_VALUE)

    for i in range(n):
        for j in range(n):
            distances[i, j] = np.sqrt(
                (df.iloc[i]["x_coord"] - df.iloc[j]["x_coord"]) ** 2
                + (df.iloc[i]["y_coord"] - df.iloc[j]["y_coord"]) ** 2
            )
    return (distances, pheromones)


def reset_ants(ants):
    for ant in ants:
        ant.previously_visited = []
        ant.current_node = ant.index
        ant.starting_node = ant.index

    return ants


# Initialize
distances, pheromones = initialize_matrices()
num_ants = len(distances)
num_nodes = len(distances)
ants = []

for ant in range(num_ants):
    index = ant
    ants.append(Ant(index))
    current_ant = ants[ant]

best_solution = {"dist": MAX_DIST, path: []}
for iter in range(ITERS):
    for ant in ants:
        # Construct ant solutions
        for i in range(num_nodes - 1):
            desires = ant.calc_desires(num_nodes, pheromones, distances)
            probs = desires / np.sum(desires)
            outcomes = [node for node in range(num_nodes)]
            choice = np.random.choice(outcomes, p=probs)
            ant.choose(choice)

        dist = 0
        for idx, current_node in enumerate(ant.previously_visited[:-1]):
            next_node = ant.previously_visited[idx + 1]
            dist += distances[current_node][next_node]

        # Compare Solution
        if dist < best_solution["dist"]:
            best_solution["dist"] = dist
            best_solution["path"] = ant.previously_visited

    # Update Pheromones
    # Evaporation
    pheromones *= 1 - EVAPORATION_CONSTANT
    # Pheromone Path Updates
    for idx, current_node in enumerate(ant.previously_visited[:-1]):
        next_node = ant.previously_visited[idx + 1]
        pheromones[current_node][next_node] += Q / dist
        pheromones[next_node][current_node] += Q / dist

    ants = reset_ants(ants)

print(best_solution)
