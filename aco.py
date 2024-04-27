import pandas as pd
import numpy as np
import sys

PHEREMONE_INITIAL_VALUE = 0.0
EVAPORATION_CONSTANT = 0.5
Q = 1

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
    def __init__(self, index, starting_point):
        self.current_node = index
        self.starting_point = starting_point
        self.previously_visited = [starting_point]

    def announce(self):
        print(
            f"ant at {node_dict[self.current_node]} started at point {node_dict[self.starting_point]} has visited {self.previously_visited}"
        )

    def choose(self, target_node):
        if self.current_node not in self.previously_visited:
            self.previously_visited.append(self.current_node)
        if target_node not in self.previously_visited:
            print(f"ant at {ant.current_node} chooses node {node_dict[choice]}")
            self.current_node = target_node


def initialize():
    df = pd.read_csv("aco_locations.csv", index_col="Point")

    n = len(df)
    distances = np.zeros((n, n))
    pheremones = np.full((n, n), PHEREMONE_INITIAL_VALUE)

    for i in range(n):
        for j in range(n):
            distances[i, j] = np.sqrt(
                (df.iloc[i]["x_coord"] - df.iloc[j]["x_coord"]) ** 2
                + (df.iloc[i]["y_coord"] - df.iloc[j]["y_coord"]) ** 2
            )
    return (df, distances, pheremones)


df, distances, pheremones = initialize()

num_ants = len(df)
num_nodes = len(df)

ants = []

prev_pheremones = pheremones.copy()

print(distances)

# Construct ant solutions
for ant in range(num_ants):
    starting_node = ant
    index = ant
    ants.append(Ant(index, starting_node))
    current_ant = ants[ant]
    for node in range(num_nodes):
        if node not in current_ant.previously_visited:
            pheremones[current_ant.current_node][node] += (
                Q / distances[current_ant.current_node][node]
            )


# Update pheremones
prev_pheremones = prev_pheremones * (1 - EVAPORATION_CONSTANT)
pheremones = pheremones + prev_pheremones


# Choose path
ant = ants[0]

desires = np.zeros(num_nodes)
for node in range(num_nodes):
    if node not in ant.previously_visited:
        desires[node] = pheremones[ant.current_node][node] * (
            1 / distances[ant.current_node][node]
        )

probs = desires / np.sum(desires)
choice = np.random.choice([node for node in range(num_nodes)], p=probs)
ant.choose(choice)
ant.announce()
