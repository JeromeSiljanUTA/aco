import pandas as pd
import numpy as np
import sys

PHEREMONE_INITIAL_VALUE = 0.0
Q = 1

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
        self.index = index
        self.starting_point = starting_point
        self.previously_visited = [index]

    def announce(self):
        print(
            f"ant {self.index} starting at point {node_dict[self.starting_point]} has visited {self.previously_visited}"
        )


def print_arr(arr):
    fmt = ["%f" for _ in range(arr.shape[1])]
    np.savetxt(sys.stdout, arr, fmt=fmt, delimiter="\t")


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

    # Convert distances to DataFrame

    # distances_df = pd.DataFrame(distances, index=df.index, columns=df.index)
    # pheremones_df = pd.DataFrame(pheremones, index=df.index, columns=df.index)

    return (df, distances, pheremones)


df, distances, pheremones = initialize()

num_ants = len(df)
num_nodes = len(df)

ants = []

# Construct ant solutions
for ant in range(num_ants):
    starting_node = ant
    ants.append(Ant(ant, starting_node))
    current_ant = ants[ant]
    current_ant.announce()
    for node in range(num_nodes):
        if node not in current_ant.previously_visited:
            pheremones[current_ant.index][node] += (
                Q / distances[current_ant.index][node]
            )

print_arr(pheremones)
