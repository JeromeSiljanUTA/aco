import pandas as pd
import numpy as np

PHEREMONE_INITIAL_VALUE = 0.0
Q = 1


class Ant:
    def __init__(self, index, starting_point):
        self.index = index
        self.starting_point = starting_point
        self.previously_visited = [index]

    def announce(self):
        print(
            f"ant {self.index} starting at point {self.starting_point} has visited {self.previously_visited}"
        )


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
    distances_df = pd.DataFrame(distances, index=df.index, columns=df.index)
    pheremones_df = pd.DataFrame(pheremones, index=df.index, columns=df.index)

    return (df, distances_df, pheremones_df)


df, distances_df, pheremones_df = initialize()

num_ants = len(distances_df)
num_nodes = len(distances_df)

ants = []

# Construct ant solutions
for ant in range(num_ants):
    ants.append(Ant(ant, distances_df.index[ant]))
    current_ant = ants[ant]
    current_ant.announce()
    for node in range(num_nodes):
        if node not in current_ant.previously_visited:
            pheremones_df.iat[current_ant.index, node] += (
                Q / distances_df.iat[current_ant.index, node]
            )
