import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ITERS = 20
DEBUG = 0
PHEROMONE_INITIAL_VALUE = 0.1
EVAPORATION_CONSTANT = 0.5
Q = 1
MAX_DIST = 100000
SIZE_ANT_DATA = 3
NUM_ANTS = 8
NUM_NODES = NUM_ANTS
INDEX_COL = 0
CURRENT_NODE_COL = 1
STARTING_NODE_COL = 2

ALPHA = 1
BETA = 1

np.set_printoptions(precision=4)

ant_matrix = np.zeros((NUM_ANTS, SIZE_ANT_DATA), dtype=int)
prev_visited_matrix = np.full((NUM_ANTS, NUM_NODES), -1)
desires_matrix = np.zeros((NUM_ANTS, NUM_NODES))
probability_matrix = np.zeros((NUM_ANTS, NUM_NODES))


class Ant:
    def __init__(self, index):
        self.index = index
        self.current_node = index
        self.starting_node = index
        self.previously_visited = []

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
                if DEBUG:
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
    return (df, distances, pheromones)


def reset_ants(ants):
    for ant in ants:
        ant.previously_visited = []
        ant.current_node = ant.index
        ant.starting_node = ant.index

    return ants


def show_path(df, best_solution, iteration, display=True):
    path_coords = df.loc[best_solution["path"]]

    plt.plot(df["x_coord"], df["y_coord"], marker="o", linestyle="", color="b")

    # Plot the path between consecutive points
    for i in range(len(best_solution["path"]) - 1):
        plt.plot(
            [path_coords.iloc[i]["x_coord"], path_coords.iloc[i + 1]["x_coord"]],
            [path_coords.iloc[i]["y_coord"], path_coords.iloc[i + 1]["y_coord"]],
            linestyle="-",
            color="r",
        )

    # Plot the path from the last point to the first point
    plt.plot(
        [path_coords.iloc[-1]["x_coord"], path_coords.iloc[0]["x_coord"]],
        [path_coords.iloc[-1]["y_coord"], path_coords.iloc[0]["y_coord"]],
        linestyle="-",
        color="r",
    )

    if display:
        plt.show()
    else:
        plt.savefig(f"iter {iteration:02}.png")
        plt.clf()


# Initialize
df, distances, pheromones = initialize_matrices()

if len(df) != NUM_NODES:
    printf("------------------------------check dims------------------------------\n")

for ant in range(NUM_ANTS):
    prev_visited_matrix[ant][0] = ant
    ant_matrix[ant][INDEX_COL] = ant
    ant_matrix[ant][CURRENT_NODE_COL] = ant
    ant_matrix[ant][STARTING_NODE_COL] = ant


best_solution = {"dist": MAX_DIST, "path": []}

outcomes = [node for node in range(NUM_NODES)]
for iteration in range(ITERS):
    # Construct ant solutions
    for ant in range(NUM_ANTS):
        # Calculate desires
        for node in range(NUM_NODES):
            current_node = ant_matrix[ant][CURRENT_NODE_COL]
            # if haven't visited and not already there
            if (
                prev_visited_matrix[ant][node] != node
                and ant_matrix[ant][STARTING_NODE_COL] != node
            ):

                if DEBUG:
                    print(f"Comparing {current_node} and {node}")

                desires_matrix[ant][node] = (
                    (pheromones[current_node][node]) ** ALPHA
                ) * ((1 / distances[current_node][node]) ** BETA)
            else:
                if DEBUG:
                    print(f"Not comparing {current_node} and {node}")

                desires_matrix[ant][node] = 0

        # Calculate probabilities
        probability_matrix[ant] = desires_matrix[ant] / np.sum(desires_matrix[ant])

        # Choose next node
        # Randomly choose node
        choice = np.random.choice(outcomes, p=probability_matrix[ant])

    # ant.choose(choice)

    # dist = 0
    # for idx, current_node in enumerate(ant.previously_visited):
    #     if idx + 1 == len(ant.previously_visited):
    #         next_node = ant.previously_visited[0]
    #     else:
    #         next_node = ant.previously_visited[idx + 1]
    #     dist += distances[current_node][next_node]

    # # Compare Solution
    # if dist < best_solution["dist"]:
    #     best_solution["dist"] = dist
    #     best_solution["path"] = ant.previously_visited
    break
    # Update Pheromones
    # Evaporation
    pheromones *= 1 - EVAPORATION_CONSTANT
    # Pheromone Path Updates
    for idx, current_node in enumerate(ant.previously_visited):
        if idx + 1 == len(ant.previously_visited):
            next_node = ant.previously_visited[0]
        else:
            next_node = ant.previously_visited[idx + 1]
        pheromones[current_node][next_node] += Q / dist
        pheromones[next_node][current_node] += Q / dist

    ants = reset_ants(ants)

    show_path(df, best_solution, iteration)
    break


print(best_solution)
