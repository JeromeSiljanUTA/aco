import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ctypes
from ctypes import *

np.set_printoptions(precision=4)


ITERS = 3
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


def get_ant_solution():
    dll = ctypes.CDLL("./ant_solution.so", mode=ctypes.RTLD_GLOBAL)
    func = dll.ant_solution
    func.argtypes = [POINTER(c_float), POINTER(c_float), POINTER(c_int)]
    return func


__ant_solution = get_ant_solution()


def ant_solution(distances_matrix, pheromones_matrix, prev_visited_matrix):
    distances_matrix_p = distances_matrix.ctypes.data_as(POINTER(c_float))
    pheromones_matrix_p = pheromones_matrix.ctypes.data_as(POINTER(c_float))
    prev_visited_matrix_p = prev_visited_matrix.ctypes.data_as(POINTER(c_int))

    __ant_solution(distances_matrix_p, pheromones_matrix_p, prev_visited_matrix_p)


def initialize_matrices():
    df = pd.read_csv("aco_locations.csv", index_col="Point")

    n = len(df)
    distances_matrix = np.zeros((n, n))
    pheromones_matrix = np.full((n, n), PHEROMONE_INITIAL_VALUE)

    for i in range(n):
        for j in range(n):
            distances_matrix[i, j] = np.sqrt(
                (df.iloc[i]["x_coord"] - df.iloc[j]["x_coord"]) ** 2
                + (df.iloc[i]["y_coord"] - df.iloc[j]["y_coord"]) ** 2
            )
    return (df, distances_matrix, pheromones_matrix)


def reset_ants(ants):
    for ant in ants:
        ant.previously_visited = []
        ant.current_node = ant.index
        ant.starting_node = ant.index

    return ants


# Initialize
df, distances_matrix, pheromones_matrix = initialize_matrices()

prev_visited_matrix = np.full((NUM_ANTS, NUM_NODES), -1, dtype=int)
ant_matrix = np.zeros((NUM_ANTS, SIZE_ANT_DATA), dtype=int)
probability_matrix = np.zeros((NUM_ANTS, NUM_NODES))
desires_matrix = np.zeros((NUM_ANTS, NUM_NODES))

# First element is distance, the rest show the actual path
path_solution_matrix = np.full((NUM_ANTS, NUM_NODES + 1), MAX_DIST, dtype=float)


if len(df) != NUM_NODES:
    print("------------------------------check dims------------------------------\n")

for ant in range(NUM_ANTS):
    prev_visited_matrix[ant][0] = ant
    ant_matrix[ant][INDEX_COL] = ant
    ant_matrix[ant][CURRENT_NODE_COL] = ant
    ant_matrix[ant][STARTING_NODE_COL] = ant


best_solution = {"dist": MAX_DIST, "path": []}

outcomes = [node for node in range(NUM_NODES)]
for iteration in range(ITERS):
    distances_matrix_cuda = distances_matrix.flatten().astype("float32")
    pheromones_matrix_cuda = pheromones_matrix.flatten().astype("float32")
    prev_visited_matrix_cuda = prev_visited_matrix.flatten().astype("int32")
    ant_solution(
        distances_matrix_cuda, pheromones_matrix_cuda, prev_visited_matrix_cuda
    )

    break
    # Construct ant solutions
    # for ant in range(NUM_ANTS):
    #     # Calculate desires
    #     for node in range(NUM_NODES):
    #         for desire_node in range(NUM_NODES):
    #             current_node = ant_matrix[ant][CURRENT_NODE_COL]

    #             desire_node_visited = False
    #             # Has desire node not been marked visited
    #             for i in range(NUM_NODES):
    #                 if desire_node == prev_visited_matrix[ant][i]:
    #                     desire_node_visited = True
    #                     break

    #             if desire_node != current_node and not desire_node_visited:
    #                 desire = (
    #                     (pheromones_matrix[current_node][desire_node]) ** ALPHA
    #                 ) * ((1 / distances_matrix[current_node][desire_node]) ** BETA)

    #                 desires_matrix[ant][desire_node] = desire

    #                 if DEBUG:
    #                     print(f"Comparing {current_node} and {desire_node}: {desire}")

    #             else:
    #                 if DEBUG:
    #                     print(f"Not comparing {current_node} and {desire_node}")

    #                 desires_matrix[ant][desire_node] = 0

    #         if node != NUM_NODES - 1:
    #             probability_matrix[ant] = desires_matrix[ant] / np.sum(
    #                 desires_matrix[ant]
    #             )
    #             # Choosing next node
    #             target_node = np.random.choice(outcomes, p=probability_matrix[ant])
    #             current_node_visited = False
    #             target_node_visited = False

    #             # Has current node not been marked visited
    #             for i in range(NUM_NODES):
    #                 if ant_matrix[ant][CURRENT_NODE_COL] != prev_visited_matrix[ant][i]:
    #                     current_node_visited = True

    #             if not current_node_visited:
    #                 for i in range(NUM_NODES):
    #                     if prev_visited_matrix[ant][i] == -1:
    #                         prev_visited_matrix[ant][i] = ant_matrix[ant][
    #                             CURRENT_NODE_COL
    #                         ]

    #             # Has target_node been marked visited?
    #             for i in range(NUM_NODES):
    #                 if target_node == prev_visited_matrix[ant][i]:
    #                     target_node_visited = True

    #             target_node_set = False
    #             if not target_node_visited:
    #                 for i in range(NUM_NODES):
    #                     if prev_visited_matrix[ant][i] == -1 and not target_node_set:
    #                         prev_visited_matrix[ant][i] = target_node
    #                         ant_matrix[ant][CURRENT_NODE_COL] = target_node
    #                         target_node_set = True

    #     # Calculate path solution, distance
    #     dist = 0
    #     for idx in range(NUM_NODES - 1):
    #         current_node = prev_visited_matrix[ant][idx]
    #         next_node = prev_visited_matrix[ant][idx + 1]
    #         dist += distances_matrix[current_node][next_node]

    #     dist += distances_matrix[int(prev_visited_matrix[ant][0])][
    #         int(prev_visited_matrix[ant][NUM_NODES - 1])
    #     ]

    #     if dist < path_solution_matrix[ant][0]:
    #         path_solution_matrix[ant][0] = dist
    #         for i in range(1, NUM_NODES + 1):
    #             path_solution_matrix[ant][i] = prev_visited_matrix[ant][i - 1]

    # for ant in range(NUM_ANTS):
    #     prev_visited_matrix[ant][0] = ant
    #     for node in range(1, NUM_NODES):
    #         prev_visited_matrix[ant][node] = -1
    #     ant_matrix[ant][INDEX_COL] = ant
    #     ant_matrix[ant][CURRENT_NODE_COL] = ant
    #     ant_matrix[ant][STARTING_NODE_COL] = ant
