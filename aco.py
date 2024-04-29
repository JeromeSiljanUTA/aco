import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ctypes
from ctypes import *

np.set_printoptions(precision=4)


ITERS = 30
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
    func.argtypes = [
        POINTER(c_float),
        POINTER(c_float),
        POINTER(c_int),
        POINTER(c_int),
        POINTER(c_float),
        POINTER(c_float),
        POINTER(c_float),
    ]
    return func


__ant_solution = get_ant_solution()


def ant_solution(
    distances_matrix,
    pheromones_matrix,
    prev_visited_matrix,
    ant_matrix,
    desires_matrix,
    probability_matrix,
    path_solution_matrix,
):
    distances_matrix_p = distances_matrix.ctypes.data_as(POINTER(c_float))
    pheromones_matrix_p = pheromones_matrix.ctypes.data_as(POINTER(c_float))
    prev_visited_matrix_p = prev_visited_matrix.ctypes.data_as(POINTER(c_int))
    ant_matrix_p = ant_matrix.ctypes.data_as(POINTER(c_int))
    desires_matrix_p = desires_matrix.ctypes.data_as(POINTER(c_float))
    probability_matrix_p = desires_matrix.ctypes.data_as(POINTER(c_float))
    path_solution_matrix_p = path_solution_matrix.ctypes.data_as(POINTER(c_float))

    __ant_solution(
        distances_matrix_p,
        pheromones_matrix_p,
        prev_visited_matrix_p,
        ant_matrix_p,
        desires_matrix_p,
        probability_matrix_p,
        path_solution_matrix_p,
    )


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
probability_matrix = np.zeros((NUM_ANTS, NUM_NODES), dtype=float)
desires_matrix = np.zeros((NUM_ANTS, NUM_NODES), dtype=float)

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

# outcomes = [node for node in range(NUM_NODES)]
distances_matrix_cuda = distances_matrix.flatten().astype("float32")
pheromones_matrix_cuda = pheromones_matrix.flatten().astype("float32")
prev_visited_matrix_cuda = prev_visited_matrix.flatten().astype("int32")
ant_matrix_cuda = ant_matrix.flatten().astype("int32")
desires_matrix_cuda = desires_matrix.flatten().astype("float32")
probability_matrix_cuda = probability_matrix.flatten().astype("float32")
path_solution_matrix_cuda = path_solution_matrix.flatten().astype("float32")


for iteration in range(ITERS):
    ant_solution(
        distances_matrix_cuda,
        pheromones_matrix_cuda,
        prev_visited_matrix_cuda,
        ant_matrix_cuda,
        desires_matrix_cuda,
        probability_matrix_cuda,
        path_solution_matrix_cuda,
    )

    print(path_solution_matrix_cuda.reshape(NUM_ANTS, NUM_NODES + 1).astype(int))
