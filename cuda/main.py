import numpy as np
import ctypes
from ctypes import *


# extract dist_sum function pointer in the shared object dist_sum.so
def get_dist_sum():
    dll = ctypes.CDLL("./dist_sum.so", mode=ctypes.RTLD_GLOBAL)
    func = dll.dist_sum
    func.argtypes = [POINTER(c_float), POINTER(c_float), POINTER(c_float)]
    return func


# create __dist_sum function with get_dist_sum()
__dist_sum = get_dist_sum()


# convenient python wrapper for __dist_sum
# it does all job with types convertation
# from python ones to C++ ones
def dist_sum(prev_visited, distances, dist_sums):
    prev_visited_p = prev_visited.ctypes.data_as(POINTER(c_float))
    distances_p = distances.ctypes.data_as(POINTER(c_float))
    dist_sum_p = dist_sums.ctypes.data_as(POINTER(c_float))

    __dist_sum(prev_visited_p, distances_p, dist_sum_p)


# testing, sum of two arrays of ones and output head part of resulting array
if __name__ == "__main__":
    num_ants = 8
    size = int(num_ants * num_ants)

    prev_visited = np.array(
        [
            0,
            3,
            2,
            1,
            6,
            5,
            4,
            7,
            0,
            3,
            2,
            1,
            6,
            5,
            4,
            7,
            0,
            3,
            2,
            1,
            6,
            5,
            4,
            7,
            0,
            3,
            2,
            1,
            6,
            5,
            4,
            7,
            0,
            3,
            2,
            1,
            6,
            5,
            4,
            7,
            0,
            3,
            2,
            1,
            6,
            5,
            4,
            7,
            0,
            3,
            2,
            1,
            6,
            5,
            4,
            7,
            0,
            3,
            2,
            1,
            6,
            5,
            4,
            7,
            0,
            3,
            2,
            1,
            6,
            5,
            4,
            7,
        ]
    ).astype("float32")

    distances = np.array(
        [
            0.0,
            4.1231,
            3.6056,
            3.6056,
            5.831,
            7.2111,
            7.2801,
            8.544,
            4.1231,
            0.0,
            2.8284,
            6.3246,
            8.0623,
            5.0,
            6.3246,
            9.8995,
            3.6056,
            2.8284,
            0.0,
            4.0,
            5.3852,
            3.6056,
            4.0,
            7.0711,
            3.6056,
            6.3246,
            4.0,
            0.0,
            2.2361,
            6.7082,
            5.6569,
            5.099,
            5.831,
            8.0623,
            5.3852,
            2.2361,
            0.0,
            7.0711,
            5.3852,
            3.0,
            7.2111,
            5.0,
            3.6056,
            6.7082,
            7.0711,
            0.0,
            2.2361,
            7.2801,
            7.2801,
            6.3246,
            4.0,
            5.6569,
            5.3852,
            2.2361,
            0.0,
            5.099,
            8.544,
            9.8995,
            7.0711,
            5.099,
            3.0,
            7.2801,
            5.099,
            0.0,
        ]
    ).astype("float32")

    dist_sums = np.zeros(num_ants).astype("float32")

    dist_sum(prev_visited, distances, dist_sums)
    print(dist_sums)
