#include <cuda.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <stdio.h>

#define NUM_ANTS 8
#define NUM_NODES 8
#define SIZE_ANT_DATA 3

#define INDEX_COL 0
#define CURRENT_NODE_COL 1
#define STARTING_NODE_COL 2

__global__ void ant_solution_kernel(float *distances_matrix,
                                    float *pheromones_matrix,
                                    int *prev_visited_matrix, int *ant_matrix,
                                    float *desires_matrix,
                                    float *probability_matrix,
                                    float *path_solution_matrix) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= NUM_ANTS)
    return;

  for (int node = 0; node < NUM_NODES; node++) {
    for (int desire_node = 0; desire_node < NUM_NODES; desire_node++) {
      int current_node = ant_matrix[idx * SIZE_ANT_DATA + CURRENT_NODE_COL];
      bool desire_node_visited = false;
      for (int i = 0; i < NUM_NODES; i++) {
        if (desire_node == prev_visited_matrix[idx * NUM_NODES + i]) {
          desire_node_visited = true;
          break;
        }
      }
      if ((desire_node != current_node) && !(desire_node_visited)) {
        int _idx = current_node * NUM_NODES + desire_node;
        float desire =
            (pheromones_matrix[_idx]) * (1 / (distances_matrix[_idx]));
        desires_matrix[idx * NUM_NODES + desire_node] = desire;
      } else {
        desires_matrix[idx * NUM_NODES + desire_node] = 0;
      }
    }

    if (node != NUM_NODES - 1) {
      // Calculate sum of desires_matrix row
      float desires_row_sum = 0;
      for (int desires_node_offset = 0; desires_node_offset < NUM_NODES;
           desires_node_offset++) {
        desires_row_sum +=
            desires_matrix[idx * NUM_NODES + desires_node_offset];
      }

      if (desires_row_sum == 0) {
        printf("------------------------------DIVIDE BY "
               "ZERO------------------------------n");
      }
      for (int probability_node = 0; probability_node < NUM_NODES;
           probability_node++) {
        probability_matrix[idx * NUM_NODES + probability_node] =
            desires_matrix[idx * NUM_NODES + probability_node] /
            desires_row_sum;
      }

      // Generate random target
      curandState state;
      curand_init(clock64(), idx, 0, &state);
      float random_selection = curand_uniform(&state);

      int target_node = 0;
      float probability_sum = 0;
      for (int outcome = 0; outcome < NUM_NODES; outcome++) {
        probability_sum += probability_matrix[idx * NUM_NODES + outcome];
        if (random_selection < probability_sum) {
          target_node = outcome;
          break;
        }
      }
      bool current_node_visited = false;
      bool target_node_visited = false;

      for (int i = 0; i < NUM_NODES; i++) {
        if (ant_matrix[idx * NUM_NODES + CURRENT_NODE_COL] !=
            prev_visited_matrix[idx * NUM_NODES + i]) {
          current_node_visited = true;
        }
      }

      if (!current_node_visited) {
        for (int i = 0; i < NUM_NODES; i++) {
          if (prev_visited_matrix[idx * NUM_NODES + i] == -1) {
            prev_visited_matrix[idx * NUM_NODES + i] =
                ant_matrix[idx * NUM_NODES + CURRENT_NODE_COL];
          }
        }
      }

      for (int i = 0; i < NUM_NODES; i++) {
        if (target_node == prev_visited_matrix[idx * NUM_NODES + i]) {
          target_node_visited = true;
        }
      }
      bool target_node_set = false;

      if (!target_node_visited) {
        for (int i = 0; i < NUM_NODES; i++) {
          if ((prev_visited_matrix[idx * NUM_NODES + i] == -1) &&
              (!target_node_set)) {

            prev_visited_matrix[idx * NUM_NODES + i] = target_node;
            ant_matrix[idx * NUM_NODES + CURRENT_NODE_COL] = target_node;
            target_node_set = true;
          }
        }
      }
    }
  }
  float dist = 0;
  int current_node, next_node;
  for (int i = 0; i < NUM_NODES - 1; i++) {
    current_node = prev_visited_matrix[idx * NUM_NODES + i];
    next_node = prev_visited_matrix[idx * NUM_NODES + i + 1];
    dist += distances_matrix[current_node * NUM_NODES + next_node];
  }

  current_node = prev_visited_matrix[idx * NUM_NODES + 0];
  next_node = prev_visited_matrix[idx * NUM_NODES + NUM_NODES - 1];
  dist += distances_matrix[current_node * NUM_NODES + next_node];

  if (dist < path_solution_matrix[idx * (NUM_NODES+1) + 0]) {
    path_solution_matrix[idx * (NUM_NODES+1) + 0] = dist;
    for (int i = 1; i < NUM_NODES + 1; i++) {
      path_solution_matrix[idx * (NUM_NODES+1) + i] =
	(float)prev_visited_matrix[idx * (NUM_NODES+1) + i - 1];
    }
  }

  // Reset ant
  prev_visited_matrix[idx * NUM_NODES + 0] = idx;
  for (int i = 1; i < NUM_NODES; i++) {
    prev_visited_matrix[idx * NUM_NODES + i] = -1;
  }
  ant_matrix[idx * NUM_NODES + INDEX_COL] = idx;
  ant_matrix[idx * NUM_NODES + CURRENT_NODE_COL] = idx;
  ant_matrix[idx * NUM_NODES + STARTING_NODE_COL] = idx;
}

extern "C" {
void ant_solution(float *distances_matrix, float *pheromones_matrix,
                  int *prev_visited_matrix, int *ant_matrix,
                  float *desires_matrix, float *probability_matrix,
                  float *path_solution_matrix) {

  float *d_distances_matrix, *d_pheromones_matrix, *d_desires_matrix,
      *d_probability_matrix, *d_path_solution_matrix;
  int *d_prev_visited_matrix, *d_ant_matrix;

  cudaMalloc((void **)&d_distances_matrix,
             NUM_ANTS * NUM_NODES * sizeof(float));
  cudaMalloc((void **)&d_pheromones_matrix,
             NUM_ANTS * NUM_NODES * sizeof(float));
  cudaMalloc((void **)&d_prev_visited_matrix,
             NUM_ANTS * NUM_NODES * sizeof(int));
  cudaMalloc((void **)&d_ant_matrix, NUM_ANTS * SIZE_ANT_DATA * sizeof(int));
  cudaMalloc((void **)&d_desires_matrix, NUM_ANTS * NUM_NODES * sizeof(float));
  cudaMalloc((void **)&d_probability_matrix,
             NUM_ANTS * NUM_NODES * sizeof(float));
  cudaMalloc((void **)&d_path_solution_matrix,
             NUM_ANTS * (NUM_NODES + 1) * sizeof(float));

  cudaMemcpy(d_distances_matrix, distances_matrix,
             NUM_ANTS * NUM_NODES * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_pheromones_matrix, pheromones_matrix,
             NUM_ANTS * NUM_NODES * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_prev_visited_matrix, prev_visited_matrix,
             NUM_ANTS * NUM_NODES * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_ant_matrix, ant_matrix, NUM_ANTS * SIZE_ANT_DATA * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_desires_matrix, desires_matrix,
             NUM_ANTS * NUM_NODES * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_probability_matrix, probability_matrix,
             NUM_ANTS * NUM_NODES * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_path_solution_matrix, path_solution_matrix,
             NUM_ANTS * (NUM_NODES + 1) * sizeof(float),
             cudaMemcpyHostToDevice);

  ant_solution_kernel<<<NUM_ANTS, 1>>>(d_distances_matrix, d_pheromones_matrix,
                                d_prev_visited_matrix, d_ant_matrix,
                                d_desires_matrix, d_probability_matrix,
                                d_path_solution_matrix);

  cudaMemcpy(prev_visited_matrix, d_prev_visited_matrix,
             NUM_ANTS * NUM_NODES * sizeof(float), cudaMemcpyDeviceToHost);

  cudaMemcpy(path_solution_matrix, d_path_solution_matrix,
             NUM_ANTS * (NUM_NODES + 1) * sizeof(float),
             cudaMemcpyDeviceToHost);

  cudaFree(d_prev_visited_matrix);
  cudaFree(d_distances_matrix);
  cudaFree(d_pheromones_matrix);
  cudaFree(d_desires_matrix);
  cudaFree(d_probability_matrix);
  cudaFree(d_path_solution_matrix);
  cudaFree(d_prev_visited_matrix);
  cudaFree(d_ant_matrix);
}
}