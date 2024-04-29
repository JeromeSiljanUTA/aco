#include <cuda.h>
#include <cuda_runtime_api.h>
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
    }
  }
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
  cudaMemcpy(d_probability_matrix, probability_matrix,
             NUM_ANTS * (NUM_NODES + 1) * sizeof(float),
             cudaMemcpyHostToDevice);

  ant_solution_kernel<<<NUM_ANTS, 1>>>(d_distances_matrix, d_pheromones_matrix,
                                       d_prev_visited_matrix, d_ant_matrix,
                                       d_desires_matrix, d_probability_matrix,
                                       d_path_solution_matrix);

  cudaMemcpy(desires_matrix, d_desires_matrix,
             NUM_ANTS * NUM_NODES * sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i < NUM_ANTS; i++) {
    for (int j = 0; j < NUM_NODES; j++) {
      printf("%0.4f\t", desires_matrix[i * NUM_NODES + j]);
    }
    printf("\n");
  }

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