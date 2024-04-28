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
                                    float *probability_matrix) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= NUM_ANTS)
    return;

  for (int i = 0; i < NUM_NODES; i++) {
    printf("Index: %0.2d\tvalue: %0.2f\n", idx * NUM_NODES + i,
           probability_matrix[idx * NUM_NODES + i]);
  }
}

extern "C" {
void ant_solution(float *distances_matrix, float *pheromones_matrix,
                  int *prev_visited_matrix, int *ant_matrix,
                  float *desires_matrix, float *probability_matrix) {

  float *d_distances_matrix, *d_pheromones_matrix, *d_desires_matrix,
      *d_probability_matrix;
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

  ant_solution_kernel<<<NUM_ANTS, 1>>>(d_distances_matrix, d_pheromones_matrix,
                                       d_prev_visited_matrix, d_ant_matrix,
                                       d_desires_matrix, d_probability_matrix);

  cudaFree(d_prev_visited_matrix);
}
}