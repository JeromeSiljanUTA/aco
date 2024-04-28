#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>

#define NUM_ANTS 8
#define NUM_NODES 8

__global__ void ant_solution_kernel(float *distances_matrix,
                                    float *pheromones_matrix,
                                    int *prev_visited_matrix) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= NUM_ANTS)
    return;

  for (int i = 0; i < NUM_NODES; i++) {
    printf("Index: %0.2d\tvalue: %0.2f\n", idx * NUM_NODES + i,
           pheromones_matrix[idx * NUM_NODES + i]);
  }
}

extern "C" {
void ant_solution(float *distances_matrix, float *pheromones_matrix,
                  int *prev_visited_matrix) {

  float *d_distances_matrix, *d_pheromones_matrix;
  int *d_prev_visited_matrix;

  cudaMalloc((void **)&d_distances_matrix,
             NUM_ANTS * NUM_NODES * sizeof(float));
  cudaMalloc((void **)&d_pheromones_matrix,
             NUM_ANTS * NUM_NODES * sizeof(float));
  cudaMalloc((void **)&d_prev_visited_matrix,
             NUM_ANTS * NUM_NODES * sizeof(int));

  cudaMemcpy(d_distances_matrix, distances_matrix,
             NUM_ANTS * NUM_NODES * sizeof(float), cudaMemcpyHostToDevice);

  cudaMemcpy(d_pheromones_matrix, pheromones_matrix,
             NUM_ANTS * NUM_NODES * sizeof(float), cudaMemcpyHostToDevice);

  cudaMemcpy(d_prev_visited_matrix, prev_visited_matrix,
             NUM_ANTS * NUM_NODES * sizeof(int), cudaMemcpyHostToDevice);

  ant_solution_kernel<<<NUM_ANTS, 1>>>(d_distances_matrix, d_pheromones_matrix,
                                       d_prev_visited_matrix);

  cudaFree(d_prev_visited_matrix);
}
}