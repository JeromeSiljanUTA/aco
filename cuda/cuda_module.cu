#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>

#define NUM_ANTS 8

__global__ void dist_sum_kernel(float *prev_visited, float *distances,
                                float *dist_sums) {
  int num_nodes = NUM_ANTS;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_nodes) {
    return;
  }
  for (int i = 0; i < num_nodes - 1; i++) {
    int current_node = (int)prev_visited[i];
    int next_node = (int)prev_visited[i + 1];
    dist_sums[idx] += distances[(NUM_ANTS * next_node) + current_node];
  }
  dist_sums[idx] +=
      distances[NUM_ANTS * (int)prev_visited[0] + (int)prev_visited[num_nodes]];
}

extern "C" {
void dist_sum(float *prev_visited, float *distances, float *dist_sums) {

  size_t size = (size_t)(NUM_ANTS * NUM_ANTS);
  float *d_prev_visited, *d_distances, *d_dist_sums;

  // Assuming one ant per node
  cudaMalloc((void **)&d_prev_visited, size * sizeof(float));
  cudaMalloc((void **)&d_distances, size * sizeof(float));
  cudaMalloc((void **)&d_dist_sums, NUM_ANTS * sizeof(float));

  cudaMemcpy(d_prev_visited, prev_visited, size * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_distances, distances, size * sizeof(float),
             cudaMemcpyHostToDevice);

  dist_sum_kernel<<<NUM_ANTS, 1>>>(d_prev_visited, d_distances, d_dist_sums);

  cudaMemcpy(dist_sums, d_dist_sums, NUM_ANTS * sizeof(float),
             cudaMemcpyDeviceToHost);

  cudaFree(d_prev_visited);
  cudaFree(d_distances);
  cudaFree(d_dist_sums);
}
}