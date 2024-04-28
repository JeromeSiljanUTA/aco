#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>

#define NUM_ANTS 8
#define NUM_NODES 8

__global__ void ant_solution_kernel(int *prev_visited_matrix) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // for (int i = 0; i < NUM_NODES; i++) {
    //   printf("Index: %0.2d\tvalue: %0.2d\n", idx * NUM_NODES + i, prev_visited_matrix[idx * NUM_NODES + i]);
    // }
}

extern "C" {
void ant_solution(int *prev_visited_matrix) {

  for(int i = 0; i < NUM_ANTS * NUM_NODES; i++){
    printf("i %0.2d: %0.2d\n", i, prev_visited_matrix[i]);
  }

  
  int *d_prev_visited_matrix;

  // Assuming one ant per node
  cudaMalloc((void **)&d_prev_visited_matrix, NUM_ANTS * NUM_NODES  * sizeof(int));
  cudaMemcpy(d_prev_visited_matrix, prev_visited_matrix, NUM_ANTS * NUM_NODES * sizeof(int),
             cudaMemcpyHostToDevice);

  ant_solution_kernel<<<NUM_ANTS, 1>>>(d_prev_visited_matrix);


  cudaFree(d_prev_visited_matrix);
}
}