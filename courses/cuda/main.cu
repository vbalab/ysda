#include <chrono>
#include <cstdio>
#include <iostream>
#include <numeric>
#include <stdexcept>

void CudaCheck(cudaError_t cuda_error) {
  if (cuda_error == cudaSuccess) {
    return;
  }

  throw std::runtime_error(cudaGetErrorString(cuda_error));
}

__global__ void Kernel(int* array) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  array[idx] *= array[idx];

  printf("array[%d] = %d\n", idx, array[idx]);
}

int main() {
  constexpr size_t kElemenents = 1 << 12;
  constexpr size_t kBufferSize = kElemenents * sizeof(int);

  int* aHost = reinterpret_cast<int*>(malloc(kBufferSize));
  int* bHost = reinterpret_cast<int*>(malloc(kBufferSize));

  int* aDevice;
  CudaCheck(cudaMalloc(&aDevice, kBufferSize));

  std::iota(aHost, aHost + kElemenents, 0);  // fill 0...N

  cudaMemcpy(aDevice, aHost, kBufferSize, cudaMemcpyHostToDevice);
  cudaMemcpy(bHost, aDevice, kBufferSize, cudaMemcpyDeviceToHost);

  constexpr size_t kThreads = 1024;
  constexpr dim3 kBlockDim{kThreads};
  constexpr dim3 kGridDim{(kElemenents + kThreads - 1) / kThreads};

  const auto start = std::chrono::high_resolution_clock::now();

  Kernel<<<kGridDim, kBlockDim>>>(aDevice);
  CudaCheck(cudaGetLastError());

  CudaCheck(cudaDeviceSynchronize());

  const auto end = std::chrono::high_resolution_clock::now();
  std::cout << std::chrono::nanoseconds(end - start).count() / 1000.f
            << " us\n";

  cudaFree(aDevice);
  free(bHost);
  free(aHost);

  return 0;
}
