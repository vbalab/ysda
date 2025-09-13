#include <stdexcept>

void CudaCheck(cudaError_t cuda_error) {
  if (cuda_error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(cuda_error));
  }
}

constexpr size_t kThreads = 1024;
