#include <cstdint>
#include <iostream>
#include <numeric>

#include "utils.cu"

// row-based -> stride = row's size
__global__ void MatMul(const int32_t* a, const int32_t* b, int32_t* c, size_t m,
                       size_t n, size_t k) {
  const size_t linearIdx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t i = linearIdx / n;
  const size_t j = linearIdx % n;

  if (i >= m || j >= n) {
    return;
  }

  int32_t sum = 0;
  for (size_t k_ = 0; k_ < k; ++k_) {
    sum += a[i * k + k_] * b[k_ * n + j];
  }

  c[linearIdx] = sum;
}

template <typename T>
void PrintMatrix(T* matrix, size_t m, size_t n) {
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      std::cout << matrix[i * n + j] << ' ';
    }
    std::cout << "\n";
  }
}

int main() {
  size_t m;
  size_t k;
  size_t n;
  std::cout << "m: ";
  std::cin >> m;
  std::cout << "k: ";
  std::cin >> k;
  std::cout << "n: ";
  std::cin >> n;

  size_t aElements = m * k;
  size_t bElements = k * n;
  size_t cElements = m * n;

  constexpr size_t kIntSize = sizeof(int32_t);

  int32_t* aHost = new int32_t[aElements];
  int32_t* bHost = new int32_t[bElements];
  int32_t* cHost = new int32_t[cElements];
  std::iota(aHost, aHost + aElements, 0);  // fill 0...N
  std::iota(bHost, bHost + bElements, 0);  // fill 0...N

  int32_t* aDevice;
  int32_t* bDevice;
  int32_t* cDevice;
  CudaCheck(cudaMalloc(&aDevice, aElements * kIntSize));
  CudaCheck(cudaMalloc(&bDevice, bElements * kIntSize));
  CudaCheck(cudaMalloc(&cDevice, cElements * kIntSize));

  CudaCheck(
      cudaMemcpy(aDevice, aHost, aElements * kIntSize, cudaMemcpyHostToDevice));
  CudaCheck(
      cudaMemcpy(bDevice, bHost, bElements * kIntSize, cudaMemcpyHostToDevice));

  size_t nThreads = std::min(cElements, kThreads);
  size_t nBlocks = (cElements + nThreads - 1) / nThreads;

  MatMul<<<nBlocks, nThreads>>>(aDevice, bDevice, cDevice, m, n, k);

  CudaCheck(
      cudaMemcpy(cHost, cDevice, cElements * kIntSize, cudaMemcpyDeviceToHost));

  std::cout << "A\n";
  PrintMatrix(aHost, m, k);
  std::cout << "\nB\n";
  PrintMatrix(bHost, k, n);
  std::cout << "\nC\n";
  PrintMatrix(cHost, m, n);

  delete[] aHost;
  delete[] bHost;
  delete[] cHost;
  CudaCheck(cudaFree(aDevice));
  CudaCheck(cudaFree(bDevice));
  CudaCheck(cudaFree(cDevice));

  return 0;
}
