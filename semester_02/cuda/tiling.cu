// nvcc tile_mm.cu -O3
#include <cuda_runtime.h>

#include <cstdio>

// C (M x N) = A (M x K) * B (K x N), row-major
__global__ void matmul_tiled(const float* __restrict__ A,
                             const float* __restrict__ B, float* __restrict__ C,
                             int M, int N, int K) {
  constexpr size_t kTile = 32;
  __shared__ float As[kTile][kTile];
  __shared__ float Bs[kTile][kTile];

  const int row = blockIdx.y * kTile + threadIdx.y;
  const int col = blockIdx.x * kTile + threadIdx.x;

  float acc = 0.0f;

  const int tiles = (K + kTile - 1) / kTile;

  for (int t = 0; t < tiles; ++t) {
    const int aCol = t * kTile + threadIdx.x;  // A[row, aCol]
    const int bRow = t * kTile + threadIdx.y;  // B[bRow, col]

    As[threadIdx.y][threadIdx.x] =
        (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
    Bs[threadIdx.y][threadIdx.x] =
        (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;

    __syncthreads();

    for (int k = 0; k < kTile; ++k)
      acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];

    __syncthreads();  // ensure tile isn't overwritten early
  }

  if (row < M && col < N) {
    C[row * N + col] = acc;
  }
}
