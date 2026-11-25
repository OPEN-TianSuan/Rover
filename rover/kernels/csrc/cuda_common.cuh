#pragma once

#include <cuda_bf16.h>
#include <torch/extension.h>

#define WARP_SIZE 32
#define TILE_DIM  128

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__device__ __forceinline__ int unpack_code(const int32_t* packed, int dim_idx, int bits) {
    const int pack_factor = 32 / bits;
    const int packed_idx = dim_idx / pack_factor;
    const int shift = (dim_idx % pack_factor) * bits;
    return (packed[packed_idx] >> shift) & ((1 << bits) - 1);
}

__device__ __forceinline__ float block_reduce_sum(float val) {
    __shared__ float shared[128];
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int warp = threadIdx.x / WARP_SIZE;
    val = warp_reduce_sum(val);
    if (lane == 0) {
        shared[warp] = val;
    }
    __syncthreads();
    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0.0f;
    if (warp == 0) {
        val = warp_reduce_sum(val);
    }
    return val;
}

template<typename T>
__device__ __forceinline__ float turboquant_to_float(T value);

template<>
__device__ __forceinline__ float turboquant_to_float<float>(float value) {
    return value;
}

template<>
__device__ __forceinline__ float turboquant_to_float<__nv_bfloat16>(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

__device__ __forceinline__ int turboquant_bucketize(float value, const float* boundaries, int boundary_count) {
    int code = 0;
    while (code < boundary_count && value > boundaries[code]) {
        code++;
    }
    return code;
}
