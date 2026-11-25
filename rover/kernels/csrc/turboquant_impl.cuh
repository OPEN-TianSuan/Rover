#pragma once

#include <ATen/ops/bucketize.h>
#include <ATen/ops/index_select.h>
#include <ATen/ops/linalg_vector_norm.h>
#include <ATen/ops/matmul.h>
#include <ATen/ops/pad.h>
#include <ATen/ops/sum.h>
#include <ATen/ops/bitwise_left_shift.h>
#include <ATen/cuda/CUDAContext.h>

// Returns how many feature codes fit into one int32 word for a given bit width.
inline int turboquant_feat_per_int(int bits) {
    return 32 / bits;
}

// Returns the packed last-dimension size after storing dim codes into int32 words.
inline int turboquant_packed_dim(int dim, int bits) {
    const int feat = turboquant_feat_per_int(bits);
    return (dim + feat - 1) / feat;
}

// Packs per-feature integer codes into int32 words along the last dimension on CUDA.
inline torch::Tensor turboquant_pack_codes_cuda(torch::Tensor codes, int bits) {
    const int feat = turboquant_feat_per_int(bits);
    const int packed_dim = turboquant_packed_dim(codes.size(-1), bits);
    const int pad = packed_dim * feat - codes.size(-1);
    if (pad > 0) {
        codes = at::pad(codes, {0, pad});
    }
    auto shifts = torch::arange(feat, torch::TensorOptions().device(codes.device()).dtype(torch::kInt32)) * bits;
    std::vector<int64_t> grouped_shape(codes.sizes().begin(), codes.sizes().end());
    grouped_shape.back() = packed_dim;
    grouped_shape.push_back(feat);
    auto grouped = codes.reshape(grouped_shape).to(torch::kInt32);
    return at::sum(at::bitwise_left_shift(grouped, shifts), -1).to(torch::kInt32);
}

// Quantizes one [head, token] MSE vector per block and appends the packed result to cache.
// idx_map: if non-null, selects `dim` channels from a `full_dim`-wide input vector.
template<int BITS, typename scalar_t>
__global__ void append_cache_mse_kernel(
    const scalar_t* __restrict__ x,
    const float* __restrict__ rotation_t,
    const float* __restrict__ boundaries,
    int32_t* __restrict__ packed_out,
    __nv_bfloat16* __restrict__ norms_out,
    float* __restrict__ norms_out_f32,
    const int32_t* __restrict__ idx_map,   // nullable; if set, gather dim channels from full_dim input
    const int* __restrict__ start_ptr,     // device pointer to start index (or nullptr if start_val is used)
    int token_len,
    int max_len,
    int dim,
    int full_dim,                          // stride of x per token (== dim when idx_map is null)
    int packed_dim,
    int start_val)                         // used when start_ptr is nullptr
{
    constexpr int FEAT = 32 / BITS;
    const int row_idx = blockIdx.x;
    const int head_idx = row_idx / token_len;
    const int token_idx = row_idx % token_len;
    const int start = start_ptr ? *start_ptr : start_val;
    const int output_token_idx = start + token_idx;
    const int linear_row_idx = row_idx * full_dim;

    extern __shared__ unsigned char shared_buf[];
    float* shared_x = reinterpret_cast<float*>(shared_buf);
    int32_t* shared_codes = reinterpret_cast<int32_t*>(shared_x + dim);
    __shared__ float shared_inv_norm;
    __shared__ float shared_norm;

    float sq = 0.0f;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        const int src_d = idx_map ? idx_map[d] : d;
        const float value = turboquant_to_float(x[linear_row_idx + src_d]);
        shared_x[d] = value;
        sq += value * value;
    }
    sq = block_reduce_sum(sq);
    if (threadIdx.x == 0) {
        shared_norm = sqrtf(fmaxf(sq, 1e-20f));
        shared_inv_norm = 1.0f / fmaxf(shared_norm, 1e-10f);
        const int norm_offset = (head_idx * max_len + output_token_idx);
        norms_out_f32[norm_offset] = shared_norm;
        norms_out[norm_offset] = __float2bfloat16(shared_norm);
    }
    __syncthreads();

    const int boundary_count = (1 << BITS) - 1;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float proj = 0.0f;
        for (int j = 0; j < dim; j++) {
            proj += shared_x[j] * rotation_t[j * dim + d];
        }
        shared_codes[d] = turboquant_bucketize(proj * shared_inv_norm, boundaries, boundary_count);
    }
    __syncthreads();

    for (int packed_idx = threadIdx.x; packed_idx < packed_dim; packed_idx += blockDim.x) {
        uint32_t packed = 0;
#pragma unroll
        for (int feat_idx = 0; feat_idx < FEAT; feat_idx++) {
            const int d_idx = packed_idx * FEAT + feat_idx;
            const uint32_t code = d_idx < dim ? static_cast<uint32_t>(shared_codes[d_idx]) : 0u;
            packed |= code << (feat_idx * BITS);
        }
        const int32_t packed_i32 = static_cast<int32_t>(packed);
        packed_out[(head_idx * max_len + output_token_idx) * packed_dim + packed_idx] = packed_i32;
    }
}


// Internal CUDA entry point for appending MSE-quantized vectors into the cache tensors.
void append_cache_mse_cuda(
    torch::Tensor x,
    torch::Tensor rotation_t,
    torch::Tensor boundaries,
    torch::Tensor packed_out,
    torch::Tensor norms_out,
    torch::Tensor norms_out_f32,
    int start,
    int bits)
{
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(rotation_t.is_cuda(), "rotation_t must be CUDA");
    TORCH_CHECK(boundaries.is_cuda(), "boundaries must be CUDA");
    TORCH_CHECK(packed_out.is_cuda(), "packed_out must be CUDA");
    TORCH_CHECK(norms_out.is_cuda(), "norms_out must be CUDA");
    TORCH_CHECK(norms_out_f32.is_cuda(), "norms_out_f32 must be CUDA");
    TORCH_CHECK(rotation_t.dtype() == torch::kFloat32, "rotation_t must be float32");
    TORCH_CHECK(boundaries.dtype() == torch::kFloat32, "boundaries must be float32");
    TORCH_CHECK(packed_out.dtype() == torch::kInt32, "packed_out must be int32");
    TORCH_CHECK(norms_out.dtype() == torch::kBFloat16, "norms_out must be bfloat16");
    TORCH_CHECK(norms_out_f32.dtype() == torch::kFloat32, "norms_out_f32 must be float32");
    TORCH_CHECK(x.dim() == 3, "x must be [kv_heads, token_len, dim]");
    TORCH_CHECK(rotation_t.dim() == 2, "rotation_t must be [dim, dim]");
    TORCH_CHECK(boundaries.dim() == 1, "boundaries must be 1D");
    TORCH_CHECK(packed_out.dim() == 3, "packed_out must be [kv_heads, max_len, packed_dim]");
    TORCH_CHECK(norms_out.dim() == 3 && norms_out.size(2) == 1, "norms_out must be [kv_heads, max_len, 1]");
    TORCH_CHECK(norms_out_f32.dim() == 3 && norms_out_f32.size(2) == 1, "norms_out_f32 must be [kv_heads, max_len, 1]");

    const int kv_heads = x.size(0);
    const int token_len = x.size(1);
    const int dim = x.size(2);
    const int max_len = packed_out.size(1);
    const int packed_dim = packed_out.size(2);

    TORCH_CHECK(rotation_t.size(0) == dim && rotation_t.size(1) == dim, "rotation_t shape mismatch");
    TORCH_CHECK(boundaries.size(0) == (1 << bits) - 1, "boundary count mismatch");
    TORCH_CHECK(packed_out.size(0) == kv_heads, "packed_out kv_heads mismatch");
    TORCH_CHECK(norms_out.size(0) == kv_heads && norms_out_f32.size(0) == kv_heads, "norm tensor kv_heads mismatch");
    TORCH_CHECK(start >= 0 && start + token_len <= max_len, "write range exceeds cache length");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(rotation_t.is_contiguous(), "rotation_t must be contiguous");
    TORCH_CHECK(boundaries.is_contiguous(), "boundaries must be contiguous");
    TORCH_CHECK(packed_out.is_contiguous(), "packed_out must be contiguous");
    TORCH_CHECK(norms_out.is_contiguous(), "norms_out must be contiguous");
    TORCH_CHECK(norms_out_f32.is_contiguous(), "norms_out_f32 must be contiguous");

    dim3 block(128);
    dim3 grid(kv_heads * token_len);
    const size_t shared_bytes = dim * sizeof(float) + dim * sizeof(int32_t);

    if (x.dtype() == torch::kFloat32) {
        switch (bits) {
            case 1:
                append_cache_mse_kernel<1, float><<<grid, block, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
                    x.data_ptr<float>(),
                    rotation_t.data_ptr<float>(),
                    boundaries.data_ptr<float>(),
                    packed_out.data_ptr<int32_t>(),
                    reinterpret_cast<__nv_bfloat16*>(norms_out.data_ptr<at::BFloat16>()),
                    norms_out_f32.data_ptr<float>(),
                    nullptr, nullptr, token_len, max_len, dim, dim, packed_dim, start);
                return;
            case 2:
                append_cache_mse_kernel<2, float><<<grid, block, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
                    x.data_ptr<float>(),
                    rotation_t.data_ptr<float>(),
                    boundaries.data_ptr<float>(),
                    packed_out.data_ptr<int32_t>(),
                    reinterpret_cast<__nv_bfloat16*>(norms_out.data_ptr<at::BFloat16>()),
                    norms_out_f32.data_ptr<float>(),
                    nullptr, nullptr, token_len, max_len, dim, dim, packed_dim, start);
                return;
            case 3:
                append_cache_mse_kernel<3, float><<<grid, block, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
                    x.data_ptr<float>(),
                    rotation_t.data_ptr<float>(),
                    boundaries.data_ptr<float>(),
                    packed_out.data_ptr<int32_t>(),
                    reinterpret_cast<__nv_bfloat16*>(norms_out.data_ptr<at::BFloat16>()),
                    norms_out_f32.data_ptr<float>(),
                    nullptr, nullptr, token_len, max_len, dim, dim, packed_dim, start);
                return;
            case 4:
                append_cache_mse_kernel<4, float><<<grid, block, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
                    x.data_ptr<float>(),
                    rotation_t.data_ptr<float>(),
                    boundaries.data_ptr<float>(),
                    packed_out.data_ptr<int32_t>(),
                    reinterpret_cast<__nv_bfloat16*>(norms_out.data_ptr<at::BFloat16>()),
                    norms_out_f32.data_ptr<float>(),
                    nullptr, nullptr, token_len, max_len, dim, dim, packed_dim, start);
                return;
        }
    }

    TORCH_CHECK(x.dtype() == torch::kBFloat16, "x must be float32 or bfloat16");
    switch (bits) {
        case 1:
            append_cache_mse_kernel<1, __nv_bfloat16><<<grid, block, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
                reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
                rotation_t.data_ptr<float>(),
                boundaries.data_ptr<float>(),
                packed_out.data_ptr<int32_t>(),
                reinterpret_cast<__nv_bfloat16*>(norms_out.data_ptr<at::BFloat16>()),
                norms_out_f32.data_ptr<float>(),
                nullptr, nullptr, token_len, max_len, dim, dim, packed_dim, start);
            return;
        case 2:
            append_cache_mse_kernel<2, __nv_bfloat16><<<grid, block, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
                reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
                rotation_t.data_ptr<float>(),
                boundaries.data_ptr<float>(),
                packed_out.data_ptr<int32_t>(),
                reinterpret_cast<__nv_bfloat16*>(norms_out.data_ptr<at::BFloat16>()),
                norms_out_f32.data_ptr<float>(),
                nullptr, nullptr, token_len, max_len, dim, dim, packed_dim, start);
            return;
        case 3:
            append_cache_mse_kernel<3, __nv_bfloat16><<<grid, block, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
                reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
                rotation_t.data_ptr<float>(),
                boundaries.data_ptr<float>(),
                packed_out.data_ptr<int32_t>(),
                reinterpret_cast<__nv_bfloat16*>(norms_out.data_ptr<at::BFloat16>()),
                norms_out_f32.data_ptr<float>(),
                nullptr, nullptr, token_len, max_len, dim, dim, packed_dim, start);
            return;
        case 4:
            append_cache_mse_kernel<4, __nv_bfloat16><<<grid, block, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
                reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
                rotation_t.data_ptr<float>(),
                boundaries.data_ptr<float>(),
                packed_out.data_ptr<int32_t>(),
                reinterpret_cast<__nv_bfloat16*>(norms_out.data_ptr<at::BFloat16>()),
                norms_out_f32.data_ptr<float>(),
                nullptr, nullptr, token_len, max_len, dim, dim, packed_dim, start);
            return;
    }
    TORCH_CHECK(false, "unsupported bits");
}


// Computes QK logits from MSE-compressed keys for one (batch-head, query position, cache tile).
// idx_map: if non-null, selects D channels from a full_D-wide query (avoids index_select on Python side).
// accumulate: if true, adds to output instead of overwriting (for fused regular+outlier in one pass).
__global__ void qk_mse_kernel(
    const float* __restrict__ query_rot,
    const int32_t* __restrict__ packed_idx,
    const float* __restrict__ norms,
    const float* __restrict__ centroids,
    float* __restrict__ output,
    const int32_t* __restrict__ idx_map,   // nullable
    const int* __restrict__ N_ptr,         // nullable; if set, read N from device
    int M,
    int D,
    int full_D,
    int N_val,
    int out_stride,
    int cache_len,
    int packed_D,
    int bits,
    int nh,
    int nh_kv,
    bool accumulate)
{
    const int bs_idx = blockIdx.x;
    const int m_idx = blockIdx.y;
    const int n_start = blockIdx.z * blockDim.x;
    const int kv_bs_idx = bs_idx / (nh / nh_kv);
    const int n_idx = n_start + threadIdx.x;
    const int N = N_ptr ? *N_ptr : N_val;

    const float* q_ptr = query_rot + (bs_idx * M + m_idx) * full_D;
    extern __shared__ float shared_q[];
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        shared_q[d] = q_ptr[idx_map ? idx_map[d] : d];
    }
    __syncthreads();

    if (n_idx >= N) {
        return;
    }

    const int32_t* p_ptr = packed_idx + (kv_bs_idx * cache_len + n_idx) * packed_D;
    float sum = 0.0f;
    for (int d = 0; d < D; d++) {
        sum += shared_q[d] * centroids[unpack_code(p_ptr, d, bits)];
    }
    const float val = sum * norms[kv_bs_idx * cache_len + n_idx];
    const int out_idx = (bs_idx * M + m_idx) * out_stride + n_idx;
    if (accumulate) {
        output[out_idx] += val;
    } else {
        output[out_idx] = val;
    }
}


// Computes QK logits from Prod-compressed keys by combining MSE and sketch residual terms.
__global__ void qk_prod_kernel(
    const float* __restrict__ query_rot,
    const float* __restrict__ query_proj,
    const int32_t* __restrict__ packed_mse,
    const int32_t* __restrict__ packed_qjl,
    const float* __restrict__ residual_norms,
    const float* __restrict__ norms,
    const float* __restrict__ centroids,
    float coeff,
    float* __restrict__ output,
    int M,
    int D,
    int N,
    int cache_len,
    int packed_mse_D,
    int packed_qjl_D,
    int mse_bits,
    int nh,
    int nh_kv)
{
    const int bs_idx = blockIdx.x;
    const int m_idx = blockIdx.y;
    const int n_start = blockIdx.z * blockDim.x;
    const int kv_bs_idx = bs_idx / (nh / nh_kv);
    const int n_idx = n_start + threadIdx.x;

    const float* q_rot_ptr = query_rot + (bs_idx * M + m_idx) * D;
    const float* q_proj_ptr = query_proj + (bs_idx * M + m_idx) * D;
    extern __shared__ float shared_q[];
    float* shared_q_proj = shared_q + D;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        shared_q[d] = q_rot_ptr[d];
        shared_q_proj[d] = q_proj_ptr[d];
    }
    __syncthreads();

    if (n_idx >= N) {
        return;
    }

    const int32_t* mse_ptr = packed_mse + (kv_bs_idx * cache_len + n_idx) * packed_mse_D;
    const int32_t* qjl_ptr = packed_qjl + (kv_bs_idx * cache_len + n_idx) * packed_qjl_D;
    const float residual_norm = residual_norms[kv_bs_idx * cache_len + n_idx];
    const float norm = norms[kv_bs_idx * cache_len + n_idx];
    float mse_sum = 0.0f;
    float qjl_sum = 0.0f;
    for (int d = 0; d < D; d++) {
        const int mse_code = unpack_code(mse_ptr, d, mse_bits);
        const int qjl_code = unpack_code(qjl_ptr, d, 1);
        const float qjl_sign = qjl_code ? 1.0f : -1.0f;
        mse_sum += shared_q[d] * centroids[mse_code];
        qjl_sum += shared_q_proj[d] * qjl_sign;
    }
    output[(bs_idx * M + m_idx) * N + n_idx] = norm * mse_sum + coeff * residual_norm * qjl_sum;
}


// Aggregates compressed MSE values into the rotated basis before the final rotation back.
template<int BITS>
__global__ void av_mse_rotated_kernel(
    const float* __restrict__ attn,
    const int32_t* __restrict__ packed_idx,
    const float* __restrict__ norms,
    const float* __restrict__ centroids,
    float* __restrict__ output,
    const int* __restrict__ N_ptr,         // nullable; if set, read N from device
    int M,
    int D,
    int N_val,
    int attn_stride,
    int cache_len,
    int packed_D,
    int nh,
    int nh_kv)
{
    constexpr int WARPS_PER_BLOCK = 8;
    constexpr int FEAT = 32 / BITS;
    constexpr int MASK = (1 << BITS) - 1;
    const int bs_idx = blockIdx.x;
    const int m_idx = blockIdx.y;
    const int warp_idx = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int packed_word_idx = blockIdx.z * WARPS_PER_BLOCK + warp_idx;
    const int kv_bs_idx = bs_idx / (nh / nh_kv);
    const bool active = packed_word_idx < packed_D;
    const int N = N_ptr ? *N_ptr : N_val;

    const float* attn_ptr = attn + (bs_idx * M + m_idx) * attn_stride;
    const int32_t* packed_ptr = packed_idx + kv_bs_idx * cache_len * packed_D;
    const float* norm_ptr = norms + kv_bs_idx * cache_len;

    float sum[FEAT];
#pragma unroll
    for (int feat_idx = 0; feat_idx < FEAT; feat_idx++) {
        sum[feat_idx] = 0.0f;
    }
    if (active) {
        for (int n_idx = lane; n_idx < N; n_idx += WARP_SIZE) {
            const float weight = attn_ptr[n_idx] * norm_ptr[n_idx];
            const int32_t packed_word = packed_ptr[n_idx * packed_D + packed_word_idx];
#pragma unroll
            for (int feat_idx = 0; feat_idx < FEAT; feat_idx++) {
                const int code = (packed_word >> (feat_idx * BITS)) & MASK;
                sum[feat_idx] += weight * centroids[code];
            }
        }
    }
    if (active) {
#pragma unroll
        for (int feat_idx = 0; feat_idx < FEAT; feat_idx++) {
            sum[feat_idx] = warp_reduce_sum(sum[feat_idx]);
        }
        if (lane != 0) {
            return;
        }
        const int d_base = packed_word_idx * FEAT;
#pragma unroll
        for (int feat_idx = 0; feat_idx < FEAT; feat_idx++) {
            const int d_idx = d_base + feat_idx;
            if (d_idx < D) {
                output[(bs_idx * M + m_idx) * D + d_idx] = sum[feat_idx];
            }
        }
    }
}


// Internal CUDA entry point for MSE QK logits from already-rotated queries.
torch::Tensor qk_mse_cuda(
    torch::Tensor query_rot,
    torch::Tensor packed_idx,
    torch::Tensor norms,
    torch::Tensor centroids,
    int total_len,
    int bits,
    int nh,
    int nh_kv)
{
    TORCH_CHECK(query_rot.is_cuda(), "query_rot must be CUDA");
    TORCH_CHECK(packed_idx.is_cuda(), "packed_idx must be CUDA");
    TORCH_CHECK(norms.is_cuda(), "norms must be CUDA");
    TORCH_CHECK(centroids.is_cuda(), "centroids must be CUDA");
    TORCH_CHECK(query_rot.dtype() == torch::kFloat32, "query_rot must be float32");
    TORCH_CHECK(norms.dtype() == torch::kFloat32, "norms must be float32");
    TORCH_CHECK(centroids.dtype() == torch::kFloat32, "centroids must be float32");
    TORCH_CHECK(total_len <= packed_idx.size(1), "total_len exceeds packed_idx length");
    TORCH_CHECK(total_len <= norms.size(1), "total_len exceeds norms length");
    TORCH_CHECK(query_rot.size(1) == 1, "qk_mse_cuda expects q_len == 1");

    const int BS = query_rot.size(0);
    const int M = query_rot.size(1);
    const int D = query_rot.size(2);
    const int N = total_len;
    const int cache_len = packed_idx.size(1);
    const int packed_D = packed_idx.size(2);
    auto out = torch::empty({BS, M, N}, torch::TensorOptions().dtype(torch::kFloat32).device(query_rot.device()));

    dim3 block(128);
    dim3 grid(BS, M, (N + block.x - 1) / block.x);
    qk_mse_kernel<<<grid, block, D * sizeof(float), at::cuda::getCurrentCUDAStream()>>>(
        query_rot.data_ptr<float>(),
        packed_idx.data_ptr<int32_t>(),
        norms.data_ptr<float>(),
        centroids.data_ptr<float>(),
        out.data_ptr<float>(),
        nullptr, nullptr,
        M, D, D, N, N, cache_len, packed_D, bits, nh, nh_kv, false);
    return out;
}


// Fused QK: takes full query [BS, M, head_dim], splits via regular_idx/outlier_idx,
// applies rotation inside the kernel, and returns regular_scores + outlier_scores.
// Replaces: 2x index_select + 2x matmul(rotation) + 2x qk_mse_cuda (6 ops) with 2 kernel launches.
torch::Tensor qk_mse_split(
    torch::Tensor query,            // [BS, M, head_dim]  f32 or bf16
    torch::Tensor regular_idx,      // [regular_dim]  int32
    torch::Tensor outlier_idx,      // [outlier_dim]  int32
    torch::Tensor rot_reg_t,        // [regular_dim, regular_dim]  f32
    torch::Tensor rot_out_t,        // [outlier_dim, outlier_dim]  f32
    torch::Tensor packed_reg,       // key_regular packed  [kv_heads, max_len, packed_reg]
    torch::Tensor norms_reg,        // [kv_heads, max_len]  f32
    torch::Tensor centroids_reg,    // [(1<<reg_bits)]  f32
    torch::Tensor packed_out,       // key_outlier packed  [kv_heads, max_len, packed_out]
    torch::Tensor norms_out,        // [kv_heads, max_len]  f32
    torch::Tensor centroids_out,    // [(1<<out_bits)]  f32
    int total_len,
    int reg_bits,
    int out_bits,
    int nh,
    int nh_kv)
{
    TORCH_CHECK(regular_idx.dtype() == torch::kInt32 && outlier_idx.dtype() == torch::kInt32);

    const int BS       = query.size(0);
    const int M        = query.size(1);
    const int head_dim = query.size(2);
    const int reg_dim  = regular_idx.size(0);
    const int out_dim  = outlier_idx.size(0);
    const int N        = total_len;
    const int cache_len_r = packed_reg.size(1);
    const int cache_len_o = packed_out.size(1);
    const int packed_reg_D = packed_reg.size(2);
    const int packed_out_D = packed_out.size(2);

    // Apply rotation: index_select + matmul in one step via at::index_select + at::matmul
    // This is 2 ops but they're fast GEMM/gather; the kernel launch savings come from qk_mse_kernel
    auto q_f32 = query.dtype() == torch::kFloat32 ? query : query.to(torch::kFloat32);
    auto q_reg_rot = at::matmul(q_f32.index_select(-1, regular_idx.to(torch::kLong)), rot_reg_t).contiguous();
    auto q_out_rot = at::matmul(q_f32.index_select(-1, outlier_idx.to(torch::kLong)), rot_out_t).contiguous();

    auto scores = torch::zeros({BS, M, N}, torch::TensorOptions().dtype(torch::kFloat32).device(query.device()));

    dim3 block(128);
    dim3 grid(BS, M, (N + block.x - 1) / block.x);

    qk_mse_kernel<<<grid, block, reg_dim * sizeof(float), at::cuda::getCurrentCUDAStream()>>>(
        q_reg_rot.data_ptr<float>(),
        packed_reg.data_ptr<int32_t>(),
        norms_reg.data_ptr<float>(),
        centroids_reg.data_ptr<float>(),
        scores.data_ptr<float>(),
        nullptr, nullptr,
        M, reg_dim, reg_dim, N, N, cache_len_r, packed_reg_D, reg_bits, nh, nh_kv, false);

    qk_mse_kernel<<<grid, block, out_dim * sizeof(float), at::cuda::getCurrentCUDAStream()>>>(
        q_out_rot.data_ptr<float>(),
        packed_out.data_ptr<int32_t>(),
        norms_out.data_ptr<float>(),
        centroids_out.data_ptr<float>(),
        scores.data_ptr<float>(),
        nullptr, nullptr,
        M, out_dim, out_dim, N, N, cache_len_o, packed_out_D, out_bits, nh, nh_kv, true);

    return scores;
}


// Graph-safe version: writes into pre-allocated scores buffer with fixed grid.
void qk_mse_split_graph(
    torch::Tensor query,            // [BS, M, head_dim]  f32
    torch::Tensor regular_idx,      // [regular_dim]  int32
    torch::Tensor outlier_idx,      // [outlier_dim]  int32
    torch::Tensor rot_reg_t,        // [regular_dim, regular_dim]  f32
    torch::Tensor rot_out_t,        // [outlier_dim, outlier_dim]  f32
    torch::Tensor packed_reg,       // [kv_heads, max_len, packed_reg]
    torch::Tensor norms_reg,        // [kv_heads, max_len]  f32
    torch::Tensor centroids_reg,
    torch::Tensor packed_out,       // [kv_heads, max_len, packed_out]
    torch::Tensor norms_out,        // [kv_heads, max_len]  f32
    torch::Tensor centroids_out,
    torch::Tensor scores,           // [BS, M, max_len] pre-allocated output
    torch::Tensor total_len_tensor, // scalar int32 device tensor
    int max_len,
    int reg_bits,
    int out_bits,
    int nh,
    int nh_kv)
{
    const int BS       = query.size(0);
    const int M        = query.size(1);
    const int reg_dim  = regular_idx.size(0);
    const int out_dim  = outlier_idx.size(0);
    const int* N_ptr   = total_len_tensor.data_ptr<int>();
    const int cache_len_r = packed_reg.size(1);
    const int cache_len_o = packed_out.size(1);
    const int packed_reg_D = packed_reg.size(2);
    const int packed_out_D = packed_out.size(2);

    auto q_reg_rot = at::matmul(query.index_select(-1, regular_idx.to(torch::kLong)), rot_reg_t).contiguous();
    auto q_out_rot = at::matmul(query.index_select(-1, outlier_idx.to(torch::kLong)), rot_out_t).contiguous();

    scores.zero_();

    dim3 block(128);
    dim3 grid(BS, M, (max_len + block.x - 1) / block.x);

    qk_mse_kernel<<<grid, block, reg_dim * sizeof(float), at::cuda::getCurrentCUDAStream()>>>(
        q_reg_rot.data_ptr<float>(),
        packed_reg.data_ptr<int32_t>(),
        norms_reg.data_ptr<float>(),
        centroids_reg.data_ptr<float>(),
        scores.data_ptr<float>(),
        nullptr, N_ptr,
        M, reg_dim, reg_dim, 0, max_len, cache_len_r, packed_reg_D, reg_bits, nh, nh_kv, false);

    qk_mse_kernel<<<grid, block, out_dim * sizeof(float), at::cuda::getCurrentCUDAStream()>>>(
        q_out_rot.data_ptr<float>(),
        packed_out.data_ptr<int32_t>(),
        norms_out.data_ptr<float>(),
        centroids_out.data_ptr<float>(),
        scores.data_ptr<float>(),
        nullptr, N_ptr,
        M, out_dim, out_dim, 0, max_len, cache_len_o, packed_out_D, out_bits, nh, nh_kv, true);
}


void qk_mse_split_graph_rotated(
    torch::Tensor q_reg_rot,        // [BS, M, regular_dim] f32
    torch::Tensor q_out_rot,        // [BS, M, outlier_dim] f32
    torch::Tensor packed_reg,       // [kv_heads, max_len, packed_reg]
    torch::Tensor norms_reg,        // [kv_heads, max_len] f32
    torch::Tensor centroids_reg,
    torch::Tensor packed_out,       // [kv_heads, max_len, packed_out]
    torch::Tensor norms_out,        // [kv_heads, max_len] f32
    torch::Tensor centroids_out,
    torch::Tensor scores,           // [BS, M, max_len] pre-allocated output
    torch::Tensor total_len_tensor, // scalar int32 device tensor
    int max_len,
    int reg_bits,
    int out_bits,
    int nh,
    int nh_kv)
{
    const int BS = q_reg_rot.size(0);
    const int M = q_reg_rot.size(1);
    const int reg_dim = q_reg_rot.size(2);
    const int out_dim = q_out_rot.size(2);
    const int* N_ptr = total_len_tensor.data_ptr<int>();
    const int cache_len_r = packed_reg.size(1);
    const int cache_len_o = packed_out.size(1);
    const int packed_reg_D = packed_reg.size(2);
    const int packed_out_D = packed_out.size(2);

    scores.zero_();

    dim3 block(128);
    dim3 grid(BS, M, (max_len + block.x - 1) / block.x);

    qk_mse_kernel<<<grid, block, reg_dim * sizeof(float), at::cuda::getCurrentCUDAStream()>>>(
        q_reg_rot.data_ptr<float>(),
        packed_reg.data_ptr<int32_t>(),
        norms_reg.data_ptr<float>(),
        centroids_reg.data_ptr<float>(),
        scores.data_ptr<float>(),
        nullptr, N_ptr,
        M, reg_dim, reg_dim, 0, max_len, cache_len_r, packed_reg_D, reg_bits, nh, nh_kv, false);

    qk_mse_kernel<<<grid, block, out_dim * sizeof(float), at::cuda::getCurrentCUDAStream()>>>(
        q_out_rot.data_ptr<float>(),
        packed_out.data_ptr<int32_t>(),
        norms_out.data_ptr<float>(),
        centroids_out.data_ptr<float>(),
        scores.data_ptr<float>(),
        nullptr, N_ptr,
        M, out_dim, out_dim, 0, max_len, cache_len_o, packed_out_D, out_bits, nh, nh_kv, true);
}


// Internal CUDA entry point for Prod QK logits from rotated and sketched queries.
torch::Tensor qk_prod_cuda(
    torch::Tensor query_rot,
    torch::Tensor query_proj,
    torch::Tensor packed_mse,
    torch::Tensor packed_qjl,
    torch::Tensor residual_norms,
    torch::Tensor norms,
    torch::Tensor centroids,
    float coeff,
    int total_len,
    int mse_bits,
    int nh,
    int nh_kv)
{
    TORCH_CHECK(query_rot.is_cuda(), "query_rot must be CUDA");
    TORCH_CHECK(query_proj.is_cuda(), "query_proj must be CUDA");
    TORCH_CHECK(packed_mse.is_cuda(), "packed_mse must be CUDA");
    TORCH_CHECK(packed_qjl.is_cuda(), "packed_qjl must be CUDA");
    TORCH_CHECK(residual_norms.is_cuda(), "residual_norms must be CUDA");
    TORCH_CHECK(norms.is_cuda(), "norms must be CUDA");
    TORCH_CHECK(centroids.is_cuda(), "centroids must be CUDA");
    TORCH_CHECK(query_rot.dtype() == torch::kFloat32, "query_rot must be float32");
    TORCH_CHECK(query_proj.dtype() == torch::kFloat32, "query_proj must be float32");
    TORCH_CHECK(residual_norms.dtype() == torch::kFloat32, "residual_norms must be float32");
    TORCH_CHECK(norms.dtype() == torch::kFloat32, "norms must be float32");
    TORCH_CHECK(centroids.dtype() == torch::kFloat32, "centroids must be float32");
    TORCH_CHECK(total_len <= packed_mse.size(1), "total_len exceeds packed_mse length");
    TORCH_CHECK(total_len <= packed_qjl.size(1), "total_len exceeds packed_qjl length");
    TORCH_CHECK(total_len <= residual_norms.size(1), "total_len exceeds residual_norms length");
    TORCH_CHECK(total_len <= norms.size(1), "total_len exceeds norms length");
    TORCH_CHECK(query_rot.size(1) == 1, "qk_prod_cuda expects q_len == 1");
    TORCH_CHECK(query_proj.size(1) == 1, "qk_prod_cuda expects q_len == 1");

    const int BS = query_rot.size(0);
    const int M = query_rot.size(1);
    const int D = query_rot.size(2);
    const int N = total_len;
    const int cache_len = packed_mse.size(1);
    const int packed_mse_D = packed_mse.size(2);
    const int packed_qjl_D = packed_qjl.size(2);
    auto out = torch::empty({BS, M, N}, torch::TensorOptions().dtype(torch::kFloat32).device(query_rot.device()));

    dim3 block(128);
    dim3 grid(BS, M, (N + block.x - 1) / block.x);
    qk_prod_kernel<<<grid, block, 2 * D * sizeof(float), at::cuda::getCurrentCUDAStream()>>>(
        query_rot.data_ptr<float>(),
        query_proj.data_ptr<float>(),
        packed_mse.data_ptr<int32_t>(),
        packed_qjl.data_ptr<int32_t>(),
        residual_norms.data_ptr<float>(),
        norms.data_ptr<float>(),
        centroids.data_ptr<float>(),
        coeff,
        out.data_ptr<float>(),
        M,
        D,
        N,
        cache_len,
        packed_mse_D,
        packed_qjl_D,
        mse_bits,
        nh,
        nh_kv);
    return out;
}

// Helper: launch av_mse_rotated_kernel for one store.
static void launch_av_mse_rotated(
    const torch::Tensor& attn,
    const torch::Tensor& packed_idx,
    const torch::Tensor& norms,
    const torch::Tensor& centroids,
    torch::Tensor& out,
    int total_len, int attn_stride, int bits, int nh, int nh_kv,
    const int* N_ptr = nullptr)
{
    const int BS = attn.size(0);
    const int M  = attn.size(1);
    const int N  = total_len;
    const int cache_len = packed_idx.size(1);
    const int packed_D  = packed_idx.size(2);
    const int D = (32 / bits) * packed_D;

    dim3 block(8 * WARP_SIZE);
    dim3 grid(BS, M, (packed_D + 7) / 8);

    #define LAUNCH_AV(B) \
        av_mse_rotated_kernel<B><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>( \
            attn.data_ptr<float>(), packed_idx.data_ptr<int32_t>(), \
            norms.data_ptr<float>(), centroids.data_ptr<float>(), \
            out.data_ptr<float>(), N_ptr, M, D, N, attn_stride, cache_len, packed_D, nh, nh_kv)

    switch (bits) {
        case 1: LAUNCH_AV(1); return;
        case 2: LAUNCH_AV(2); return;
        case 3: LAUNCH_AV(3); return;
        case 4: LAUNCH_AV(4); return;
    }
    TORCH_CHECK(false, "unsupported bits");
    #undef LAUNCH_AV
}


// Internal CUDA entry point that returns AV aggregation in the rotated MSE basis.
torch::Tensor av_mse_rotated_cuda(
    torch::Tensor attn,
    torch::Tensor packed_idx,
    torch::Tensor norms,
    torch::Tensor centroids,
    int total_len,
    int bits,
    int nh,
    int nh_kv)
{
    TORCH_CHECK(attn.is_cuda(), "attn must be CUDA");
    TORCH_CHECK(packed_idx.is_cuda(), "packed_idx must be CUDA");
    TORCH_CHECK(norms.is_cuda(), "norms must be CUDA");
    TORCH_CHECK(centroids.is_cuda(), "centroids must be CUDA");
    TORCH_CHECK(attn.dtype() == torch::kFloat32, "attn must be float32");
    TORCH_CHECK(norms.dtype() == torch::kFloat32, "norms must be float32");
    TORCH_CHECK(centroids.dtype() == torch::kFloat32, "centroids must be float32");
    TORCH_CHECK(total_len <= packed_idx.size(1), "total_len exceeds packed_idx length");
    TORCH_CHECK(total_len <= norms.size(1), "total_len exceeds norms length");
    TORCH_CHECK(attn.size(1) == 1, "av_mse_rotated_cuda expects q_len == 1");

    const int BS = attn.size(0);
    const int M = attn.size(1);
    const int N = total_len;
    const int packed_D = packed_idx.size(2);
    const int D = (32 / bits) * packed_D;
    auto out = torch::empty({BS, M, D}, torch::TensorOptions().dtype(torch::kFloat32).device(attn.device()));
    launch_av_mse_rotated(attn, packed_idx, norms, centroids, out, total_len, N, bits, nh, nh_kv);
    return out;
}


// Fused AV: runs av_mse_rotated for regular+outlier, rotates back, and scatters into
// a single [BS, M, head_dim] output tensor.
// Replaces: 2x av_mse_rotated + 2x matmul(rotation) + 2x index_copy_ (6 ops) with 4 ops.
torch::Tensor av_mse_split(
    torch::Tensor attn,             // [BS, M, total_len]  f32
    torch::Tensor regular_idx,      // [regular_dim]  int64 (for index_copy_)
    torch::Tensor outlier_idx,      // [outlier_dim]  int64
    torch::Tensor packed_vr,        // value_regular packed  [kv_heads, max_len, packed_reg]
    torch::Tensor norms_vr,         // [kv_heads, max_len]  f32
    torch::Tensor centroids_vr,     // [(1<<reg_bits)]  f32
    torch::Tensor rotation_reg,     // [regular_dim, regular_dim]  f32
    torch::Tensor packed_vo,        // value_outlier packed
    torch::Tensor norms_vo,
    torch::Tensor centroids_vo,
    torch::Tensor rotation_out,     // [outlier_dim, outlier_dim]  f32
    int total_len,
    int reg_bits,
    int out_bits,
    int nh,
    int nh_kv,
    int head_dim)
{
    const int BS = attn.size(0);
    const int M  = attn.size(1);
    const int N  = total_len;
    auto attn_c = attn.contiguous();

    const int reg_packed_D = packed_vr.size(2);
    const int out_packed_D = packed_vo.size(2);
    const int reg_D = (32 / reg_bits) * reg_packed_D;
    const int out_D = (32 / out_bits) * out_packed_D;
    auto reg_rotated = torch::empty({BS, M, reg_D}, torch::TensorOptions().dtype(torch::kFloat32).device(attn.device()));
    auto out_rotated = torch::empty({BS, M, out_D}, torch::TensorOptions().dtype(torch::kFloat32).device(attn.device()));

    launch_av_mse_rotated(attn_c, packed_vr, norms_vr, centroids_vr, reg_rotated, total_len, N, reg_bits, nh, nh_kv);
    launch_av_mse_rotated(attn_c, packed_vo, norms_vo, centroids_vo, out_rotated, total_len, N, out_bits, nh, nh_kv);

    const int reg_dim = regular_idx.size(0);
    const int out_dim = outlier_idx.size(0);

    auto reg_out = at::matmul(reg_rotated.narrow(-1, 0, reg_dim), rotation_reg);  // [BS, M, reg_dim]
    auto out_out = at::matmul(out_rotated.narrow(-1, 0, out_dim), rotation_out);  // [BS, M, out_dim]

    auto output = torch::zeros({BS, M, head_dim}, torch::TensorOptions().dtype(torch::kFloat32).device(attn.device()));
    output.index_copy_(-1, regular_idx, reg_out);
    output.index_copy_(-1, outlier_idx, out_out);
    return output;
}


// Graph-safe version: writes into pre-allocated buffers, uses attn with max_len stride.
void av_mse_split_graph(
    torch::Tensor attn,             // [BS, M, max_len]  f32 (stride = max_len)
    torch::Tensor regular_idx,      // [regular_dim]  int64
    torch::Tensor outlier_idx,      // [outlier_dim]  int64
    torch::Tensor packed_vr,
    torch::Tensor norms_vr,
    torch::Tensor centroids_vr,
    torch::Tensor rotation_reg,
    torch::Tensor packed_vo,
    torch::Tensor norms_vo,
    torch::Tensor centroids_vo,
    torch::Tensor rotation_out,
    torch::Tensor reg_rotated_buf,  // [BS, M, reg_D] pre-allocated
    torch::Tensor out_rotated_buf,  // [BS, M, out_D] pre-allocated
    torch::Tensor output,           // [BS, M, head_dim] pre-allocated
    torch::Tensor total_len_tensor, // scalar int32 device tensor
    int max_len,
    int reg_bits,
    int out_bits,
    int nh,
    int nh_kv,
    int head_dim)
{
    auto attn_c = attn.contiguous();
    const int* N_ptr = total_len_tensor.data_ptr<int>();
    int total_len = 0; // dummy, N_ptr is used instead

    launch_av_mse_rotated(attn_c, packed_vr, norms_vr, centroids_vr, reg_rotated_buf, total_len, max_len, reg_bits, nh, nh_kv, N_ptr);
    launch_av_mse_rotated(attn_c, packed_vo, norms_vo, centroids_vo, out_rotated_buf, total_len, max_len, out_bits, nh, nh_kv, N_ptr);

    const int reg_dim = regular_idx.size(0);
    const int out_dim = outlier_idx.size(0);

    auto reg_out = at::matmul(reg_rotated_buf.narrow(-1, 0, reg_dim), rotation_reg);
    auto out_out = at::matmul(out_rotated_buf.narrow(-1, 0, out_dim), rotation_out);

    output.zero_();
    output.index_copy_(-1, regular_idx, reg_out);
    output.index_copy_(-1, outlier_idx, out_out);
}


void av_mse_split_graph_rotated(
    torch::Tensor attn,             // [BS, M, max_len] f32, contiguous
    torch::Tensor packed_vr,
    torch::Tensor norms_vr,
    torch::Tensor centroids_vr,
    torch::Tensor packed_vo,
    torch::Tensor norms_vo,
    torch::Tensor centroids_vo,
    torch::Tensor reg_rotated_buf,  // [BS, M, reg_D] pre-allocated
    torch::Tensor out_rotated_buf,  // [BS, M, out_D] pre-allocated
    torch::Tensor total_len_tensor, // scalar int32 device tensor
    int max_len,
    int reg_bits,
    int out_bits,
    int nh,
    int nh_kv)
{
    const int* N_ptr = total_len_tensor.data_ptr<int>();
    int total_len = 0; // dummy, N_ptr is used instead

    launch_av_mse_rotated(attn, packed_vr, norms_vr, centroids_vr, reg_rotated_buf, total_len, max_len, reg_bits, nh, nh_kv, N_ptr);
    launch_av_mse_rotated(attn, packed_vo, norms_vo, centroids_vo, out_rotated_buf, total_len, max_len, out_bits, nh, nh_kv, N_ptr);
}


// Python-visible API: append dense vectors to an MSE-compressed cache segment.
void append_cache_mse(
    torch::Tensor x,
    torch::Tensor rotation_t,
    torch::Tensor boundaries,
    torch::Tensor packed_out,
    torch::Tensor norms_out,
    torch::Tensor norms_out_f32,
    int start,
    int bits)
{
    append_cache_mse_cuda(
        x,
        rotation_t,
        boundaries,
        packed_out,
        norms_out,
        norms_out_f32,
        start,
        bits);
}


// Helper: launch append_cache_mse_kernel for one store with channel selection via idx_map.
template<typename scalar_t>
static void launch_append_mse_indexed(
    const scalar_t* x_ptr,
    const float* rot_t_ptr,
    const float* bounds_ptr,
    int32_t* packed_ptr,
    __nv_bfloat16* norms_ptr,
    float* norms_f32_ptr,
    const int32_t* idx_map_ptr,
    int kv_heads, int token_len, int max_len,
    int dim, int head_dim, int packed_dim,
    int start, int bits, const int* start_ptr = nullptr)
{
    dim3 block(128);
    dim3 grid(kv_heads * token_len);
    const size_t shared_bytes = dim * sizeof(float) + dim * sizeof(int32_t);
    switch (bits) {
        case 1: append_cache_mse_kernel<1, scalar_t><<<grid, block, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
            x_ptr, rot_t_ptr, bounds_ptr, packed_ptr, norms_ptr, norms_f32_ptr,
            idx_map_ptr, start_ptr, token_len, max_len, dim, head_dim, packed_dim, start); return;
        case 2: append_cache_mse_kernel<2, scalar_t><<<grid, block, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
            x_ptr, rot_t_ptr, bounds_ptr, packed_ptr, norms_ptr, norms_f32_ptr,
            idx_map_ptr, start_ptr, token_len, max_len, dim, head_dim, packed_dim, start); return;
        case 3: append_cache_mse_kernel<3, scalar_t><<<grid, block, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
            x_ptr, rot_t_ptr, bounds_ptr, packed_ptr, norms_ptr, norms_f32_ptr,
            idx_map_ptr, start_ptr, token_len, max_len, dim, head_dim, packed_dim, start); return;
        case 4: append_cache_mse_kernel<4, scalar_t><<<grid, block, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
            x_ptr, rot_t_ptr, bounds_ptr, packed_ptr, norms_ptr, norms_f32_ptr,
            idx_map_ptr, start_ptr, token_len, max_len, dim, head_dim, packed_dim, start); return;
    }
    TORCH_CHECK(false, "unsupported bits");
}


// Fused write: takes full [kv_heads, token_len, head_dim] K and V tensors, splits channels
// via regular_idx/outlier_idx inside the kernel, and writes all 4 stores in 4 launches
// (vs 8 previously: 4x index_select + 4x append_cache_mse).
void append_kv_mse_split(
    torch::Tensor key,              // [kv_heads, token_len, head_dim]  bf16
    torch::Tensor value,            // [kv_heads, token_len, head_dim]  bf16
    torch::Tensor regular_idx,      // [regular_dim]  int32
    torch::Tensor outlier_idx,      // [outlier_dim]  int32
    torch::Tensor rot_kr_t,         // [regular_dim, regular_dim]  f32
    torch::Tensor rot_ko_t,         // [outlier_dim, outlier_dim]  f32
    torch::Tensor bounds_kr,        // [(1<<reg_bits)-1]  f32
    torch::Tensor bounds_ko,        // [(1<<out_bits)-1]  f32
    torch::Tensor rot_vr_t,         // [regular_dim, regular_dim]  f32
    torch::Tensor rot_vo_t,         // [outlier_dim, outlier_dim]  f32
    torch::Tensor bounds_vr,        // [(1<<reg_bits)-1]  f32
    torch::Tensor bounds_vo,        // [(1<<out_bits)-1]  f32
    torch::Tensor packed_kr,        // key_regular   [kv_heads, max_len, packed_reg]  int32
    torch::Tensor norms_kr,
    torch::Tensor norms_kr_f32,
    torch::Tensor packed_ko,        // key_outlier
    torch::Tensor norms_ko,
    torch::Tensor norms_ko_f32,
    torch::Tensor packed_vr,        // value_regular
    torch::Tensor norms_vr,
    torch::Tensor norms_vr_f32,
    torch::Tensor packed_vo,        // value_outlier
    torch::Tensor norms_vo,
    torch::Tensor norms_vo_f32,
    int start,
    int reg_bits,
    int out_bits)
{
    TORCH_CHECK(key.is_contiguous() && value.is_contiguous(), "key/value must be contiguous");
    TORCH_CHECK(key.dtype() == torch::kBFloat16, "key must be bfloat16");
    TORCH_CHECK(value.dtype() == torch::kBFloat16, "value must be bfloat16");

    const int kv_heads  = key.size(0);
    const int token_len = key.size(1);
    const int head_dim  = key.size(2);
    const int reg_dim   = regular_idx.size(0);
    const int out_dim   = outlier_idx.size(0);
    const int max_len   = packed_kr.size(1);
    const int packed_reg = packed_kr.size(2);
    const int packed_out = packed_ko.size(2);

    using BF16 = __nv_bfloat16;
    const BF16* k_ptr = reinterpret_cast<const BF16*>(key.data_ptr<at::BFloat16>());
    const BF16* v_ptr = reinterpret_cast<const BF16*>(value.data_ptr<at::BFloat16>());
    const int32_t* reg_idx = regular_idx.data_ptr<int32_t>();
    const int32_t* out_idx = outlier_idx.data_ptr<int32_t>();

    launch_append_mse_indexed<BF16>(k_ptr,
        rot_kr_t.data_ptr<float>(), bounds_kr.data_ptr<float>(),
        packed_kr.data_ptr<int32_t>(),
        reinterpret_cast<BF16*>(norms_kr.data_ptr<at::BFloat16>()), norms_kr_f32.data_ptr<float>(),
        reg_idx, kv_heads, token_len, max_len, reg_dim, head_dim, packed_reg, start, reg_bits);

    launch_append_mse_indexed<BF16>(k_ptr,
        rot_ko_t.data_ptr<float>(), bounds_ko.data_ptr<float>(),
        packed_ko.data_ptr<int32_t>(),
        reinterpret_cast<BF16*>(norms_ko.data_ptr<at::BFloat16>()), norms_ko_f32.data_ptr<float>(),
        out_idx, kv_heads, token_len, max_len, out_dim, head_dim, packed_out, start, out_bits);

    launch_append_mse_indexed<BF16>(v_ptr,
        rot_vr_t.data_ptr<float>(), bounds_vr.data_ptr<float>(),
        packed_vr.data_ptr<int32_t>(),
        reinterpret_cast<BF16*>(norms_vr.data_ptr<at::BFloat16>()), norms_vr_f32.data_ptr<float>(),
        reg_idx, kv_heads, token_len, max_len, reg_dim, head_dim, packed_reg, start, reg_bits);

    launch_append_mse_indexed<BF16>(v_ptr,
        rot_vo_t.data_ptr<float>(), bounds_vo.data_ptr<float>(),
        packed_vo.data_ptr<int32_t>(),
        reinterpret_cast<BF16*>(norms_vo.data_ptr<at::BFloat16>()), norms_vo_f32.data_ptr<float>(),
        out_idx, kv_heads, token_len, max_len, out_dim, head_dim, packed_out, start, out_bits);
}


// Graph-safe version: reads start from a device tensor.
void append_kv_mse_split_graph(
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor regular_idx,
    torch::Tensor outlier_idx,
    torch::Tensor rot_kr_t,
    torch::Tensor rot_ko_t,
    torch::Tensor bounds_kr,
    torch::Tensor bounds_ko,
    torch::Tensor rot_vr_t,
    torch::Tensor rot_vo_t,
    torch::Tensor bounds_vr,
    torch::Tensor bounds_vo,
    torch::Tensor packed_kr,
    torch::Tensor norms_kr,
    torch::Tensor norms_kr_f32,
    torch::Tensor packed_ko,
    torch::Tensor norms_ko,
    torch::Tensor norms_ko_f32,
    torch::Tensor packed_vr,
    torch::Tensor norms_vr,
    torch::Tensor norms_vr_f32,
    torch::Tensor packed_vo,
    torch::Tensor norms_vo,
    torch::Tensor norms_vo_f32,
    torch::Tensor start_tensor,  // scalar int32 device tensor
    int reg_bits,
    int out_bits)
{
    const int kv_heads  = key.size(0);
    const int token_len = key.size(1);
    const int head_dim  = key.size(2);
    const int reg_dim   = regular_idx.size(0);
    const int out_dim   = outlier_idx.size(0);
    const int max_len   = packed_kr.size(1);
    const int packed_reg = packed_kr.size(2);
    const int packed_out_dim = packed_ko.size(2);

    using BF16 = __nv_bfloat16;
    const BF16* k_ptr = reinterpret_cast<const BF16*>(key.data_ptr<at::BFloat16>());
    const BF16* v_ptr = reinterpret_cast<const BF16*>(value.data_ptr<at::BFloat16>());
    const int32_t* reg_idx = regular_idx.data_ptr<int32_t>();
    const int32_t* out_idx = outlier_idx.data_ptr<int32_t>();
    const int* start_ptr = start_tensor.data_ptr<int>();

    launch_append_mse_indexed<BF16>(k_ptr,
        rot_kr_t.data_ptr<float>(), bounds_kr.data_ptr<float>(),
        packed_kr.data_ptr<int32_t>(),
        reinterpret_cast<BF16*>(norms_kr.data_ptr<at::BFloat16>()), norms_kr_f32.data_ptr<float>(),
        reg_idx, kv_heads, token_len, max_len, reg_dim, head_dim, packed_reg, 0, reg_bits, start_ptr);

    launch_append_mse_indexed<BF16>(k_ptr,
        rot_ko_t.data_ptr<float>(), bounds_ko.data_ptr<float>(),
        packed_ko.data_ptr<int32_t>(),
        reinterpret_cast<BF16*>(norms_ko.data_ptr<at::BFloat16>()), norms_ko_f32.data_ptr<float>(),
        out_idx, kv_heads, token_len, max_len, out_dim, head_dim, packed_out_dim, 0, out_bits, start_ptr);

    launch_append_mse_indexed<BF16>(v_ptr,
        rot_vr_t.data_ptr<float>(), bounds_vr.data_ptr<float>(),
        packed_vr.data_ptr<int32_t>(),
        reinterpret_cast<BF16*>(norms_vr.data_ptr<at::BFloat16>()), norms_vr_f32.data_ptr<float>(),
        reg_idx, kv_heads, token_len, max_len, reg_dim, head_dim, packed_reg, 0, reg_bits, start_ptr);

    launch_append_mse_indexed<BF16>(v_ptr,
        rot_vo_t.data_ptr<float>(), bounds_vo.data_ptr<float>(),
        packed_vo.data_ptr<int32_t>(),
        reinterpret_cast<BF16*>(norms_vo.data_ptr<at::BFloat16>()), norms_vo_f32.data_ptr<float>(),
        out_idx, kv_heads, token_len, max_len, out_dim, head_dim, packed_out_dim, 0, out_bits, start_ptr);
}


void append_cache_prod(
    torch::Tensor x,
    torch::Tensor rotation_t,
    torch::Tensor boundaries,
    torch::Tensor centroids,
    torch::Tensor sketch_t,
    torch::Tensor packed_mse,
    torch::Tensor packed_qjl,
    torch::Tensor residual_norms_out,
    torch::Tensor residual_norms_out_f32,
    torch::Tensor norms_out,
    torch::Tensor norms_out_f32,
    int start,
    int mse_bits)
{
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(rotation_t.is_cuda(), "rotation_t must be CUDA");
    TORCH_CHECK(boundaries.is_cuda(), "boundaries must be CUDA");
    TORCH_CHECK(centroids.is_cuda(), "centroids must be CUDA");
    TORCH_CHECK(sketch_t.is_cuda(), "sketch_t must be CUDA");
    TORCH_CHECK(packed_mse.is_cuda(), "packed_mse must be CUDA");
    TORCH_CHECK(packed_qjl.is_cuda(), "packed_qjl must be CUDA");
    TORCH_CHECK(residual_norms_out.is_cuda(), "residual_norms_out must be CUDA");
    TORCH_CHECK(residual_norms_out_f32.is_cuda(), "residual_norms_out_f32 must be CUDA");
    TORCH_CHECK(norms_out.is_cuda(), "norms_out must be CUDA");
    TORCH_CHECK(norms_out_f32.is_cuda(), "norms_out_f32 must be CUDA");

    const int end = start + x.size(1);
    const std::vector<int64_t> last_dim = {-1};
    auto x_f32 = x.to(torch::kFloat32);
    auto norms = at::linalg_vector_norm(x_f32, 2, last_dim, false, torch::kFloat32);
    auto projected = at::matmul(x_f32 / norms.unsqueeze(-1).clamp_min(1e-10), rotation_t);
    auto mse_idx = at::bucketize(projected, boundaries, true, false).to(torch::kInt32);
    auto rotation = rotation_t.transpose(0, 1).contiguous();
    auto flat_idx = mse_idx.reshape({-1}).to(torch::kLong);
    auto centroid_values = at::index_select(centroids, 0, flat_idx).reshape(mse_idx.sizes().vec());
    auto x_mse = at::matmul(centroid_values, rotation) * norms.unsqueeze(-1);
    auto residual = x_f32 - x_mse;
    auto residual_norms = at::linalg_vector_norm(residual, 2, last_dim, false, torch::kFloat32);
    auto qjl = at::matmul(residual, sketch_t).sign().to(torch::kInt32);
    qjl.masked_fill_(qjl == 0, 1);
    qjl = ((qjl + 1) / 2).to(torch::kInt32);

    packed_mse.slice(1, start, end).copy_(turboquant_pack_codes_cuda(mse_idx, mse_bits));
    packed_qjl.slice(1, start, end).copy_(turboquant_pack_codes_cuda(qjl, 1));
    residual_norms_out.slice(1, start, end).copy_(residual_norms.to(residual_norms_out.scalar_type()).unsqueeze(-1));
    residual_norms_out_f32.slice(1, start, end).copy_(residual_norms.unsqueeze(-1));
    norms_out.slice(1, start, end).copy_(norms.to(norms_out.scalar_type()).unsqueeze(-1));
    norms_out_f32.slice(1, start, end).copy_(norms.unsqueeze(-1));
}


// Python-visible API: compute attention QK logits from dense queries and MSE-compressed keys.
torch::Tensor qk_mse(
    torch::Tensor query,
    torch::Tensor packed_idx,
    torch::Tensor norms,
    torch::Tensor rotation_t,
    torch::Tensor centroids,
    int total_len,
    int bits,
    int nh,
    int nh_kv)
{
    TORCH_CHECK(query.size(1) == 1, "qk_mse expects q_len == 1");
    auto query_rot = at::matmul(query.to(torch::kFloat32), rotation_t).contiguous();
    return qk_mse_cuda(
        query_rot,
        packed_idx,
        norms.contiguous(),
        centroids.contiguous(),
        total_len,
        bits,
        nh,
        nh_kv);
}


// Python-visible API: compute attention QK logits from dense queries and Prod-compressed keys.
torch::Tensor qk_prod(
    torch::Tensor query,
    torch::Tensor packed_mse,
    torch::Tensor packed_qjl,
    torch::Tensor residual_norms,
    torch::Tensor norms,
    torch::Tensor rotation_t,
    torch::Tensor sketch_t,
    torch::Tensor centroids,
    float coeff,
    int total_len,
    int mse_bits,
    int nh,
    int nh_kv)
{
    TORCH_CHECK(query.size(1) == 1, "qk_prod expects q_len == 1");
    auto query_f32 = query.to(torch::kFloat32);
    auto query_rot = at::matmul(query_f32, rotation_t).contiguous();
    auto query_proj = at::matmul(query_f32, sketch_t).contiguous();
    return qk_prod_cuda(
        query_rot,
        query_proj,
        packed_mse,
        packed_qjl,
        residual_norms.contiguous(),
        norms.contiguous(),
        centroids.contiguous(),
        coeff,
        total_len,
        mse_bits,
        nh,
        nh_kv);
}


// Python-visible API: compute attention AV output from weights and MSE-compressed values.
torch::Tensor av_mse(
    torch::Tensor attn,
    torch::Tensor packed_idx,
    torch::Tensor norms,
    torch::Tensor centroids,
    torch::Tensor rotation,
    int total_len,
    int bits,
    int nh,
    int nh_kv)
{
    TORCH_CHECK(attn.size(1) == 1, "av_mse expects q_len == 1");
    auto rotated = av_mse_rotated(
        attn.to(torch::kFloat32).contiguous(),
        packed_idx,
        norms.contiguous(),
        centroids.contiguous(),
        total_len,
        bits,
        nh,
        nh_kv);
    const int dim = rotation.size(0);
    return at::matmul(rotated.narrow(-1, 0, dim), rotation).to(attn.scalar_type());
}


// Python-visible API: compute attention AV output in the rotated MSE basis.
torch::Tensor av_mse_rotated(
    torch::Tensor attn,
    torch::Tensor packed_idx,
    torch::Tensor norms,
    torch::Tensor centroids,
    int total_len,
    int bits,
    int nh,
    int nh_kv)
{
    TORCH_CHECK(attn.size(1) == 1, "av_mse_rotated expects q_len == 1");
    auto rotated = av_mse_rotated_cuda(
        attn.to(torch::kFloat32).contiguous(),
        packed_idx,
        norms.contiguous(),
        centroids.contiguous(),
        total_len,
        bits,
        nh,
        nh_kv);
    return rotated;
}


// Python-visible API: compute attention AV output from weights and Prod-compressed values.
torch::Tensor av_prod(
    torch::Tensor attn,
    torch::Tensor packed_mse,
    torch::Tensor packed_qjl,
    torch::Tensor residual_norms,
    torch::Tensor norms,
    torch::Tensor centroids,
    torch::Tensor rotation,
    torch::Tensor sketch,
    float coeff,
    int total_len,
    int mse_bits,
    int nh,
    int nh_kv)
{
    TORCH_CHECK(attn.size(1) == 1, "av_prod expects q_len == 1");
    const int dim = rotation.size(0);
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(attn.device());
    auto output = torch::zeros({attn.size(0), attn.size(1), dim}, options);
    if (mse_bits > 0) {
        auto mse_rotated = av_mse_rotated_cuda(
            attn.to(torch::kFloat32).contiguous(),
            packed_mse,
            norms.contiguous(),
            centroids.contiguous(),
            total_len,
            mse_bits,
            nh,
            nh_kv);
        output += at::matmul(mse_rotated.narrow(-1, 0, dim), rotation);
    }
    auto qjl_centroids = torch::ones({2}, options);
    qjl_centroids.narrow(0, 0, 1).fill_(-1.0f);
    auto qjl_rotated = av_mse_rotated_cuda(
        attn.to(torch::kFloat32).contiguous(),
        packed_qjl,
        residual_norms.contiguous(),
        qjl_centroids,
        total_len,
        1,
        nh,
        nh_kv);
    output += coeff * at::matmul(qjl_rotated.narrow(-1, 0, dim), sketch);
    return output.to(attn.scalar_type());
}


void pack_token_slice(
    torch::Tensor src,
    torch::Tensor dst,
    int token,
    int dst_offset)
{
    TORCH_CHECK(src.is_cuda(), "src must be CUDA");
    TORCH_CHECK(dst.is_cuda(), "dst must be CUDA");
    TORCH_CHECK(src.dim() == 3, "src must be [heads, max_len, dim]");
    TORCH_CHECK(dst.dtype() == torch::kUInt8, "dst must be uint8");
    TORCH_CHECK(dst.dim() == 1, "dst must be flat");
    TORCH_CHECK(src.is_contiguous(), "src must be contiguous");
    TORCH_CHECK(dst.is_contiguous(), "dst must be contiguous");
    TORCH_CHECK(token >= 0 && token < src.size(1), "token out of range");

    const size_t row_bytes = static_cast<size_t>(src.size(2)) * src.element_size();
    const size_t src_pitch = static_cast<size_t>(src.size(1)) * row_bytes;
    const size_t rows = static_cast<size_t>(src.size(0));
    const size_t total_bytes = row_bytes * rows;

    TORCH_CHECK(dst_offset >= 0, "dst_offset must be non-negative");
    TORCH_CHECK(static_cast<size_t>(dst_offset) + total_bytes <= static_cast<size_t>(dst.numel()), "dst too small");

    auto stream = at::cuda::getCurrentCUDAStream(src.device().index());
    const char* src_ptr = reinterpret_cast<const char*>(src.data_ptr()) + static_cast<size_t>(token) * row_bytes;
    char* dst_ptr = reinterpret_cast<char*>(dst.data_ptr<uint8_t>()) + dst_offset;
    AT_CUDA_CHECK(cudaMemcpy2DAsync(
        dst_ptr,
        row_bytes,
        src_ptr,
        src_pitch,
        row_bytes,
        rows,
        cudaMemcpyDeviceToDevice,
        stream.stream()));
}
