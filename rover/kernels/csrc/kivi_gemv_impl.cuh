#pragma once

// 2-bit kernel: pack_factor = 16
__global__ void bgemv2_outer_dim(
    const __nv_bfloat16* __restrict__ inputs,
    const uint32_t* __restrict__ weight,
    const __nv_bfloat16* __restrict__ zeros,
    const __nv_bfloat16* __restrict__ scales,
    __nv_bfloat16* __restrict__ outputs,
    int IC, int OC, int group_size, int nh, int nh_kv)
{
    constexpr int bit        = 2;
    constexpr int pack_factor = 16;  // 32 / 2
    constexpr int num = (1 << bit) - 1;

    const int bs_idx         = blockIdx.x;            // query batch index (B * nh)
    const int packed_oc_idx  = blockIdx.y * blockDim.y + threadIdx.y;  // which pack of OC
    const int m_idx          = blockIdx.z;            // query position (usually 0 for decode)
    const int oc_start       = packed_oc_idx * pack_factor;
    const int group_idx      = oc_start / group_size;

    // GQA: map query batch to kv batch
    const int kv_bs_idx = bs_idx / (nh / nh_kv);

    const __nv_bfloat16* inp = inputs  + bs_idx * IC * gridDim.z + m_idx * IC;
    __nv_bfloat16*       out = outputs + bs_idx * OC * gridDim.z + m_idx * OC;

    float psum[pack_factor] = {};

    for (int tile = 0; tile < (IC + TILE_DIM - 1) / TILE_DIM; tile++) {
        // Each thread in the warp handles 4 consecutive IC positions
        int k_base = tile * TILE_DIM + threadIdx.x * 4;

        uint32_t qw[4]  = {};
        __nv_bfloat16 inp4[4] = {};
        __nv_bfloat16 sc4[4]  = {};
        __nv_bfloat16 zr4[4]  = {};

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int k = k_base + i;
            if (k < IC) {
                inp4[i] = inp[k];
            }
            // weight layout: [OC/pack_factor, IC], row = packed_oc_idx
            int w_offset = packed_oc_idx * IC + k;
            if (oc_start < OC && w_offset < (OC / pack_factor) * IC)
                qw[i] = weight[kv_bs_idx * (OC / pack_factor) * IC + w_offset];
            // scale/zero layout: [OC/group_size, IC], row = group_idx
            int sg_offset = group_idx * IC + k;
            if (oc_start < OC && sg_offset < (OC / group_size) * IC) {
                sc4[i] = scales[kv_bs_idx * (OC / group_size) * IC + sg_offset];
                zr4[i] = zeros [kv_bs_idx * (OC / group_size) * IC + sg_offset];
            }
        }

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            if (k_base + i >= IC) break;
            float cur_inp   = __bfloat162float(inp4[i]);
            float cur_scale = __bfloat162float(sc4[i]);
            float cur_zero  = __bfloat162float(zr4[i]);
            uint32_t packed = qw[i];
            #pragma unroll
            for (int j = 0; j < pack_factor; j++) {
                int oc = oc_start + j;
                if (oc < OC) {
                    float w = (float)(packed & num);
                    psum[j] += (cur_scale * w + cur_zero) * cur_inp;
                    packed >>= bit;
                }
            }
        }
    }

    // Warp reduce and write output
    #pragma unroll
    for (int j = 0; j < pack_factor; j++) {
        int oc = oc_start + j;
        if (oc < OC) {
            psum[j] = warp_reduce_sum(psum[j]);
            if (threadIdx.x == 0)
                out[oc] = __float2bfloat16(psum[j]);
        }
    }
}


// 4-bit kernel: pack_factor = 8
__global__ void bgemv4_outer_dim(
    const __nv_bfloat16* __restrict__ inputs,
    const uint32_t* __restrict__ weight,
    const __nv_bfloat16* __restrict__ zeros,
    const __nv_bfloat16* __restrict__ scales,
    __nv_bfloat16* __restrict__ outputs,
    int IC, int OC, int group_size, int nh, int nh_kv)
{
    constexpr int bit        = 4;
    constexpr int pack_factor = 8;  // 32 / 4
    constexpr int num = (1 << bit) - 1;

    const int bs_idx         = blockIdx.x;
    const int packed_oc_idx  = blockIdx.y * blockDim.y + threadIdx.y;
    const int m_idx          = blockIdx.z;
    const int oc_start       = packed_oc_idx * pack_factor;
    const int group_idx      = oc_start / group_size;
    const int kv_bs_idx      = bs_idx / (nh / nh_kv);

    const __nv_bfloat16* inp = inputs  + bs_idx * IC * gridDim.z + m_idx * IC;
    __nv_bfloat16*       out = outputs + bs_idx * OC * gridDim.z + m_idx * OC;

    float psum[pack_factor] = {};

    for (int tile = 0; tile < (IC + TILE_DIM - 1) / TILE_DIM; tile++) {
        int k_base = tile * TILE_DIM + threadIdx.x * 4;

        uint32_t qw[4]   = {};
        __nv_bfloat16 inp4[4] = {};
        __nv_bfloat16 sc4[4]  = {};
        __nv_bfloat16 zr4[4]  = {};

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int k = k_base + i;
            if (k < IC) inp4[i] = inp[k];
            int w_offset  = packed_oc_idx * IC + k;
            int sg_offset = group_idx * IC + k;
            if (oc_start < OC && w_offset < (OC / pack_factor) * IC)
                qw[i] = weight[kv_bs_idx * (OC / pack_factor) * IC + w_offset];
            if (oc_start < OC && sg_offset < (OC / group_size) * IC) {
                sc4[i] = scales[kv_bs_idx * (OC / group_size) * IC + sg_offset];
                zr4[i] = zeros [kv_bs_idx * (OC / group_size) * IC + sg_offset];
            }
        }

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            if (k_base + i >= IC) break;
            float cur_inp   = __bfloat162float(inp4[i]);
            float cur_scale = __bfloat162float(sc4[i]);
            float cur_zero  = __bfloat162float(zr4[i]);
            uint32_t packed = qw[i];
            #pragma unroll
            for (int j = 0; j < pack_factor; j++) {
                int oc = oc_start + j;
                if (oc < OC) {
                    float w = (float)(packed & num);
                    psum[j] += (cur_scale * w + cur_zero) * cur_inp;
                    packed >>= bit;
                }
            }
        }
    }

    #pragma unroll
    for (int j = 0; j < pack_factor; j++) {
        int oc = oc_start + j;
        if (oc < OC) {
            psum[j] = warp_reduce_sum(psum[j]);
            if (threadIdx.x == 0)
                out[oc] = __float2bfloat16(psum[j]);
        }
    }
}


torch::Tensor gemv_forward_cuda_outer_dim(
    torch::Tensor inputs,
    torch::Tensor weight,
    torch::Tensor scaling_factors,
    torch::Tensor zeros,
    int bit,
    int group_size,
    int nh,
    int nh_kv)
{
    TORCH_CHECK(bit == 2 || bit == 4, "Only 2-bit and 4-bit supported");
    int BS  = inputs.size(0);
    int M   = inputs.size(1);
    int IC  = inputs.size(2);
    int pack_factor = 32 / bit;
    int OC  = zeros.size(1) * group_size;

    auto out = torch::empty({BS, M, OC},
        torch::TensorOptions().dtype(inputs.dtype()).device(inputs.device()));

    auto in_ptr  = reinterpret_cast<const __nv_bfloat16*>(inputs.data_ptr<at::BFloat16>());
    auto wt_ptr  = reinterpret_cast<const uint32_t*>(weight.data_ptr<int>());
    auto sc_ptr  = reinterpret_cast<const __nv_bfloat16*>(scaling_factors.data_ptr<at::BFloat16>());
    auto zr_ptr  = reinterpret_cast<const __nv_bfloat16*>(zeros.data_ptr<at::BFloat16>());
    auto out_ptr = reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>());

    // Grid: (BS, OC/pack_factor/4, M), Block: (WARP_SIZE, 4)
    // Each block computes pack_factor*4 = 64 outputs (2-bit) or 32 outputs (4-bit)
    dim3 grid(BS, (OC / pack_factor + 3) / 4, M);
    dim3 block(WARP_SIZE, 4);

    if (bit == 2) {
        bgemv2_outer_dim<<<grid, block>>>(
            in_ptr, wt_ptr, zr_ptr, sc_ptr, out_ptr,
            IC, OC, group_size, nh, nh_kv);
    } else {
        bgemv4_outer_dim<<<grid, block>>>(
            in_ptr, wt_ptr, zr_ptr, sc_ptr, out_ptr,
            IC, OC, group_size, nh, nh_kv);
    }

    return out;
}
