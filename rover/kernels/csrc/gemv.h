#pragma once
#include <torch/extension.h>

// Batched quantized GEMV: output = inputs @ dequant(weight)^T
// Supports GQA via nh / nh_kv ratio.
//
// inputs:           (BS, M, IC)                  bfloat16
// weight:           (BS_kv, OC // pack_factor, IC) int32  (packed along OC)
// scaling_factors:  (BS_kv, OC // group_size, IC)  bfloat16
// zeros:            (BS_kv, OC // group_size, IC)  bfloat16
// output:           (BS, M, OC)                  bfloat16
//
// BS    = B * nh (total query batches)
// BS_kv = B * nh_kv
// nh / nh_kv = GQA ratio
torch::Tensor gemv_forward_cuda_outer_dim(
    torch::Tensor inputs,
    torch::Tensor weight,
    torch::Tensor scaling_factors,
    torch::Tensor zeros,
    int bit,
    int group_size,
    int nh,
    int nh_kv);

void append_cache_mse(
    torch::Tensor x,
    torch::Tensor rotation_t,
    torch::Tensor boundaries,
    torch::Tensor packed_out,
    torch::Tensor norms_out,
    torch::Tensor norms_out_f32,
    int start,
    int bits);

void append_kv_mse_split(
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
    int start,
    int reg_bits,
    int out_bits);

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
    torch::Tensor start_tensor,
    int reg_bits,
    int out_bits);

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
    int mse_bits);

torch::Tensor qk_mse(
    torch::Tensor query,
    torch::Tensor packed_idx,
    torch::Tensor norms,
    torch::Tensor rotation_t,
    torch::Tensor centroids,
    int total_len,
    int bits,
    int nh,
    int nh_kv);

torch::Tensor qk_mse_split(
    torch::Tensor query,
    torch::Tensor regular_idx,
    torch::Tensor outlier_idx,
    torch::Tensor rot_reg_t,
    torch::Tensor rot_out_t,
    torch::Tensor packed_reg,
    torch::Tensor norms_reg,
    torch::Tensor centroids_reg,
    torch::Tensor packed_out,
    torch::Tensor norms_out,
    torch::Tensor centroids_out,
    int total_len,
    int reg_bits,
    int out_bits,
    int nh,
    int nh_kv);

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
    int nh_kv);

torch::Tensor av_mse(
    torch::Tensor attn,
    torch::Tensor packed_idx,
    torch::Tensor norms,
    torch::Tensor centroids,
    torch::Tensor rotation,
    int total_len,
    int bits,
    int nh,
    int nh_kv);

torch::Tensor av_mse_rotated(
    torch::Tensor attn,
    torch::Tensor packed_idx,
    torch::Tensor norms,
    torch::Tensor centroids,
    int total_len,
    int bits,
    int nh,
    int nh_kv);

torch::Tensor av_mse_split(
    torch::Tensor attn,
    torch::Tensor regular_idx,
    torch::Tensor outlier_idx,
    torch::Tensor packed_vr,
    torch::Tensor norms_vr,
    torch::Tensor centroids_vr,
    torch::Tensor rotation_reg,
    torch::Tensor packed_vo,
    torch::Tensor norms_vo,
    torch::Tensor centroids_vo,
    torch::Tensor rotation_out,
    int total_len,
    int reg_bits,
    int out_bits,
    int nh,
    int nh_kv,
    int head_dim);

void qk_mse_split_graph(
    torch::Tensor query,
    torch::Tensor regular_idx,
    torch::Tensor outlier_idx,
    torch::Tensor rot_reg_t,
    torch::Tensor rot_out_t,
    torch::Tensor packed_reg,
    torch::Tensor norms_reg,
    torch::Tensor centroids_reg,
    torch::Tensor packed_out,
    torch::Tensor norms_out,
    torch::Tensor centroids_out,
    torch::Tensor scores,
    torch::Tensor total_len_tensor,
    int max_len,
    int reg_bits,
    int out_bits,
    int nh,
    int nh_kv);

void qk_mse_split_graph_rotated(
    torch::Tensor q_reg_rot,
    torch::Tensor q_out_rot,
    torch::Tensor packed_reg,
    torch::Tensor norms_reg,
    torch::Tensor centroids_reg,
    torch::Tensor packed_out,
    torch::Tensor norms_out,
    torch::Tensor centroids_out,
    torch::Tensor scores,
    torch::Tensor total_len_tensor,
    int max_len,
    int reg_bits,
    int out_bits,
    int nh,
    int nh_kv);

void av_mse_split_graph(
    torch::Tensor attn,
    torch::Tensor regular_idx,
    torch::Tensor outlier_idx,
    torch::Tensor packed_vr,
    torch::Tensor norms_vr,
    torch::Tensor centroids_vr,
    torch::Tensor rotation_reg,
    torch::Tensor packed_vo,
    torch::Tensor norms_vo,
    torch::Tensor centroids_vo,
    torch::Tensor rotation_out,
    torch::Tensor reg_rotated_buf,
    torch::Tensor out_rotated_buf,
    torch::Tensor output,
    torch::Tensor total_len_tensor,
    int max_len,
    int reg_bits,
    int out_bits,
    int nh,
    int nh_kv,
    int head_dim);

void av_mse_split_graph_rotated(
    torch::Tensor attn,
    torch::Tensor packed_vr,
    torch::Tensor norms_vr,
    torch::Tensor centroids_vr,
    torch::Tensor packed_vo,
    torch::Tensor norms_vo,
    torch::Tensor centroids_vo,
    torch::Tensor reg_rotated_buf,
    torch::Tensor out_rotated_buf,
    torch::Tensor total_len_tensor,
    int max_len,
    int reg_bits,
    int out_bits,
    int nh,
    int nh_kv);

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
    int nh_kv);

void pack_token_slice(
    torch::Tensor src,
    torch::Tensor dst,
    int token,
    int dst_offset);
