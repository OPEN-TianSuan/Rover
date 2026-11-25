#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "gemv.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemv_forward_cuda_outer_dim", &gemv_forward_cuda_outer_dim,
          "Batched quantized GEMV for KV cache attention (CUDA)");
    m.def("append_cache_mse", &append_cache_mse,
          "TurboQuant MSE cache append (CUDA)");
    m.def("append_kv_mse_split", &append_kv_mse_split,
          "TurboQuant fused KV split+append (CUDA)");
    m.def("append_kv_mse_split_graph", &append_kv_mse_split_graph,
          "TurboQuant fused KV split+append graph-safe (CUDA)");
    m.def("append_cache_prod", &append_cache_prod,
          "TurboQuant Prod cache append (CUDA)");
    m.def("qk_mse", &qk_mse,
          "TurboQuant MSE QK attention logits (CUDA)");
    m.def("qk_mse_split", &qk_mse_split,
          "TurboQuant fused QK split (CUDA)");
    m.def("qk_prod", &qk_prod,
          "TurboQuant Prod QK attention logits (CUDA)");
    m.def("av_mse", &av_mse,
          "TurboQuant MSE AV attention output (CUDA)");
    m.def("av_mse_rotated", &av_mse_rotated,
          "TurboQuant MSE AV attention output in rotated basis (CUDA)");
    m.def("av_mse_split", &av_mse_split,
          "TurboQuant fused AV split (CUDA)");
    m.def("qk_mse_split_graph", &qk_mse_split_graph,
          "TurboQuant fused QK split graph-safe (CUDA)");
    m.def("qk_mse_split_graph_rotated", &qk_mse_split_graph_rotated,
          "TurboQuant QK split graph-safe from pre-rotated queries (CUDA)");
    m.def("av_mse_split_graph", &av_mse_split_graph,
          "TurboQuant fused AV split graph-safe (CUDA)");
    m.def("av_mse_split_graph_rotated", &av_mse_split_graph_rotated,
          "TurboQuant AV split graph-safe into rotated value buffers (CUDA)");
    m.def("av_prod", &av_prod,
          "TurboQuant Prod AV attention output (CUDA)");
    m.def("pack_token_slice", &pack_token_slice,
          "Pack one token slice from a strided cache tensor into a flat record buffer (CUDA)");
}
