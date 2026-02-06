
#include <torch/extension.h>

extern "C" torch::Tensor dot_mod_p31_launcher(
    torch::Tensor vals,
    torch::Tensor rpow,
    int block_size,
    int max_blocks);

extern "C" void fill_rpow_blocks_launcher(
    torch::Tensor base,      // int64 CUDA [BS]
    torch::Tensor anchors,   // int64 CUDA [blocks]
    torch::Tensor out,       // int64 CUDA [L]
    int BS);

extern "C" void fused_sketch_launcher(
    torch::Tensor vals,            // int64 CUDA [N]
    torch::Tensor challenges,      // int64 [k]
    torch::Tensor out_S,           // int64 CUDA [k]
    int block_size,
    int max_blocks);

static torch::Tensor fused_sketch(
    torch::Tensor vals,
    torch::Tensor challenges,
    int block_size,
    int max_blocks){
  TORCH_CHECK(vals.is_cuda(), "vals must be CUDA tensor");
  auto opts = torch::TensorOptions().device(vals.device()).dtype(torch::kInt64);
  auto out = torch::empty({challenges.size(0)}, opts);
  fused_sketch_launcher(vals, challenges, out, block_size, max_blocks);
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  m.def("dot_mod_p", &dot_mod_p31_launcher, "Streaming dot mod 2^31-1 (CUDA)");
  m.def("fill_rpow_blocks", &fill_rpow_blocks_launcher, "Fill r^i via tiling (CUDA)");
  m.def("fused_sketch", &fused_sketch, "Fused multi-challenge sketch over full vector (CUDA)");
}
