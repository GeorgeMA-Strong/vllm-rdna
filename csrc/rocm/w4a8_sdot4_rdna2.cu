// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// W4A8 (int4 weights, int8 activations) prefill GEMM for AMD RDNA2 (gfx1030).
//
// Opt-in drop-in replacement for the dense W4A16 prefill GEMM. Two torch ops:
//
//   w4a8_act_quant_rdna2(x, group_size, a_i8, a_scale, a_asum) -> int
//     Per-(token, group) int8 quant of the fp16 activations plus per-token
//     (or per-(token,group), per config) fp32 scales and per-group int32 sums,
//     written in the tiled layout the GEMM below consumes.
//
//   w4a8_gemm_rdna2(a_i8, w_packed, qzeros, scales, a_scale, asum, out,
//                   k, group, zero_offset, config_id, split_k) -> int
//     Runs launch_gemm<Cfg> for the requested config; split_k <= 0 picks the
//     group-aligned split with pick_split_k. Weights are the SAME packed buffer
//     RDNA2W4A16LinearKernel already produces (zero-extended nibbles +
//     gptq_shuffle); see reference.w4a16_rdna2_weights / exllama_shuffle.
//
// The kernel body lives in the sibling header `w4a8_sdot4_rdna2.cuh`
// (copied verbatim from the explore-only tree at
// /tmp/pr9_branch/csrc/rocm/explore/w4a8_sdot4.cuh and renamed so the
// production C ABI and its header are co-located; the source of truth stays
// untouched per the "do not modify csrc/rocm/explore/* in place" rule).
// This TU is the production wrapper: each op returns an int status — 0 on
// success, a positive hipError_t, or a negative "not eligible" code
// (kBadShape / kLdsTooBig / ...) so Python can fall back to
// gptq_gemm_rdna2_prefill when the shape or LDS budget does not fit.
// The M-vs-N threshold decision for routing stays INSIDE C++ (never branch
// on x.size(0) in Python — that misroute cost gfx1100 7x decode).

#include <cstddef>
#include <cstdint>
#include <cstring>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include <hip/hip_runtime.h>

#include "w4a8_sdot4_rdna2.cuh"

namespace ex = vllm::explore_w4a8;

namespace {

enum Err : int {
  kBadShape = -1,
  kBadConfig = -2,
  kNotGfx1030 = -3,
  kLdsTooBig = -4,
  kBadSplit = -5,
  kBadGroup = -6,
};

constexpr int kMaxSplit = 16;

// Recommended config: per-(token, group) activation scales, 32-wide K step,
// 8-row M tile. This is the only A_GROUP tile among the explore sweep and the
// one the G1/G2 accuracy numbers ("per-(token, G=64)") were measured with.
constexpr int kDefaultConfigId = 8;  // a8_lds_k32_ag
constexpr int kDefaultGroup = 64;

const char* error_str(int code) {
  switch (code) {
    case 0:
      return "ok";
    case kBadShape:
      return "bad shape (need N % 8 == 0, K % 32 == 0, K % group == 0)";
    case kBadConfig:
      return "unknown config id";
    case kNotGfx1030:
      return "current device is not gfx1030";
    case kLdsTooBig:
      return "K split does not fit 64 KiB of LDS";
    case kBadSplit:
      return "split_k must divide K/group, be <= 16, and be 1 for f32 out";
    case kBadGroup:
      return "group size must be 32, 64 or 128";
    default:
      return code > 0 ? hipGetErrorString(static_cast<hipError_t>(code))
                      : "unknown error";
  }
}

bool on_gfx1030() {
  thread_local int cached_dev = -1;
  thread_local bool cached_ok = false;
  const int dev = at::cuda::current_device();
  if (dev != cached_dev) {
    hipDeviceProp_t prop;
    cached_ok = hipGetDeviceProperties(&prop, dev) == hipSuccess &&
                std::strncmp(prop.gcnArchName, "gfx1030", 7) == 0;
    cached_dev = dev;
  }
  return cached_ok;
}

int group_index(int64_t group_size) {
  switch (group_size) {
    case 32:
      return 0;
    case 64:
      return 1;
    case 128:
      return 2;
    default:
      return -1;
  }
}

struct GemmArgs {
  const int8_t* a;
  const uint32_t* w;
  const uint32_t* qzeros;
  const ex::f16_t* scales;
  const float* a_scale;
  const int32_t* asum;
  void* out;
  int m, n, k, zero_offset, split_k, out_f32;
};

// Same rule as reference.pick_split_k: LDS budget by grid size, then grow
// while the grid is small or the K range long, restricted to group-aligned
// splits (K % (group * split) == 0).
template <class C>
int pick_split_k(int m, int n, int k) {
  const int blocks =
      ((m + C::M_TILE - 1) / C::M_TILE) * ((n + C::N_TILE - 1) / C::N_TILE);
  const int budget = blocks > 1024  ? 16 * 1024
                     : blocks > 256 ? 64 * 1024
                                    : 32 * 1024;
  const int groups = k / C::GROUP;
  int splits[kMaxSplit];
  int count = 0;
  for (int s = 1; s <= kMaxSplit; ++s) {
    if (groups % s == 0) {
      splits[count++] = s;
    }
  }
  auto lds = [&](int s) {
    const int kps = k / s;
    return C::M_TILE * kps +
           C::M_TILE * (kps / C::GROUP) * (C::A_GROUP ? 8 : 4);
  };
  int i = 0;
  while (i + 1 < count && lds(splits[i]) > budget) {
    ++i;
  }
  while (i + 1 < count && (blocks * splits[i] < 2048 || k / splits[i] > 2048)) {
    if (lds(splits[i + 1]) > budget) {
      break;
    }
    ++i;
  }
  while (i + 1 < count && ex::lds_bytes<C>(k / splits[i]) > 64 * 1024) {
    ++i;
  }
  return splits[i];
}

template <class C>
int launch_gemm(const GemmArgs& p, hipStream_t stream) {
  if (p.m <= 0 || p.n <= 0 || p.n % 8 || p.k % 32 || p.k % C::GROUP) {
    return kBadShape;
  }
  const int groups = p.k / C::GROUP;
  const int split = p.split_k > 0 ? p.split_k : pick_split_k<C>(p.m, p.n, p.k);
  if (split > kMaxSplit || groups % split || (p.out_f32 && split != 1)) {
    return kBadSplit;
  }
  const int k_per_split = p.k / split;
  const int lds = ex::lds_bytes<C>(k_per_split);
  if (lds > 64 * 1024) {
    return kLdsTooBig;
  }
  if (!p.out_f32 && split > 1) {
    const hipError_t e = hipMemsetAsync(
        p.out, 0, static_cast<size_t>(p.m) * p.n * sizeof(ex::f16_t), stream);
    if (e != hipSuccess) {
      return static_cast<int>(e);
    }
  }
  const dim3 grid((p.n + C::N_TILE - 1) / C::N_TILE,
                  (p.m + C::M_TILE - 1) / C::M_TILE, split);
  auto* kernel = ex::w4a8_gemm_kernel<C>;
  kernel<<<grid, dim3(C::THREADS), lds, stream>>>(
      p.a, p.w, p.qzeros, p.scales, p.a_scale, p.asum, p.out, p.m, p.n, p.k,
      p.zero_offset, k_per_split, split, p.out_f32);
  const hipError_t e = hipGetLastError();
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return e == hipSuccess ? 0 : static_cast<int>(e);
}

using LaunchFn = int (*)(const GemmArgs&, hipStream_t);
using SplitFn = int (*)(int, int, int);

struct ConfigEntry {
  int id;
  int m_tile;
  int a_group;
  LaunchFn launch[3];  // group 32, 64, 128
};

#define W4A8_CFG(th, npt, ks, mt, g, src, ag) \
  ex::Cfg<th, npt, ks, mt, g, ex::ASrc::src, (ag) != 0>
#define W4A8_ENTRY(id, name, th, npt, ks, mt, src, ag)                    \
  {id,                                                                   \
   mt,                                                                   \
   ag,                                                                   \
   {&launch_gemm<W4A8_CFG(th, npt, ks, mt, 32, src, ag)>,                \
    &launch_gemm<W4A8_CFG(th, npt, ks, mt, 64, src, ag)>,                \
    &launch_gemm<W4A8_CFG(th, npt, ks, mt, 128, src, ag)>}},

const ConfigEntry kConfigs[] = {W4A8_EXPLORE_CONFIGS(W4A8_ENTRY)};
constexpr int kNumConfigs = sizeof(kConfigs) / sizeof(kConfigs[0]);

const ConfigEntry* find_config(int id) {
  for (const ConfigEntry& c : kConfigs) {
    if (c.id == id) {
      return &c;
    }
  }
  return nullptr;
}

// Thread t owns row (t % MT) of its block's row tile; the per-group activation
// scale variant (A_GROUP) needs no block reduction, so the launch is identical
// for both — only the a_scale layout differs (per token vs [T][K/G][MT]).
template <int MT, bool PER_GROUP>
int launch_act_quant(const void* x, int64_t x_row_stride, void* a, void* a_scale,
                     void* asum, int m, int k, int group_size,
                     hipStream_t stream) {
  auto* kernel = ex::w4a8_act_quant_kernel<256, MT, PER_GROUP>;
  kernel<<<dim3((m + MT - 1) / MT), dim3(256), 0, stream>>>(
      static_cast<const ex::f16_t*>(x), x_row_stride, static_cast<int8_t*>(a),
      static_cast<float*>(a_scale), static_cast<int32_t*>(asum), m, k,
      group_size);
  const hipError_t e = hipGetLastError();
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return e == hipSuccess ? 0 : static_cast<int>(e);
}

}  // namespace

// ---------------------------------------------------------------------------
// torch ops
// ---------------------------------------------------------------------------

// x [M, K] fp16 -> a_i8 [T][K/8][MT][8], a_scale [T][K/G][MT] f32 (per-(token,
// group) for the A_GROUP config) and a_asum [T][K/G][MT] int32. MT is the M
// tile of the configured GEMM (a8_lds_k32_ag uses 8). Returns 0 or an error.
int64_t w4a8_act_quant_rdna2(const at::Tensor& x, int64_t group_size,
                         at::Tensor& a_i8, at::Tensor& a_scale,
                         at::Tensor& a_asum) {
  if (!on_gfx1030()) {
    return kNotGfx1030;
  }
  if (group_index(group_size) < 0) {
    return kBadGroup;
  }
  const int64_t m = x.size(0);
  const int64_t k = x.size(1);
  const int64_t row_stride = x.stride(0);
  if (m <= 0 || k % 32 || k % group_size || row_stride % 8 || row_stride < k) {
    return kBadShape;
  }
  const at::cuda::OptionalCUDAGuard guard(x.device());
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();
  // The wired path uses the A_GROUP config (per-(token, group) scales) with an
  // 8-row M tile; keep the M tile selection here so a config change only needs
  // one edit, and so Python never branches on M.
  constexpr int kMTile = 8;
  constexpr bool kPerGroup = true;
  return launch_act_quant<kMTile, kPerGroup>(
      x.data_ptr(), row_stride, a_i8.data_ptr(), a_scale.data_ptr(),
      a_asum.data_ptr(), static_cast<int>(m), static_cast<int>(k),
      static_cast<int>(group_size), stream);
}

int64_t w4a8_gemm_rdna2(const at::Tensor& a_i8, const at::Tensor& w_packed,
                    const at::Tensor& qzeros, const at::Tensor& scales,
                    const at::Tensor& a_scale, const at::Tensor& asum,
                    at::Tensor& out, int64_t k, int64_t group,
                    int64_t zero_offset, int64_t config_id, int64_t split_k) {
  if (!on_gfx1030()) {
    return kNotGfx1030;
  }
  static_assert(kNumConfigs > 0, "empty W4A8 config table");
  const int id = config_id > 0 ? static_cast<int>(config_id) : kDefaultConfigId;
  const int grp = group > 0 ? static_cast<int>(group) : kDefaultGroup;
  const ConfigEntry* c = find_config(id);
  const int gi = group_index(grp);
  if (!c) {
    return kBadConfig;
  }
  if (gi < 0) {
    return kBadGroup;
  }
  const at::cuda::OptionalCUDAGuard guard(out.device());
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();
  const GemmArgs p{static_cast<const int8_t*>(a_i8.data_ptr()),
                   static_cast<const uint32_t*>(w_packed.data_ptr()),
                   static_cast<const uint32_t*>(qzeros.data_ptr()),
                   static_cast<const ex::f16_t*>(scales.data_ptr()),
                   static_cast<const float*>(a_scale.data_ptr()),
                   static_cast<const int32_t*>(asum.data_ptr()),
                   out.data_ptr(),
                   static_cast<int>(out.size(0)),
                   static_cast<int>(out.size(1)),
                   static_cast<int>(k),
                   static_cast<int>(zero_offset),
                   static_cast<int>(split_k),
                   out.scalar_type() == at::kFloat};
  return c->launch[gi](p, stream);
}
