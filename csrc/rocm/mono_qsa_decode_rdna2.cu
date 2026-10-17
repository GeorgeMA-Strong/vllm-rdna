// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// MonoKernel QSA (gfx1030, mono_qsa_decode_rdna2; part of E3): QSA sparse
// paged GQA attention for decode / MTP verify rows
// in one launch (today: _qsa_sparse_paged_gqa_splitk_kernel +
// _qsa_merge_splitk_kernel). Each row attends to its selected tokens (a
// valid prefix of `indices`, -1 padded) through the paged K/V cache.
//
// grid (rows * kv_heads, splits), 256 threads = 8 waves. Lane l owns head dims
// 8l..8l+7 (head_dim 256); a wave walks every 8th selected token of its split
// with an online softmax per query head of the group (<= 8 heads), so each
// K/V row is read once for the whole group. The 8 waves combine in LDS in a
// fixed order; the last split of a (row, kv head) to finish merges the
// splits' (output, log2-sum-exp) partials and writes fp16 out. Like the
// Triton kernel, probabilities are rounded to fp16 before P @ V.

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 qsa_h2v __attribute__((ext_vector_type(2)));

namespace qsa_dec {

using h2v = ::qsa_h2v;

constexpr int kHD = 256, kThreads = 256, kWaves = 8, kMaxG = 8;
#ifndef QSA_DEC_STEP
  #define QSA_DEC_STEP 2
#endif
constexpr int kStep = QSA_DEC_STEP;  // tokens per wave step
#ifndef QSA_DEC_CHUNK
  #define QSA_DEC_CHUNK 8
#endif
constexpr int kChunk = QSA_DEC_CHUNK;  // tokens a wave resolves at once (<= 32)
constexpr int kMaxS = 64;              // splits the merge holds

struct Args {
  const half* q;  // [M, QH, HD]
  long sq_row, sq_head;
  const half* k;  // [pages, page, KVH, HD]
  const half* v;
  long sk_blk, sk_tok, sk_head, sv_blk, sv_tok, sv_head;
  const int* idx;  // [M, W]
  long sidx;
  const int* btab;  // [R, PW]
  long sbt;
  const int* t2r;  // [M]
  half* out;       // [M, QH, HD]
  long so_row, so_head;
  float* po;    // [S, M, QH, HD] fp32 partial outputs
  float* plse;  // [S, M, QH] log2-sum-exp
  int* cnt;     // [M * KVH] split arrivals (zero between calls)
  int M, QH, KVH, G, W, PAGE, PW, R, NB, S;
  float scale_log2;
};

__device__ __forceinline__ float wsum(float x) {
#pragma unroll
  for (int o = 16; o >= 1; o >>= 1) x += __shfl_xor(x, o);
  return x;
}

template <int G>
__global__ void __launch_bounds__(kThreads) qsa_dec_k(const Args a) {
  const int row = blockIdx.x / a.KVH, kvh = blockIdx.x % a.KVH,
            split = blockIdx.y;
  const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31, d0 = lane * 8;
  const int req = a.t2r[row];
  const bool req_ok = req >= 0 && req < a.R;
  const int h0 = kvh * G;

  // q kept packed (fp16 pairs) for v_dot2_f32_f16: fp32-exact products, half
  // the registers of an fp32 copy (8 waves x G <= 8 heads must not spill)
  h2v q[G][4];
#pragma unroll
  for (int h = 0; h < G; h++) {
    const half* qp = a.q + row * a.sq_row + (long)(h0 + h) * a.sq_head + d0;
    const uint4 raw = *reinterpret_cast<const uint4*>(qp);
    const h2v* hv = reinterpret_cast<const h2v*>(&raw);
#pragma unroll
    for (int e = 0; e < 4; e++) q[h][e] = hv[e];
  }
  float m[G], l[G], acc[G][8];
#pragma unroll
  for (int h = 0; h < G; h++) {
    m[h] = -1.0e20f;
    l[h] = 0.f;
#pragma unroll
    for (int e = 0; e < 8; e++) acc[h][e] = 0.f;
  }

  const int j0 = (int)((long)split * a.W / a.S),
            j1 = (int)((long)(split + 1) * a.W / a.S);
  const int* irow = a.idx + row * a.sidx;
  const int* brow = a.btab + (long)(req_ok ? req : 0) * a.sbt;
  // the selection is a valid prefix + -1 padding: a split starting on padding
  // has nothing to attend to (at short contexts about half of them)
  __shared__ int s_empty;
  if (tid == 0) s_empty = a.S > 1 && (j0 >= j1 || !req_ok || irow[j0] < 0);
  __syncthreads();
  const long base = (long)split * a.M + row;
  __shared__ int s_last;
  __shared__ float s_L[kMaxG], s_M[kMaxG];  // split partial, then merge weights
  if (s_empty) {
    if (tid < G) a.plse[base * a.QH + h0 + tid] = -INFINITY;
  } else {
    // a wave takes chunks of 32 selected tokens: lane i resolves token i's page
    // (index + block-table loads in parallel), then the wave walks the chunk
    // with the page / offset broadcast from the owning lane, K / V loads kept 2
    // ahead
#ifndef QSA_DEC_NOLOOP
    for (int c0 = j0 + wave * kChunk; c0 < j1; c0 += kWaves * kChunk) {
      long my_off = -1;
      {
        const int j = c0 + (lane < kChunk ? lane : kChunk);
        const int tok = (j < j1 && req_ok) ? irow[j] : -1;
        const int lp = tok >= 0 ? tok / a.PAGE : 0;
        const int pg = (tok >= 0 && lp < a.PW) ? brow[lp] : -1;
        if (pg >= 0 && pg < a.NB)
          my_off = (long)pg * a.PAGE + (tok - lp * a.PAGE);
      }
      const int n = min(kChunk, j1 - c0);
      auto fetch = [&](int t, uint4& kr, uint4& vr) -> bool {
        const long so = __shfl(my_off, t < n ? t : 0);
        if (t >= n || so < 0) return false;
        const long pg = so / a.PAGE, off = so - pg * a.PAGE;
        kr = *reinterpret_cast<const uint4*>(
            a.k + pg * a.sk_blk + off * a.sk_tok + kvh * a.sk_head + d0);
        vr = *reinterpret_cast<const uint4*>(
            a.v + pg * a.sv_blk + off * a.sv_tok + kvh * a.sv_head + d0);
        return true;
      };
      // 4 tokens per step: their K / V loads are in flight together, and the
      // online softmax is updated once per step (tile max, like the Triton
      // tile)
      for (int t = 0; t < n; t += kStep) {
        uint4 kr[kStep], vr[kStep];
        bool ok[kStep];
        bool any = false;
  #pragma unroll
        for (int u = 0; u < kStep; u++) {
          kr[u] = vr[u] = make_uint4(0, 0, 0, 0);
          ok[u] = fetch(t + u, kr[u], vr[u]);
          any |= ok[u];
        }
        if (!any) continue;
        float sc[kStep][G];
  #pragma unroll
        for (int u = 0; u < kStep; u++) {
          const ::qsa_h2v* kh = reinterpret_cast<const ::qsa_h2v*>(&kr[u]);
  #pragma unroll
          for (int h = 0; h < G; h++) {
            float x = 0.f;
  #pragma unroll
            for (int e = 0; e < 4; e++)
              x = __builtin_amdgcn_fdot2(q[h][e], kh[e], x, false);
            sc[u][h] = x;
          }
        }
  #pragma unroll
        for (int o = 16; o >= 1; o >>= 1)
  #pragma unroll
          for (int u = 0; u < kStep; u++)
  #pragma unroll
            for (int h = 0; h < G; h++) sc[u][h] += __shfl_xor(sc[u][h], o);
  #pragma unroll
        for (int h = 0; h < G; h++) {
          float mt = m[h];
  #pragma unroll
          for (int u = 0; u < kStep; u++) {
            sc[u][h] = ok[u] ? sc[u][h] * a.scale_log2 : -1.0e20f;
            mt = fmaxf(mt, sc[u][h]);
          }
          const float al = exp2f(m[h] - mt);
          float ps = 0.f;
  #pragma unroll
          for (int e = 0; e < 8; e++) acc[h][e] *= al;
  #pragma unroll
          for (int u = 0; u < kStep; u++) {
            const float pu = ok[u] ? exp2f(sc[u][h] - mt) : 0.f;
            ps += pu;
            const float p16 = __half2float(__float2half(pu));
            const half* vh = reinterpret_cast<const half*>(&vr[u]);
  #pragma unroll
            for (int e = 0; e < 8; e++) acc[h][e] += p16 * __half2float(vh[e]);
          }
          l[h] = l[h] * al + ps;
          m[h] = mt;
        }
      }
    }
#endif
    // combine the 8 waves in a fixed order
    __shared__ float s_m[kWaves][kMaxG], s_l[kWaves][kMaxG];
    __shared__ float s_acc[kMaxG][kHD];
    if (lane == 0)
#pragma unroll
      for (int h = 0; h < G; h++) {
        s_m[wave][h] = m[h];
        s_l[wave][h] = l[h];
      }
    for (int i = tid; i < G * kHD; i += kThreads) s_acc[i / kHD][i % kHD] = 0.f;
    __syncthreads();
    float mx[G], wl[G];
#pragma unroll
    for (int h = 0; h < G; h++) {
      float mm = -1.0e20f;
      for (int w = 0; w < kWaves; w++) mm = fmaxf(mm, s_m[w][h]);
      mx[h] = mm;
      wl[h] = exp2f(m[h] - mm);
    }
    for (int w = 0; w < kWaves; w++) {
      if (wave == w)
#pragma unroll
        for (int h = 0; h < G; h++)
#pragma unroll
          for (int e = 0; e < 8; e++) s_acc[h][d0 + e] += acc[h][e] * wl[h];
      __syncthreads();
    }
    // this split's partial: o = acc / L, lse = M + log2 L  (L = 0: empty split)
    if (tid < G) {
      float mm = -1.0e20f;
      for (int w = 0; w < kWaves; w++) mm = fmaxf(mm, s_m[w][tid]);
      float L = 0.f;
      for (int w = 0; w < kWaves; w++)
        L += s_l[w][tid] * exp2f(s_m[w][tid] - mm);
      s_L[tid] = L;
      s_M[tid] = mm;
    }
    __syncthreads();
    for (int i = tid; i < G * kHD; i += kThreads) {
      const int h = i / kHD, d = i % kHD;
      const float L = s_L[h];
      const float o = L > 0.f ? s_acc[h][d] / L : 0.f;
      if (a.S == 1) {
        a.out[row * a.so_row + (long)(h0 + h) * a.so_head + d] =
            __float2half(o);
      } else {
        a.po[(base * a.QH + h0 + h) * kHD + d] = o;
        if (d == 0)
          a.plse[base * a.QH + h0 + h] =
              L > 0.f ? s_M[h] + log2f(L) : -INFINITY;
      }
    }
    if (a.S == 1) return;
  }  // non-empty split
  __threadfence();
  __syncthreads();
  if (tid == 0) s_last = atomicAdd(&a.cnt[blockIdx.x], 1) == a.S - 1;
  __syncthreads();
  if (!s_last) return;
  // acquire at agent scope invalidates this CU's L0/L1, so the other splits'
  // partials can be read with plain (pipelined) loads
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  // merge the splits (last split of this (row, kv head))
  __shared__ float s_w[kMaxG][kMaxS];
  {
    // weights: thread (h, s) for every split of every head of the group
    for (int i = tid; i < G * a.S; i += kThreads) {
      const int hl = i / a.S, s2 = i % a.S;
      s_w[hl][s2] = a.plse[((long)s2 * a.M + row) * a.QH + h0 + hl];
    }
    __syncthreads();
    if (tid < G) {
      float mm = -INFINITY;
      for (int s2 = 0; s2 < a.S; s2++) mm = fmaxf(mm, s_w[tid][s2]);
      float den = 0.f;
      for (int s2 = 0; s2 < a.S; s2++) {
        const float wgt = mm > -INFINITY ? exp2f(s_w[tid][s2] - mm) : 0.f;
        s_w[tid][s2] = wgt;
        den += wgt;
      }
      s_L[tid] = den;
    }
    __syncthreads();
  }
  for (int i = tid; i < G * kHD; i += kThreads) {
    const int hl = i / kHD, d = i % kHD, h = h0 + hl;
    const float* src = a.po + ((long)row * a.QH + h) * kHD + d;
    const long sstride = (long)a.M * a.QH * kHD;
    float num = 0.f;
#pragma unroll 8
    for (int s2 = 0; s2 < a.S; s2++)  // empty splits wrote no output
      if (s_w[hl][s2] != 0.f) num += s_w[hl][s2] * src[s2 * sstride];
    const float den = s_L[hl];
    a.out[row * a.so_row + (long)h * a.so_head + d] =
        __float2half(den > 0.f ? num / den : 0.f);
  }
  if (tid == 0) a.cnt[blockIdx.x] = 0;
}

}  // namespace qsa_dec

int64_t mono_qsa_decode_rdna2_splits(int64_t M, int64_t KVH, int64_t W) {
  int64_t s = 288 / std::max<int64_t>(1, M * KVH);
  s = std::min<int64_t>(std::max<int64_t>(s, 1), qsa_dec::kMaxS);
  // >= one chunk per wave per split
  const int64_t per = (int64_t)qsa_dec::kWaves * qsa_dec::kChunk;
  return std::min<int64_t>(s, std::max<int64_t>(1, (W + per - 1) / per));
}

// out[M, QH, HD] = sparse paged GQA attention of q over the selected tokens.
// scratch: fp32 >= splits * M * QH * (HD + 1); counters: int32 [>= M * KVH],
// zero-initialized once (the kernel leaves them zero).
void mono_qsa_decode_rdna2(const torch::Tensor& q, const torch::Tensor& k_cache,
                           const torch::Tensor& v_cache,
                           const torch::Tensor& indices,
                           const torch::Tensor& block_table,
                           const torch::Tensor& token_to_req,
                           torch::Tensor& out, torch::Tensor& scratch,
                           torch::Tensor& counters, int64_t splits) {
  using namespace qsa_dec;
  const int M = q.size(0), QH = q.size(1), KVH = k_cache.size(2);
  TORCH_CHECK(
      q.scalar_type() == at::kHalf && k_cache.scalar_type() == at::kHalf &&
          v_cache.scalar_type() == at::kHalf && out.scalar_type() == at::kHalf,
      "mono_qsa_decode_rdna2: fp16 q / k / v / out");
  TORCH_CHECK(q.size(2) == kHD && k_cache.size(3) == kHD && q.stride(2) == 1 &&
                  k_cache.stride(3) == 1 && v_cache.stride(3) == 1 &&
                  out.stride(2) == 1,
              "mono_qsa_decode_rdna2: head dim 256, unit last stride");
  TORCH_CHECK(QH % KVH == 0 && QH / KVH <= kMaxG,
              "mono_qsa_decode_rdna2: group <= 8");
  TORCH_CHECK(indices.scalar_type() == at::kInt && indices.size(0) == M &&
                  indices.stride(1) == 1,
              "mono_qsa_decode_rdna2: indices int32 [M, W]");
  TORCH_CHECK(
      block_table.scalar_type() == at::kInt && block_table.stride(1) == 1,
      "mono_qsa_decode_rdna2: block table int32");
  TORCH_CHECK(
      token_to_req.scalar_type() == at::kInt && token_to_req.numel() >= M,
      "mono_qsa_decode_rdna2: token_to_req int32 [M]");
  const int S = (int)splits;
  TORCH_CHECK(S >= 1 && S <= kMaxS, "mono_qsa_decode_rdna2: 1 <= splits <= 64");
  TORCH_CHECK(scratch.scalar_type() == at::kFloat &&
                  scratch.numel() >= (int64_t)S * M * QH * (kHD + 1),
              "mono_qsa_decode_rdna2: scratch too small");
  TORCH_CHECK(counters.scalar_type() == at::kInt &&
                  counters.numel() >= (int64_t)M * KVH,
              "mono_qsa_decode_rdna2: counters too small");
  // 16-byte K/V/Q rows
  TORCH_CHECK(k_cache.stride(1) % 8 == 0 && k_cache.stride(2) % 8 == 0 &&
                  k_cache.stride(0) % 8 == 0 && v_cache.stride(0) % 8 == 0 &&
                  v_cache.stride(1) % 8 == 0 && v_cache.stride(2) % 8 == 0 &&
                  q.stride(0) % 8 == 0 && q.stride(1) % 8 == 0,
              "mono_qsa_decode_rdna2: 16-byte aligned rows");
  if (M == 0) return;
  const at::cuda::OptionalCUDAGuard guard(q.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  Args a{};
  a.q = reinterpret_cast<const half*>(q.data_ptr());
  a.sq_row = q.stride(0);
  a.sq_head = q.stride(1);
  a.k = reinterpret_cast<const half*>(k_cache.data_ptr());
  a.v = reinterpret_cast<const half*>(v_cache.data_ptr());
  a.sk_blk = k_cache.stride(0);
  a.sk_tok = k_cache.stride(1);
  a.sk_head = k_cache.stride(2);
  a.sv_blk = v_cache.stride(0);
  a.sv_tok = v_cache.stride(1);
  a.sv_head = v_cache.stride(2);
  a.idx = indices.data_ptr<int>();
  a.sidx = indices.stride(0);
  a.btab = block_table.data_ptr<int>();
  a.sbt = block_table.stride(0);
  a.t2r = token_to_req.data_ptr<int>();
  a.out = reinterpret_cast<half*>(out.data_ptr());
  a.so_row = out.stride(0);
  a.so_head = out.stride(1);
  a.po = scratch.data_ptr<float>();
  a.plse = a.po + (int64_t)S * M * QH * kHD;
  a.cnt = counters.data_ptr<int>();
  a.M = M;
  a.QH = QH;
  a.KVH = KVH;
  a.G = QH / KVH;
  a.W = indices.size(1);
  a.PAGE = k_cache.size(1);
  a.PW = block_table.size(1);
  a.R = block_table.size(0);
  a.NB = k_cache.size(0);
  a.S = S;
  a.scale_log2 = (1.0f / sqrtf((float)kHD)) * 1.4426950408889634f;
  const dim3 grid(M * KVH, S);
  // clang-format off
  switch (a.G) {
#define QSA_DEC_CASE(g)                                                   \
  case g:                                                                 \
    hipLaunchKernelGGL(qsa_dec_k<g>, grid, dim3(kThreads), 0, stream, a); \
    break;
    QSA_DEC_CASE(1) QSA_DEC_CASE(2) QSA_DEC_CASE(3) QSA_DEC_CASE(4)
    QSA_DEC_CASE(5) QSA_DEC_CASE(6) QSA_DEC_CASE(7) QSA_DEC_CASE(8)
#undef QSA_DEC_CASE
    default:
      TORCH_CHECK(false, "mono_qsa_decode_rdna2: unsupported group");
  }
  // clang-format on
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------
// mono_qsa_store3_rdna2: the QSA indexer's three per-token cache stores
// (compressed keys at the compressed slots, raw keys and the int64 RoPE
// positions at the raw slots) in one launch. HIP replacement for the Triton
// _store_rows3_kernel in mono/rdna2/qsa.py; same masking (a slot is stored
// only when 0 <= slot < num_blocks * page, so PAD = -1 is skipped).
// Row mapping follows qsa_store_cache_rows_kernel (Waldecir Santos): every
// lane moves one 16-byte vector of a dense row, instead of one row per block.
// One lane group per token: the low half stores the compressed row (+ the
// RoPE row), the high half the raw row. Decode widths (<= 64 tokens) use
// 64-lane groups (more lanes per token, ~3.1 us at 1-24 tokens); wider
// batches use 32-lane groups (16 lanes per 16-vector row, 8 tokens per
// block).
// ---------------------------------------------------------------------------
namespace qsa_store3 {

constexpr int kThreads = 256;

struct Store {
  char* cache;
  const int64_t* slots;
  const char* rows;
  int64_t s_block, s_token, s_dim;  // cache strides, bytes
  int64_t r_row, r_dim;             // row strides, bytes
  int64_t limit;                    // num_blocks * page
  int page, width, esz;
  bool dense;  // both last dims contiguous and width*esz a multiple of 16
};

__device__ __forceinline__ void store_row(const Store& s, int64_t row,
                                          int lane, int lanes) {
  const int64_t slot = s.slots[row];
  if (slot < 0 || slot >= s.limit) return;
  char* dst = s.cache + (slot / s.page) * s.s_block + (slot % s.page) * s.s_token;
  const char* src = s.rows + row * s.r_row;
  if (s.dense) {
    const int vecs = s.width * s.esz / 16;
    for (int v = lane; v < vecs; v += lanes)
      *reinterpret_cast<uint4*>(dst + (int64_t)v * 16) =
          *reinterpret_cast<const uint4*>(src + (int64_t)v * 16);
    return;
  }
  for (int i = lane; i < s.width; i += lanes) {
    const char* e = src + (int64_t)i * s.r_dim;
    char* o = dst + (int64_t)i * s.s_dim;
    if (s.esz == 8)
      *reinterpret_cast<int64_t*>(o) = *reinterpret_cast<const int64_t*>(e);
    else if (s.esz == 4)
      *reinterpret_cast<int32_t*>(o) = *reinterpret_cast<const int32_t*>(e);
    else
      *reinterpret_cast<int16_t*>(o) = *reinterpret_cast<const int16_t*>(e);
  }
}

template <int GROUP>
__global__ void __launch_bounds__(kThreads)
    store3_k(Store c, Store r, Store p, int64_t num_rows) {
  constexpr int kHalf = GROUP / 2;
  const int64_t row =
      (int64_t)blockIdx.x * (kThreads / GROUP) + threadIdx.x / GROUP;
  if (row >= num_rows) return;
  const int lane = threadIdx.x % GROUP;
  if (lane < kHalf) {
    store_row(c, row, lane, kHalf);
    store_row(p, row, lane, kHalf);
  } else {
    store_row(r, row, lane - kHalf, kHalf);
  }
}

Store make(const torch::Tensor& cache, const torch::Tensor& slots,
           const torch::Tensor& rows, const char* what) {
  TORCH_CHECK(cache.dim() == 4 && cache.size(2) == 1,
              "mono_qsa_store3_rdna2: ", what, " cache must be [blocks, page, 1, W]");
  TORCH_CHECK(rows.dim() == 2 && rows.size(1) == cache.size(3),
              "mono_qsa_store3_rdna2: ", what, " rows must be [n, W]");
  TORCH_CHECK(slots.scalar_type() == torch::kInt64 && slots.dim() == 1 &&
                  slots.size(0) == rows.size(0),
              "mono_qsa_store3_rdna2: ", what, " slots must be int64 [n]");
  TORCH_CHECK(rows.scalar_type() == cache.scalar_type(),
              "mono_qsa_store3_rdna2: ", what, " dtype mismatch");
  Store s;
  s.esz = (int)cache.element_size();
  TORCH_CHECK(s.esz == 2 || s.esz == 4 || s.esz == 8,
              "mono_qsa_store3_rdna2: unsupported element size");
  s.cache = reinterpret_cast<char*>(cache.data_ptr());
  s.slots = slots.data_ptr<int64_t>();
  s.rows = reinterpret_cast<const char*>(rows.data_ptr());
  s.s_block = cache.stride(0) * s.esz;
  s.s_token = cache.stride(1) * s.esz;
  s.s_dim = cache.stride(3) * s.esz;
  s.r_row = rows.stride(0) * s.esz;
  s.r_dim = rows.stride(1) * s.esz;
  s.page = (int)cache.size(1);
  s.width = (int)cache.size(3);
  s.limit = cache.size(0) * cache.size(1);
  const auto aligned = [](const void* ptr, int64_t stride) {
    return (reinterpret_cast<uintptr_t>(ptr) % 16 == 0) && (stride % 16 == 0);
  };
  s.dense = s.s_dim == s.esz && s.r_dim == s.esz &&
            (s.width * s.esz) % 16 == 0 && aligned(s.rows, s.r_row) &&
            aligned(s.cache, s.s_token) && s.s_block % 16 == 0;
  return s;
}

}  // namespace qsa_store3

void mono_qsa_store3_rdna2(torch::Tensor& c_cache, const torch::Tensor& c_slots,
                           const torch::Tensor& c_rows, torch::Tensor& r_cache,
                           const torch::Tensor& r_slots,
                           const torch::Tensor& r_rows, torch::Tensor& p_cache,
                           const torch::Tensor& p_rows) {
  const int64_t n = c_rows.size(0);
  TORCH_CHECK(r_rows.size(0) == n && p_rows.size(0) == n,
              "mono_qsa_store3_rdna2: the three stores need the same row count");
  if (n == 0) return;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(c_cache));
  const auto c = qsa_store3::make(c_cache, c_slots, c_rows, "compressed");
  const auto r = qsa_store3::make(r_cache, r_slots, r_rows, "raw");
  const auto p = qsa_store3::make(p_cache, r_slots, p_rows, "rope");
  const auto stream = at::cuda::getCurrentCUDAStream();
  const auto launch = [&](auto kernel, int group) {
    const int per_block = qsa_store3::kThreads / group;
    const dim3 grid((unsigned)((n + per_block - 1) / per_block));
    hipLaunchKernelGGL(kernel, grid, dim3(qsa_store3::kThreads), 0, stream, c,
                       r, p, n);
  };
  if (n <= 64)
    launch(qsa_store3::store3_k<64>, 64);
  else
    launch(qsa_store3::store3_k<32>, 32);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------
// mono_qsa_k_store_rdna2: the QSA indexer's compressed-key epilogue fused with
// its three cache stores. GemmaRMSNorm (weight + norm_beta) + NeoX RoPE or
// interleaved 3-section MRoPE on the pooled compressed keys (same math as the
// key side of fused_qk_rmsnorm_rope_gate: the normed value is rounded to the
// input dtype before the rotation), written straight into the compressed key
// cache, plus the raw-key and RoPE-position rows at the raw slots. Replaces
// norm_rope + store_rows3 (two launches and an intermediate tensor) on decode.
// Tokens that close no compressed group (slot PAD = -1) skip the norm/RoPE.
// One 64-lane group per token: lanes 0-31 (one wave32) do the compressed key,
// lanes 32-63 the raw row and the RoPE row. head_dim 128, one head, fp16.
// ---------------------------------------------------------------------------
namespace qsa_kstore {

constexpr int kThreads = 256;
constexpr int kGroup = 64;
constexpr int kHD = 128;

struct Args {
  // compressed-key epilogue
  const half* pooled;  // [n, 128]
  int64_t pooled_row;
  const half* weight;  // [128]
  const half* cos_sin; // [max_pos, rotary_dim]
  int64_t cache_row;
  const char* pos;     // [n] or [3, n]
  int64_t pos_m, pos_t;
  int pos_esz;
  int has_mrope, sec_h, sec_w;
  int rotary_dim;
  float eps, beta;
  qsa_store3::Store c;  // destination: compressed key cache (rows unused)
  qsa_store3::Store r;  // raw keys
  qsa_store3::Store p;  // RoPE positions
  int64_t n;
};

__device__ __forceinline__ int64_t load_pos(const Args& a, int m, int64_t t) {
  const char* ptr = a.pos + m * a.pos_m + t * a.pos_t;
  return a.pos_esz == 8 ? *reinterpret_cast<const int64_t*>(ptr)
                        : (int64_t)*reinterpret_cast<const int32_t*>(ptr);
}

__global__ void __launch_bounds__(kThreads) kstore_k(Args a) {
  __shared__ float s_x[kThreads / kGroup][kHD];
  const int g = threadIdx.x / kGroup;
  const int64_t row = (int64_t)blockIdx.x * (kThreads / kGroup) + g;
  if (row >= a.n) return;
  const int lane = threadIdx.x % kGroup;
  if (lane >= 32) {
    qsa_store3::store_row(a.r, row, lane - 32, 32);
    qsa_store3::store_row(a.p, row, lane - 32, 32);
    return;
  }
  const int64_t slot = a.c.slots[row];
  if (slot < 0 || slot >= a.c.limit) return;  // whole wave: same row
  // RMSNorm: each lane holds elements lane + 32 * j.
  const half* x = a.pooled + row * a.pooled_row;
  float v[4];
  float ss = 0.0f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    v[j] = __half2float(x[lane + 32 * j]);
    ss += v[j] * v[j];
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor(ss, off, 32);
  const float inv_rms = rsqrtf(ss / kHD + a.eps);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int i = lane + 32 * j;
    const float w = __half2float(a.weight[i]) + a.beta;
    // round to the input dtype before the rotation, like the Triton kernel
    s_x[g][i] = __half2float(__float2half_rn(v[j] * inv_rms * w));
  }
  __builtin_amdgcn_wave_barrier();
  __builtin_amdgcn_s_waitcnt(0);
  half* dst = reinterpret_cast<half*>(a.c.cache + (slot / a.c.page) * a.c.s_block +
                                      (slot % a.c.page) * a.c.s_token);
  const int half_rot = a.rotary_dim / 2;
  const int64_t pt = load_pos(a, 0, row);
  const int64_t ph = a.has_mrope ? load_pos(a, 1, row) : pt;
  const int64_t pw = a.has_mrope ? load_pos(a, 2, row) : pt;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int i = lane + 32 * j;
    float out;
    if (i >= a.rotary_dim) {
      out = s_x[g][i];  // pass-through tail (partial rotary)
    } else {
      const int r = i < half_rot ? i : i - half_rot;
      int64_t pos = pt;
      if (a.has_mrope) {
        if ((r % 3 == 1) && (r < 3 * a.sec_h))
          pos = ph;
        else if ((r % 3 == 2) && (r < 3 * a.sec_w))
          pos = pw;
      }
      const half* cs = a.cos_sin + pos * a.cache_row;
      const float c = __half2float(cs[r]);
      const float s = __half2float(cs[half_rot + r]);
      const float x1 = s_x[g][r];
      const float x2 = s_x[g][half_rot + r];
      out = i < half_rot ? x1 * c - x2 * s : x2 * c + x1 * s;
    }
    dst[(int64_t)i * (a.c.s_dim / 2)] = __float2half_rn(out);
  }
}

}  // namespace qsa_kstore

void mono_qsa_k_store_rdna2(
    const torch::Tensor& pooled, const torch::Tensor& weight,
    const torch::Tensor& cos_sin_cache, const torch::Tensor& positions,
    double eps, double norm_beta, int64_t rotary_dim, int64_t mrope_h,
    int64_t mrope_w, torch::Tensor& c_cache, const torch::Tensor& c_slots,
    torch::Tensor& r_cache, const torch::Tensor& r_slots,
    const torch::Tensor& r_rows, torch::Tensor& p_cache,
    const torch::Tensor& p_rows) {
  const int64_t n = pooled.size(0);
  TORCH_CHECK(pooled.dim() == 2 && pooled.size(1) == qsa_kstore::kHD &&
                  pooled.scalar_type() == torch::kHalf && pooled.stride(1) == 1,
              "mono_qsa_k_store_rdna2: pooled must be fp16 [n, 128], dense rows");
  TORCH_CHECK(weight.numel() == qsa_kstore::kHD &&
                  weight.scalar_type() == torch::kHalf && weight.is_contiguous(),
              "mono_qsa_k_store_rdna2: weight must be fp16 [128]");
  TORCH_CHECK(cos_sin_cache.scalar_type() == torch::kHalf &&
                  cos_sin_cache.dim() == 2 && cos_sin_cache.stride(1) == 1 &&
                  cos_sin_cache.size(1) == rotary_dim,
              "mono_qsa_k_store_rdna2: cos_sin_cache must be fp16 [max_pos, rotary_dim]");
  TORCH_CHECK(rotary_dim > 0 && rotary_dim <= qsa_kstore::kHD && rotary_dim % 2 == 0,
              "mono_qsa_k_store_rdna2: rotary_dim must be even and <= 128");
  TORCH_CHECK(c_cache.scalar_type() == torch::kHalf,
              "mono_qsa_k_store_rdna2: compressed cache must be fp16");
  TORCH_CHECK(r_rows.size(0) == n && p_rows.size(0) == n && c_slots.numel() == n,
              "mono_qsa_k_store_rdna2: row counts must match");
  const bool mrope = positions.dim() == 2;
  TORCH_CHECK((mrope && positions.size(0) == 3 && positions.size(1) == n) ||
                  (!mrope && positions.dim() == 1 && positions.size(0) == n),
              "mono_qsa_k_store_rdna2: positions must be [n] or [3, n]");
  TORCH_CHECK(positions.scalar_type() == torch::kInt64 ||
                  positions.scalar_type() == torch::kInt32,
              "mono_qsa_k_store_rdna2: positions must be int32/int64");
  if (n == 0) return;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(pooled));
  qsa_kstore::Args a;
  a.pooled = reinterpret_cast<const half*>(pooled.data_ptr());
  a.pooled_row = pooled.stride(0);
  a.weight = reinterpret_cast<const half*>(weight.data_ptr());
  a.cos_sin = reinterpret_cast<const half*>(cos_sin_cache.data_ptr());
  a.cache_row = cos_sin_cache.stride(0);
  a.pos = reinterpret_cast<const char*>(positions.data_ptr());
  a.pos_esz = (int)positions.element_size();
  a.pos_m = mrope ? positions.stride(0) * a.pos_esz : 0;
  a.pos_t = positions.stride(mrope ? 1 : 0) * a.pos_esz;
  a.has_mrope = mrope ? 1 : 0;
  a.sec_h = (int)mrope_h;
  a.sec_w = (int)mrope_w;
  a.rotary_dim = (int)rotary_dim;
  a.eps = (float)eps;
  a.beta = (float)norm_beta;
  // The compressed Store only needs its slots/cache geometry; rows = pooled.
  a.c = qsa_store3::make(c_cache, c_slots, pooled, "compressed");
  a.r = qsa_store3::make(r_cache, r_slots, r_rows, "raw");
  a.p = qsa_store3::make(p_cache, r_slots, p_rows, "rope");
  a.n = n;
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int per_block = qsa_kstore::kThreads / qsa_kstore::kGroup;
  const dim3 grid((unsigned)((n + per_block - 1) / per_block));
  hipLaunchKernelGGL(qsa_kstore::kstore_k, grid, dim3(qsa_kstore::kThreads), 0,
                     stream, a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
