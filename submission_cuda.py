# okc: problem=muse_bf16_proj
"""Native CUDA candidate built once with PyTorch load_inline.

GEMV uses vector loads and warp reductions. GEMM uses BF16 WMMA with FP32
accumulation, weight-only tile packing, and a fused hi/lo router path.
Requires an NVIDIA GPU with compute capability >= 8.0.
"""

import torch
from torch.utils.cpp_extension import load_inline


_CPP = r"""
#include <torch/extension.h>
torch::Tensor project(torch::Tensor x, torch::Tensor raw,
                      torch::Tensor high, torch::Tensor low,
                      int64_t n, int64_t splits);
"""

_CUDA = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>

using bf16 = __nv_bfloat16;
namespace wmma = nvcuda::wmma;

template<bool Router, bool Partial>
__device__ __forceinline__ void write_result(
        void* output, float* partial, int index, float value) {
    if constexpr (Partial) {
        partial[index] = value;
    } else if constexpr (Router) {
        static_cast<float*>(output)[index] = value;
    } else {
        static_cast<bf16*>(output)[index] = __float2bfloat16_rn(value);
    }
}

template<bool Router, bool Partial>
__global__ void gemv(const bf16* __restrict__ x, const void* __restrict__ weights,
                     void* output, float* partial, int n, int k, int splits) {
    // Each warp handles one output. Pair loads cover a whole 128-byte BF16
    // segment across the warp, with no intermediate BF16 arithmetic.
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + threadIdx.x / 32;
    if (row >= n) return;
    const int split = blockIdx.y;
    const int begin = split * (k / splits) / 2;
    const int end = (split + 1) * (k / splits) / 2;
    const auto* xp = reinterpret_cast<const __nv_bfloat162*>(x);
    float acc = 0.0f;
    for (int pair = begin + lane; pair < end; pair += 32) {
        const float2 xv = __bfloat1622float2(xp[pair]);
        float2 wv;
        if constexpr (Router) {
            wv = reinterpret_cast<const float2*>(weights)[row * (k / 2) + pair];
        } else {
            wv = __bfloat1622float2(
                reinterpret_cast<const __nv_bfloat162*>(weights)[row * (k / 2) + pair]);
        }
        acc = fmaf(xv.x, wv.x, acc);
        acc = fmaf(xv.y, wv.y, acc);
    }
    #pragma unroll
    for (int delta = 16; delta > 0; delta /= 2)
        acc += __shfl_down_sync(0xffffffff, acc, delta);
    if (lane == 0)
        write_result<Router, Partial>(output, partial, split * n + row, acc);
}

template<int BM, bool Router, bool Partial>
__global__ void gemm(const bf16* __restrict__ x,
                     const bf16* __restrict__ high,
                     const bf16* __restrict__ low,
                     void* output, float* partial,
                     int m, int n, int k, int splits) {
    constexpr int BK = 64;
    constexpr int AStride = BK + 16;
    constexpr int Warps = (BM / 16) * 4;
    __shared__ __align__(32) bf16 a_shared[BM * AStride];
    __shared__ __align__(32) float results[Warps][16 * 16];
    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid & 31;
    const int warp_m = warp / 4;
    const int warp_n = warp % 4;
    const int row0 = blockIdx.y * BM;
    const int col0 = blockIdx.x * 64 + warp_n * 16;
    const int split = blockIdx.z;
    const int chunks = k / BK;
    const int begin = split * (chunks / splits);
    const int end = (split + 1) * (chunks / splits);

    wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_low;
    wmma::fill_fragment(acc, 0.0f);
    if constexpr (Router) wmma::fill_fragment(acc_low, 0.0f);

    for (int chunk = begin; chunk < end; ++chunk) {
        for (int i = tid; i < BM * BK; i += Warps * 32) {
            const int row = i / BK;
            const int kk = i % BK;
            a_shared[row * AStride + kk] = row0 + row < m
                ? x[(row0 + row) * k + chunk * BK + kk]
                : __float2bfloat16_rn(0.0f);
        }
        __syncthreads();
        // Packed weights: [N/16, K/64, 16, 64]. Each warp loads
        // a contiguous 16x64 weight tile, viewed as column-major B.
        const int wbase = ((col0 / 16) * chunks + chunk) * (16 * BK);
        #pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            wmma::load_matrix_sync(a, a_shared + warp_m * 16 * AStride + kk, AStride);
            wmma::load_matrix_sync(b, high + wbase + kk, BK);
            wmma::mma_sync(acc, a, b, acc);
            if constexpr (Router) {
                wmma::load_matrix_sync(b, low + wbase + kk, BK);
                wmma::mma_sync(acc_low, a, b, acc_low);
            }
        }
        __syncthreads();
    }
    if constexpr (Router) {
        #pragma unroll
        for (int i = 0; i < acc.num_elements; ++i)
            acc.x[i] += acc_low.x[i];
    }
    wmma::store_matrix_sync(results[warp], acc, 16, wmma::mem_row_major);
    __syncwarp();
    for (int i = lane; i < 16 * 16; i += 32) {
        const int row = row0 + warp_m * 16 + i / 16;
        const int col = col0 + i % 16;
        if (row < m && col < n)
            write_result<Router, Partial>(
                output, partial, (split * m + row) * n + col, results[warp][i]);
    }
}

template<bool Router>
__global__ void reduce(const float* __restrict__ partial, void* output,
                       int total, int splits) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    float acc = 0.0f;
    for (int s = 0; s < splits; ++s) acc += partial[s * total + i];
    write_result<Router, false>(output, nullptr, i, acc);
}

template<bool Router, bool Partial>
void launch(const bf16* x, const void* raw, const bf16* high, const bf16* low,
            void* out, float* partial, int m, int n, int k, int splits,
            cudaStream_t stream) {
    if (m == 1) {
        gemv<Router, Partial><<<dim3((n + 7) / 8, splits), 256, 0, stream>>>(
            x, raw, out, partial, n, k, splits);
    } else if (m <= 16) {
        gemm<16, Router, Partial><<<dim3(n / 64, (m + 15) / 16, splits), 128, 0, stream>>>(
            x, high, low, out, partial, m, n, k, splits);
    } else {
        gemm<32, Router, Partial><<<dim3(n / 64, (m + 31) / 32, splits), 256, 0, stream>>>(
            x, high, low, out, partial, m, n, k, splits);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor project(torch::Tensor x, torch::Tensor raw,
                      torch::Tensor high, torch::Tensor low,
                      int64_t n64, int64_t splits64) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dim() == 2
                && x.scalar_type() == at::kBFloat16, "expected contiguous CUDA BF16 x");
    TORCH_CHECK(raw.is_cuda() && high.is_cuda() && low.is_cuda()
                && raw.device() == x.device() && high.device() == x.device()
                && low.device() == x.device(), "weights must be on the input device");
    const c10::cuda::CUDAGuard guard(x.device());
    const int m = x.size(0), n = n64, k = x.size(1), splits = splits64;
    const bool router = raw.scalar_type() == at::kFloat;
    TORCH_CHECK(m > 0 && n > 0 && n % 64 == 0 && k % 64 == 0
                && splits > 0 && k % (64 * splits) == 0,
                "unsupported matrix or split-K dimensions");
    TORCH_CHECK(raw.dim() == 2 && raw.size(0) >= n && raw.size(1) == k
                && raw.is_contiguous()
                && (router || raw.scalar_type() == at::kBFloat16), "invalid raw weights");
    TORCH_CHECK(high.is_contiguous() && low.is_contiguous()
                && high.scalar_type() == at::kBFloat16 && low.scalar_type() == at::kBFloat16
                && high.numel() >= n64 * k && low.numel() >= n64 * k,
                "invalid packed weights");
    auto out = torch::empty({m, n}, x.options().dtype(router ? at::kFloat : at::kBFloat16));
    torch::Tensor partial;
    float* pp = nullptr;
    if (splits > 1) {
        partial = torch::empty({splits, m, n}, x.options().dtype(at::kFloat));
        pp = partial.data_ptr<float>();
    }
    const auto* xp = reinterpret_cast<const bf16*>(x.data_ptr<at::BFloat16>());
    const auto* hp = reinterpret_cast<const bf16*>(high.data_ptr<at::BFloat16>());
    const auto* lp = reinterpret_cast<const bf16*>(low.data_ptr<at::BFloat16>());
    void* op = out.data_ptr();
    const auto stream = c10::cuda::getCurrentCUDAStream(x.get_device()).stream();
    if (router) {
        if (splits > 1) launch<true, true>(xp, raw.data_ptr(), hp, lp, op, pp, m, n, k, splits, stream);
        else launch<true, false>(xp, raw.data_ptr(), hp, lp, op, pp, m, n, k, splits, stream);
    } else {
        if (splits > 1) launch<false, true>(xp, raw.data_ptr(), hp, lp, op, pp, m, n, k, splits, stream);
        else launch<false, false>(xp, raw.data_ptr(), hp, lp, op, pp, m, n, k, splits, stream);
    }
    if (splits > 1) {
        if (router) reduce<true><<<(m * n + 255) / 256, 256, 0, stream>>>(pp, op, m * n, splits);
        else reduce<false><<<(m * n + 255) / 256, 256, 0, stream>>>(pp, op, m * n, splits);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return out;
}
"""


def _pack(w):
    n, k = w.shape
    return w.reshape(n // 16, 16, k // 64, 64).permute(0, 2, 1, 3).contiguous()


def _splits(tiles, k, limit=32):
    splits = 1
    while tiles * splits < 256 and splits < limit and k % (splits * 2 * 256) == 0:
        splits *= 2
    return splits


def prepare(weights):
    # load_inline selects the local GPU architecture, so this also builds on
    # Ampere/Ada Colab runtimes rather than emitting H200-only machine code.
    module = load_inline(
        name="muse_bf16_proj_native_v1", cpp_sources=_CPP, cuda_sources=_CUDA,
        functions=["project"], extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "-std=c++17"], with_cuda=True, verbose=False)
    raw = {name: w.contiguous() for name, w in weights.items()}
    high, low = {}, {}
    for name, w in raw.items():
        if name == "router":
            hi = w.to(torch.bfloat16)
            lo = (w - hi.float()).to(torch.bfloat16)
            high[name], low[name] = _pack(hi), _pack(lo)
        else:
            high[name] = _pack(w)
            low[name] = high[name]
    return {"module": module, "raw": raw, "high": high, "low": low}


def kernel_fn(data, state):
    x, config = data["x"], data["config"]
    m, n, k = config["m"], config["n"], config["k"]
    layer = config["layer"]
    if m == 1:
        splits = _splits((n + 7) // 8, k, limit=8)
    else:
        bm = 16 if m <= 16 else 32
        splits = _splits(((m + bm - 1) // bm) * (n // 64), k)
    return state["module"].project(
        x, state["raw"][layer], state["high"][layer], state["low"][layer], n, splits)
