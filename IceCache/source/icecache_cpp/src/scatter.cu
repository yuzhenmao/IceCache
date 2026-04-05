/******************************************************************************
 * Copyright (c) 2025, Yuzhen Mao, Qitong Wang and Ke Li.
 ******************************************************************************/

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Half.h>
#include <cuda_fp16.h>

#include "pytorch_extension_utils.h"
#include "scatter.cuh"

// TODO: move to template
// template <int BlockDim = 256, int SharedMemBytes = 128>
// void scatter_pages(...) {}
#define BlockDim 256
#define SharedMemBytes 512

// #define DEBUG

__global__ void _scatter_kernel(
    const __half *__restrict__ transit_buffer,       // [1, 2 * 32 * 4 = 256,
                                                     // 16 * 128 = 2048]
    __half *__restrict__ kvc_buffer_ptr,             // [81920, 2, 32, 16, 128]
    const int32_t *__restrict__ evict_page_ids_ptr,  // [32, 4]
    const int64_t *__restrict__ transit_page_offsets_ptr,  // [32]
    int n_heads, int n_max_head_pages, int page_dim) {
  const size_t bid = blockIdx.x;
  const size_t tid = threadIdx.x;

  __shared__ int32_t evict_page_id;
  __shared__ int64_t transit_page_offset;
  extern __shared__ int64_t transit_page_offsets[];

  int n_total_pages = n_heads * n_max_head_pages;
  int k_or_v = bid / n_total_pages;
  int evict_page_i = bid % n_total_pages;
  int head_id = evict_page_i / n_max_head_pages;
  int head_page_i = evict_page_i % n_max_head_pages;

  // #ifdef DEBUG
  //   if (tid == 0) {
  //     printf("cu.sc.kernel: bid=%ld, evict_page_i=%d, head_id=%d\n", bid,
  //            evict_page_i, head_id);
  //   }
  //   __syncthreads();
  // #endif

  if (tid == 0) {
    evict_page_id = evict_page_ids_ptr[evict_page_i];
  }
  __syncthreads();

  if (evict_page_id < 0) {
    return;
  }

  transit_page_offsets[0] = 0;
  if (tid < n_heads) {
    transit_page_offsets[tid + 1] = transit_page_offsets_ptr[tid];
  }
  __syncthreads();

  if (tid == 0) {
    // #ifdef DEBUG
    //   for (int i_head = 0; i_head < n_heads; ++i_head) {
    //     printf("cu.scatter: n_evicts_ptr[%d] = %ld\n", i_head,
    //            n_evicts_ptr[i_head]);
    //   }
    // #endif

    for (int i_head = 1; i_head <= n_heads; ++i_head) {
      transit_page_offsets[i_head] += transit_page_offsets[i_head - 1];
    }

    // #ifdef DEBUG
    //     printf("cu.scatter: n_total_evict_pages = %ld\n",
    //     n_total_evict_pages);
    // #endif

#ifdef DEBUG
    for (int i_head = 0; i_head <= n_heads; ++i_head) {
      printf("cu.sc.kernel: bid=%ld, transit_page_offsets[%d] = %ld\n", bid,
             i_head, transit_page_offsets[i_head]);
    }
#endif
  }
  __syncthreads();

  transit_page_offset = transit_page_offsets[head_id] + head_page_i;

  int64_t kvc_offset =
      static_cast<int64_t>((evict_page_id * 2 + k_or_v) * n_heads + head_id) *
      page_dim;
  int64_t transit_offset =
      static_cast<int64_t>(k_or_v * transit_page_offsets[n_heads] +
                           transit_page_offset) *
      page_dim;

#ifdef DEBUG
  if (tid == 0) {
    printf("cu.sc.kernel: bid=%ld, transit_page_offsets[n_heads]=%ld\n", bid,
           transit_page_offsets[n_heads]);
    printf("cu.sc.kernel: bid=%ld, evict_page_id=%d, kvc_offset=%d\n", bid,
           evict_page_id, kvc_offset);
    printf(
        "cu.sc.kernel: bid=%ld, head_id=%d, head_page_i=%d, "
        "transit_page_offset=%ld, transit_offset=%d\n",
        bid, head_id, head_page_i, transit_page_offset, transit_offset);
  }
  __syncthreads();
#endif

  // reinterpret_cast<float4 *>(kvc_buffer_ptr + kvc_offset)[tid] =
  //     reinterpret_cast<const float4 *>(transit_buffer + transit_offset)[tid];
  for (int32_t i_float4 = tid; i_float4 < page_dim / 8; i_float4 += BlockDim) {
    reinterpret_cast<float4 *>(kvc_buffer_ptr + kvc_offset)[i_float4] =
        reinterpret_cast<const float4 *>(transit_buffer +
                                         transit_offset)[i_float4];
  }
}

void scatter_pages(torch::Tensor transit_buffer,  // [1, 2 * 32 * 4 = 256,
                                                  // 16 * 128 = 2048]
                   torch::Tensor kvc_buffer,      // [81920, 2, 32, 16, 128]
                   torch::Tensor evict_page_ids,  // [32, 4]
                   torch::Tensor n_evicts         // [32]
                                                  //  int64_t stream_ptr
) {
  CHECK_INPUT(transit_buffer);
  CHECK_INPUT(kvc_buffer);
  CHECK_INPUT(evict_page_ids);
  CHECK_INPUT(n_evicts);

  CHECK_DIM(3, transit_buffer);
  CHECK_DIM(5, kvc_buffer);
  CHECK_DIM(2, evict_page_ids);
  CHECK_DIM(1, n_evicts);

#ifdef DEBUG
  printf("cu.scatter: transit_buffer.shape = [%d, %d, %d]\n",
         transit_buffer.size(0), transit_buffer.size(1),
         transit_buffer.size(2));
  printf("cu.scatter: kvc_buffer.shape = [%d, %d, %d, %d, %d]\n",
         kvc_buffer.size(0), kvc_buffer.size(1), kvc_buffer.size(2),
         kvc_buffer.size(3), kvc_buffer.size(4));
  printf("cu.scatter: evict_page_ids.shape = [%d, %d]\n",
         evict_page_ids.size(0), evict_page_ids.size(1));
  printf("cu.scatter: n_evicts.shape = [%d]\n", n_evicts.size(0));
#endif

  CHECK_EQ(transit_buffer.scalar_type(), torch::kFloat16);
  CHECK_EQ(kvc_buffer.scalar_type(), torch::kFloat16);
  CHECK_EQ(evict_page_ids.scalar_type(), torch::kInt32);
  CHECK_EQ(n_evicts.scalar_type(), torch::kInt64);

  int n_heads = evict_page_ids.size(0);
  CHECK_EQ(n_heads, n_evicts.size(0));

  int n_max_head_pages = evict_page_ids.size(1);

  CHECK_EQ(n_heads, kvc_buffer.size(2));
  int page_size = kvc_buffer.size(3);
  int head_dim = kvc_buffer.size(4);

  int page_dim = page_size * head_dim;
  CHECK_EQ(page_dim, transit_buffer.size(2));
  // CHECK_EQ(page_dim, BlockDim * 8);

#ifdef DEBUG
  printf("cu.scatter: n_heads = %d\n", n_heads);
  printf("cu.scatter: n_max_head_pages = %d\n", n_max_head_pages);
  printf("cu.scatter: page_size = %d\n", page_size);
  printf("cu.scatter: head_dim = %d\n", head_dim);
  printf("cu.scatter: page_dim = %d\n", page_dim);
#endif

  __half *transit_buffer_ptr =
      reinterpret_cast<__half *>(transit_buffer.data_ptr<at::Half>());
  __half *kvc_buffer_ptr =
      reinterpret_cast<__half *>(kvc_buffer.data_ptr<at::Half>());
  int32_t *evict_page_ids_ptr = evict_page_ids.data_ptr<int32_t>();
  int64_t *n_evicts_ptr = n_evicts.data_ptr<int64_t>();

  // #ifdef DEBUG
  //   const size_t bid = 137;
  //   const size_t tid = 15;
  //   printf("cu.scatter: bid = %d, tid = %d\n", bid, tid);

  //   int n_total_pages = n_heads * n_max_head_pages;
  //   int k_or_v = bid / n_total_pages;
  //   int evict_page_i = bid % n_total_pages;
  //   int head_id = evict_page_i / n_max_head_pages;
  //   int head_page_i = evict_page_i % n_max_head_pages;
  //   printf("cu.scatter: n_total_pages = %d\n", n_total_pages);
  //   printf("cu.scatter: k_or_v = %d\n", k_or_v);
  //   printf("cu.scatter: evict_page_i = %d\n", evict_page_i);
  //   printf("cu.scatter: head_id = %d\n", head_id);
  //   printf("cu.scatter: head_page_i = %d\n", head_page_i);

  //   std::vector<int32_t> cpu_evict_page_ids(n_total_pages);
  //   cudaMemcpy(cpu_evict_page_ids.data(), evict_page_ids_ptr,
  //              n_total_pages * sizeof(int32_t), cudaMemcpyDeviceToHost);

  //   // TODO: update
  //   std::vector<int64_t> cpu_n_evicts(n_heads);
  //   cudaMemcpy(cpu_n_evicts.data(), n_evicts_ptr, n_heads * sizeof(int64_t),
  //              cudaMemcpyDeviceToHost);

  //   cudaError_t error = cudaGetLastError();
  //   if (error != cudaSuccess) {
  //     printf("CUDA error: %s\n", cudaGetErrorString(error));
  //   }

  //   int32_t evict_page_id = cpu_evict_page_ids[evict_page_i];
  //   int64_t transit_page_offset = n_evicts_ptr[head_id] + head_page_i;
  //   printf("cu.scatter: evict_page_id = %d\n", evict_page_id);
  //   printf("cu.scatter: transit_page_offset = %d\n", transit_page_offset);

  //   int transit_offset =
  //       (k_or_v * n_total_evict_pages + transit_page_offset) * page_dim;
  //   int kvc_offset =
  //       ((evict_page_id * 2 + k_or_v) * n_heads + head_id) * page_dim;
  //   printf("cu.scatter: transit_offset = %d\n", transit_offset / page_dim);
  //   printf("cu.scatter: kvc_offset = %d\n", kvc_offset / page_dim);
  // #endif

  int GridDim = 2 * n_heads * n_max_head_pages;
#ifdef DEBUG
  printf("cu.scatter: GridDim = %d\n", GridDim);
#endif

  //   auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  //   auto device_index = kvc_buffer.device().index();
  //   at::cuda::CUDAStream cuda_stream =
  //       c10::cuda::getStreamFromExternal(stream, device_index);
  // #ifdef DEBUG
  //   // at::cuda::CUDAStream s = at::cuda::getCurrentCUDAStream();
  //   std::printf("scatter: device=%d id=%d handle=%p\n",
  //               cuda_stream.device_index(), cuda_stream.id(),
  //               (void *)cuda_stream.stream());
  // #endif

  // _scatter_kernel<<<GridDim, BlockDim, SharedMemBytes,
  // cuda_stream.stream()>>>(
  _scatter_kernel<<<GridDim, BlockDim, SharedMemBytes>>>(
      transit_buffer_ptr, kvc_buffer_ptr, evict_page_ids_ptr, n_evicts_ptr,
      n_heads, n_max_head_pages, page_dim);
}