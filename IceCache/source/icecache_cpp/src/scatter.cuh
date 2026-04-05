/******************************************************************************
 * Copyright (c) 2025, Yuzhen Mao, Qitong Wang and Ke Li.
 ******************************************************************************/

#include <cuda_runtime.h>
#include <torch/extension.h>

void scatter_pages(torch::Tensor transit_buffer,  // [1, 2 * 32 * 4 = 256,
                                                  // 16 * 128 = 2048]
                   torch::Tensor kvc_buffer,      // [81920, 2, 32, 16, 128]
                   torch::Tensor evict_ids,       // [32, 4]
                   torch::Tensor n_evicts         // [32]
                   //  int64_t stream_ptr
);