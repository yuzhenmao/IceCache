from dataclasses import dataclass
from typing import List

import torch

from . import utils


@dataclass
class PagePool:
    n_max_pages: int
    page_shape: List[int]
    dtype: torch.dtype
    device: torch.device

    def __post_init__(self):
        self.buf_shape = (self.n_max_pages, *self.page_shape)
        self.buffer = torch.zeros(
            self.buf_shape,
            dtype=self.dtype,
            device=self.device,
            pin_memory=(self.device.type == "cpu"),
        )
        self._free_ids = set(range(self.n_max_pages))

    def __getitem__(self, idx):
        return self.buffer[idx]

    @property
    def n_free_pages(self):
        return len(self._free_ids)

    def alloc_page(self):
        return self._free_ids.pop()
    
    def alloc_contiguous_pages(self, num):
        if num <= 0:
            return []
        
        if len(self._free_ids) < num:
            return None  # Not enough free IDs
    
        sorted_ids = sorted(self._free_ids)
        
        for i in range(len(sorted_ids) - num + 1):
            # Check if we have num contiguous IDs starting at sorted_ids[i]
            if sorted_ids[i + num - 1] - sorted_ids[i] == num - 1:
                result = list(range(sorted_ids[i], sorted_ids[i] + num))
                # Remove the allocated IDs from free set
                self._free_ids -= set(result)
                return result
        
        return None  # No contiguous block found

    def free_page(self, page_id):
        assert 0 <= page_id < self.n_max_pages
        assert page_id not in self._free_ids
        self._free_ids.add(page_id)

    def clear(self):
        self._free_ids = set(range(self.n_max_pages))


class KvPool(PagePool):
    def __init__(
        self,
        n_max_pages,
        page_size,
        n_kv_heads,
        head_dim,
        dtype: torch.dtype,
        device: torch.device,
        layout_map: tuple = (0, 1, 2, 3),
    ):
        self._layout_map = layout_map
        self._orig_shape = (2, page_size, n_kv_heads, head_dim)
        # (0, 2, 1, 3)
        # (2, n_kv_heads, page_size, head_dim)
        self.page_size = page_size
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        super().__init__(
            n_max_pages,
            tuple(self._orig_shape[i] for i in layout_map),
            dtype,
            device,
        )


@dataclass
class KvCache:
    pool: KvPool
    batch_size: int
    budget: int = None
    n_sink_pages: int = 2
    n_win_pages: int = 2
    n_groups: int = 1
    offload_ratio: int = 2

    def __post_init__(self):
        self.seq_len = 0
        self._i32 = dict(dtype=torch.int32, device=self.device)
        self._fp = dict(dtype=self.dtype, device=self.device)

        self.n_final_win_pages = self.n_win_pages

        # (gpu/cpu)_cache_page_id -> (gpu/cpu)_pool_page_id
        self.c2p = torch.empty([self.batch_size, 0], **self._i32)
        if self.budget is not None:
            self.n_win_pages = self.budget - self.n_sink_pages
            # # gpu_cache_page_id -> cpu_cache_page_id
            # self.gc2cc = torch.empty([self.batch_size, self.n_groups, 0], **self._i32)
            self.gc2cc = torch.full([self.batch_size, self.n_kv_heads, self.budget], -1, **self._i32)

            # # cpu_cache_page_id -> gpu_pool_page_id or -1
            self.cc2gp = None
            self.ccc = None

            self._evict_cnt = 0
            self.evict_idx = -2
            assert self.n_win_pages >= 2

    @property
    def buffer(self):
        return self.pool.buffer

    def __getitem__(self, idx):
        return self.pool[self.c2p[idx]]

    @property
    def dtype(self):
        return self.pool.dtype

    @property
    def device(self):
        return self.pool.device

    @property
    def page_size(self):
        return self.pool.page_size

    @property
    def n_kv_heads(self):
        return self.pool.n_kv_heads

    @property
    def head_dim(self):
        return self.pool.head_dim

    @property
    def n_pages(self):
        return (self.seq_len + self.page_size - 1) // self.page_size

    @property
    def n_real_pages(self):
        return self.c2p.shape[-1]
    
    @property
    def next_evict_idx(self):
        return self.budget - self.n_win_pages + self._evict_cnt % (self.n_win_pages - 1) if self.n_win_pages > 1 else self.n_sink_pages + self._evict_cnt % (self.budget - self.n_sink_pages - 1)

    @property
    def last_page_len(self):
        return (self.seq_len - 1) % self.page_size + 1

    def _decode_alloc_1_page(self, alloc_page=None):
        alloc_page = alloc_page or self.pool.alloc_page
        if self.budget is None or self.n_real_pages + 1 <= self.budget:
            self.c2p = utils.cat(
                self.c2p,
                torch.tensor(
                    [[alloc_page()] for _ in range(self.batch_size)], **self._i32
                ),
                dim=-1,
            )
        else:
            if self.n_final_win_pages == self.n_win_pages:
                if self.n_win_pages > 1:
                    e_gci = (
                        self.budget
                        - self.n_win_pages
                        + self._evict_cnt % (self.n_win_pages - 1)
                    )
                else:
                    e_gci = self.n_sink_pages + self._evict_cnt % (
                        self.budget - self.n_sink_pages - 1
                    )
                self._evict_cnt += 1
                self.evict_idx = e_gci

                # evict page with gpu_cache_page_id == e_gci, cpu_cache_id == e_cci
                # add page with gpu_cache_page_id == budget - 1, cpu_cache_page_id == n_pages - 1
                for i in range(self.batch_size):
                    self[i, e_gci].copy_(self[i, -1], non_blocking=True)
            else:
                if self.n_win_pages - self.offload_ratio + 1 >= self.n_final_win_pages:
                    self.n_win_pages = self.n_win_pages - self.offload_ratio + 1
                else:
                    self.n_win_pages = self.n_final_win_pages
                
                for i in range(self.batch_size):
                    self[i, -self.n_win_pages:-1].copy_(self[i, -self.n_win_pages+1:], non_blocking=True)      

    def decode_alloc_1_token(self, alloc_page=None):
        old_n_pages = self.n_pages
        self.seq_len += 1
        n_new_pages = self.n_pages - old_n_pages
        if n_new_pages > 0:
            self._decode_alloc_1_page(alloc_page=alloc_page)
        return n_new_pages
    
    def decode_alloc_n_tokens(self, n, alloc_page=None):
        old_n_pages = self.n_pages
        self.seq_len += n
        n_new_pages = self.n_pages - old_n_pages
        if n_new_pages > 0:
            [self._decode_alloc_1_page(alloc_page=alloc_page) for _ in range(n_new_pages)]
        return n_new_pages

    def _prefill_alloc_n_pages(self, n, alloc_page=None):
        if n == 1:
            alloc_page = alloc_page or self.pool.alloc_page
            self.c2p = torch.tensor(
                [alloc_page() for _ in range(self.batch_size * n)], **self._i32
            ).reshape(self.batch_size, n)
        elif n > 1:
            pages = self.pool.alloc_contiguous_pages(n)
            if pages is None:
                raise RuntimeError("Not enough contiguous free pages in pool")
            self.c2p = torch.tensor(
                [pages for _ in range(self.batch_size)], **self._i32
            )

    def prefill_alloc_n_tokens(self, n, alloc_page=None):
        old_n_pages = self.n_pages
        self.seq_len += n
        n_new_pages = self.n_pages - old_n_pages
        if n_new_pages > 0:
            self._prefill_alloc_n_pages(n_new_pages, alloc_page=alloc_page)
        return n_new_pages

    def clear(self):
        for page_id in self.c2p.reshape(-1).tolist():
            self.pool.free_page(page_id)
        self.__post_init__()
