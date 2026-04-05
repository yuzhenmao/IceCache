import warnings
from typing import List, Optional, Tuple, Union, Dict
from collections import defaultdict
from functools import wraps
import asyncio


import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from transformers.cache_utils import Cache

# use these classes just for hint
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaForCausalLM,
    LlamaRMSNorm,
    LlamaDecoderLayer,
    LlamaModel,
    repeat_kv,
)

from icecache.infer_state import InferState
from icecache import kernels
from time import time
import numpy as np

global tt

class NotTestedError(NotImplementedError):
    """Exception raised when a specific feature is not tested."""
    pass

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    if k is None:
        return q_embed, None
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def _icecache_prefill(
    self: LlamaAttention,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    infer_state: InferState = None,
    debug: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()
    cur_id: int = self.layer_idx
    state = infer_state
    n_layers = state.n_layers

    if cur_id == 0:
        global tt
        tt = time()
        state._prepare_prefill(bsz, q_len)

    if hasattr(self.config, 'pretraining_tp') and self.config.pretraining_tp > 1:
        key_value_slicing = (
            self.config.num_key_value_heads * self.head_dim
        ) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.config.num_attention_heads * self.head_dim) // self.config.pretraining_tp, dim=0
        )
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [
            F.linear(hidden_states, query_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [
            F.linear(hidden_states, key_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [
            F.linear(hidden_states, value_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        value_states = torch.cat(value_states, dim=-1)
    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    if "qwen3" in self.config._name_or_path.lower():
        query_states = self.q_norm(query_states.view(bsz, q_len, self.config.num_attention_heads, self.head_dim)).transpose(1, 2)
        key_states = self.k_norm(key_states.view(bsz, q_len, self.config.num_key_value_heads, self.head_dim)).transpose(1, 2)
    else:
        query_states = query_states.view(bsz, q_len, self.config.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)

    value_states = value_states.view(
        bsz, q_len, self.config.num_key_value_heads, self.head_dim
    )

    kvc = state.kv_caches[cur_id]

    cos, sin = position_embeddings

    # Q: should be updated in any cases
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    query_states = query_states.transpose(1, 2).contiguous()
    key_states = key_states.transpose(1, 2).contiguous()

    # TODO: key embedding projection (overlap with append_paged_kv_cache)
    projected = None
    # if q_len > 1:  # prefill
    #     assert do_send_pf == do_recv_pf == do_reuse_pf == False
    if state.layer2budget[cur_id] is not None:
        start = state.n_sink_pages * state.page_size
        end = (state.n_kv_pages - state.n_win_pages) * state.page_size
        key = key_states.reshape(state.n_kv_heads, -1, state.head_dim)[:, start:end, :]
        projected = torch.matmul(key, state.proj_vec[:-1]).reshape(-1, 1)

    state.attn_layers[cur_id] = self
    kvc.prefill_alloc_n_tokens(q_len, state.alloc_page)

    state.append_paged_kv_cache(cur_id, key_states, value_states)

    if state._dci_future is not None:
        _ = state._dci_future.result() # (timeout=1)
        state._dci_future = None

    # assert state._dci_future is None
    state._dci_future = asyncio.run_coroutine_threadsafe(
        state.prefill_evict_extra_pages_wrapper(
            cur_id, query_states[:, -1:, ...].contiguous(), projected), 
        state._loop)

    attn_output = state.prefill_sdpa(cur_id, query_states)

    attn_output = attn_output.reshape(bsz, q_len, -1)

    if 'llama' in self.config._name_or_path and self.config.pretraining_tp > 1:
        attn_output = attn_output.split(
            (self.config.head_dim * self.config.num_attention_heads) // self.config.pretraining_tp, dim=2
        )
        o_proj_slices = self.o_proj.weight.split(
            (self.config.head_dim * self.config.num_attention_heads) // self.config.pretraining_tp, dim=1
        )
        attn_output = sum(
            [
                F.linear(attn_output[i], o_proj_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
        )
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    if cur_id == n_layers - 1:
        # state.end_forward(bsz, q_len)

        if state._dci_future is not None:
            _ = state._dci_future.result() # (timeout=1)
            state._dci_future = None

        state._finish_prefill(bsz, q_len)

    return attn_output, attn_weights


def _icecache_decode(
    self: LlamaAttention,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    infer_state: InferState = None,
    debug: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()
    cur_id: int = self.layer_idx
    state = infer_state
    n_layers = state.n_layers

    if cur_id == 0:
        global tt
        tt = time()
        # state.begin_forward(bsz, q_len)
        state._prepare_decode(bsz)

    if hasattr(self.config, 'pretraining_tp') and self.config.pretraining_tp > 1:
        key_value_slicing = (
            self.config.num_key_value_heads * self.head_dim
        ) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.config.num_attention_heads * self.head_dim) // self.config.pretraining_tp, dim=0
        )
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [
            F.linear(hidden_states, query_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [
            F.linear(hidden_states, key_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [
            F.linear(hidden_states, value_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        value_states = torch.cat(value_states, dim=-1)
    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    if "qwen3" in self.config._name_or_path.lower():
        query_states = self.q_norm(query_states.view(bsz, q_len, self.config.num_attention_heads, self.head_dim)).transpose(1, 2)
        key_states = self.k_norm(key_states.view(bsz, q_len, self.config.num_key_value_heads, self.head_dim)).transpose(1, 2)
    else:
        query_states = query_states.view(bsz, q_len, self.config.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)

    value_states = value_states.view(
        bsz, q_len, self.config.num_key_value_heads, self.head_dim
    )

    kvc = state.kv_caches[cur_id]
    budget = state.layer2budget[cur_id]

    n_pf_layers = state.n_prefetch_layers
    do_send_pf = do_recv_pf = do_reuse = False
    if q_len == 1 and n_pf_layers > 0 and cur_id >= 2:
        pf_dst_id = cur_id + n_pf_layers
        if pf_dst_id < n_layers:
            dst_budget = state.layer2budget[pf_dst_id]
            if dst_budget is not None and dst_budget < state.n_pages:
                do_send_pf = True
        pf_src_id = cur_id - n_pf_layers
        if pf_src_id >= 2 and budget is not None and budget < state.n_pages:
            do_recv_pf = True
    
    cos, sin = position_embeddings

    # Q: should be updated in any cases
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    query_states = query_states.transpose(1, 2).contiguous()
    key_states = key_states.transpose(1, 2).contiguous()

    if do_recv_pf:
        if budget is not None and kvc.n_pages > budget:
            assert state._dci_future is not None
            eids, nr = state._dci_future.result() # (timeout=1)
            state._dci_future = None

            state.scatter_pages(cur_id, eids, nr)
        else:
            raise NotImplemented("kv cache is expected in the receive state")
    else:
        if budget is not None and kvc.n_pages > budget:
            eids, nr = state.estimate_select_recall(cur_id, query_states)
            
            state.scatter_pages(cur_id, eids, nr)

    if do_send_pf:
        query_states1 = (
            state.attn_layers[pf_dst_id]
            .q_proj(hidden_states)
            .view(bsz, q_len, self.config.num_attention_heads, self.head_dim)
        )
        if "qwen3" in self.config._name_or_path.lower():
            query_states1 = state.attn_layers[pf_dst_id].q_norm(query_states1)
        query_states1, _ = apply_rotary_pos_emb(query_states1, None, cos, sin)
        query_states1 = query_states1.transpose(1, 2).contiguous()

        assert state._dci_future is None
        state._dci_future = asyncio.run_coroutine_threadsafe(
            state.estimate_select_recall_wrapper(pf_dst_id, query_states1), state._loop)

    state.append_paged_kv_cache(cur_id, key_states, value_states)

    attn_page_ids = kvc.c2p
    attn_output = state.decode_sdpa(cur_id, query_states, attn_page_ids)

    if cur_id == 0 and state.offload_win_flag[-1]:
        for l in range(state.n_layers):
            state.default_stream.wait_stream(state.decode_backup_stream)
            state.offload_win_page_to_DCI(l)
            state.offload_win_flag[l] = False

    attn_output = attn_output.reshape(bsz, q_len, -1)

    if 'llama' in self.config._name_or_path and self.config.pretraining_tp > 1:
        attn_output = attn_output.split(
            (self.config.head_dim * self.config.num_attention_heads) // self.config.pretraining_tp, dim=2
        )
        o_proj_slices = self.o_proj.weight.split(
            (self.config.head_dim * self.config.num_attention_heads) // self.config.pretraining_tp, dim=1
        )
        attn_output = sum(
            [
                F.linear(attn_output[i], o_proj_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
        )
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    if cur_id == n_layers - 1:
        state.end_forward(bsz, q_len)

    return attn_output, attn_weights


def _icecache_attn_forward(
    self: LlamaAttention,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    infer_state: InferState = None,
    debug: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    _, q_len, _ = hidden_states.size()

    if q_len > 1:
        return _icecache_prefill(
            self,
            hidden_states,
            attention_mask,
            position_embeddings,
            past_key_value,
            output_attentions,
            use_cache,
            infer_state,
            debug,
            **kwargs,
        )
    else:
        return _icecache_decode(
            self,
            hidden_states,
            attention_mask,
            position_embeddings,
            past_key_value,
            output_attentions,
            use_cache,
            infer_state,
            debug,
            **kwargs,
        )


def enable_icecache(
    self: LlamaForCausalLM,
    dtype: torch.dtype,
    device: torch.device,
    page_size=32,
    infer_state: InferState = None,
    debug: bool = False,
    **kwargs,
):
    if infer_state is None:
        config = self.model.config
        infer_state = InferState(
            n_layers=config.num_hidden_layers,
            n_qo_heads=config.num_attention_heads,
            n_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim if config.head_dim is not None else config.hidden_size // config.num_attention_heads,
            page_size=page_size,
            dtype=dtype,
            device=device,
            debug=debug,
            **kwargs,
        )

    if hasattr(self, "lm_head"):
        _lm_head_forward = self.lm_head.forward
        self.lm_head.forward = lambda x: _lm_head_forward(x[:, -1:, :])

    for mod in self.modules():
        mod_cls = str(mod.__class__)
        if "Attention" in mod_cls:
            mod.forward = (
                lambda mod: lambda *args, **kwargs: _icecache_attn_forward(
                    mod, *args, infer_state=infer_state, debug=debug, **kwargs
                )
            )(mod)
    
    return self