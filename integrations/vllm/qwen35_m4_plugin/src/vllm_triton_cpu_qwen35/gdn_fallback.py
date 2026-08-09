"""Pure PyTorch GDN recurrence used to quarantine Darwin Triton crashes.

This is intentionally an inference-first implementation.  It mirrors the
math in vLLM's FLA kernels and keeps all recurrent accumulation in FP32.  The
optimized W4A8 projections remain on Triton/libtriton_jit.
"""

from __future__ import annotations

from itertools import pairwise

import torch
import torch.nn.functional as F

_INSTALLED = False


def _apply_activation(x: torch.Tensor, activation: bool | str | None) -> torch.Tensor:
    if activation is True or activation in ("silu", "swish"):
        return F.silu(x)
    if activation is False or activation is None:
        return x
    raise NotImplementedError("activation must be None, silu, or swish")


def _conv_step(
    state: torch.Tensor,
    token: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation: bool | str | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one depthwise causal-convolution step in FP32."""
    width = weight.shape[1]
    history = state[:, -(width - 1) :] if width > 1 else state[:, :0]
    window = torch.cat((history.float(), token.float().unsqueeze(-1)), dim=-1)
    output = torch.sum(window * weight.float(), dim=-1)
    if bias is not None:
        output = output + bias.float()
    output = _apply_activation(output, activation)
    next_state = torch.cat((state, token.to(state.dtype).unsqueeze(-1)), dim=-1)
    return output, next_state[:, -state.shape[-1] :]


def _cache_index(
    indices: torch.Tensor | None,
    sequence: int,
    offset: int = 0,
) -> int:
    if indices is None:
        return sequence
    if indices.ndim == 1:
        return int(indices[sequence].item())
    return int(indices[sequence, offset].item())


def torch_causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    activation: str | None = "silu",
    pad_slot_id: int = -1,
    block_idx_first_scheduled_token: torch.Tensor | None = None,
    block_idx_last_scheduled_token: torch.Tensor | None = None,
    initial_state_idx: torch.Tensor | None = None,
    num_computed_tokens: torch.Tensor | None = None,
    block_size_to_align=0,
    metadata=None,
    validate_data=False,
) -> torch.Tensor:
    """CPU reference for Qwen3.5 GDN continuous-batch prefill convolution."""
    del block_idx_first_scheduled_token, num_computed_tokens
    del block_size_to_align, metadata, validate_data
    if block_idx_last_scheduled_token is not None:
        raise NotImplementedError("prefix-cached GDN convolution is not enabled on Darwin")
    if x.ndim != 2:
        raise ValueError(f"expected x shaped [dim, tokens], got {tuple(x.shape)}")

    offsets = query_start_loc.detach().cpu().tolist()
    output = torch.zeros_like(x)
    for sequence, (begin, end) in enumerate(pairwise(offsets)):
        state_offset = int(initial_state_idx[sequence].item()) if initial_state_idx is not None else 0
        cache_index = _cache_index(cache_indices, sequence, state_offset)
        if cache_index == pad_slot_id:
            continue
        use_initial = has_initial_state is not None and bool(has_initial_state[sequence].item())
        state = conv_states[cache_index].clone() if use_initial else torch.zeros_like(conv_states[cache_index])
        for token_index in range(begin, end):
            token_output, state = _conv_step(state, x[:, token_index], weight, bias, activation)
            output[:, token_index].copy_(token_output.to(output.dtype))
        conv_states[cache_index].copy_(state)
    return output


def torch_causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: bool | str | None = None,
    conv_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    max_query_len: int = -1,
    pad_slot_id: int = -1,
    block_idx_last_scheduled_token: torch.Tensor | None = None,
    initial_state_idx: torch.Tensor | None = None,
    validate_data=False,
) -> torch.Tensor:
    """CPU reference for Qwen3.5 GDN decode convolution and cache update."""
    del max_query_len, validate_data
    if num_accepted_tokens is not None:
        raise NotImplementedError("speculative GDN decode is not enabled on Darwin")
    if block_idx_last_scheduled_token is not None:
        raise NotImplementedError("prefix-cached GDN decode is not enabled on Darwin")

    output = torch.zeros_like(x)
    if query_start_loc is None:
        if x.ndim == 2:
            sequences = [x[index : index + 1] for index in range(x.shape[0])]
            ranges = [(index, index + 1) for index in range(x.shape[0])]
        elif x.ndim == 3:
            sequences = [x[index].transpose(0, 1) for index in range(x.shape[0])]
            ranges = None
        else:
            raise ValueError(f"unsupported x shape {tuple(x.shape)}")
    else:
        offsets = query_start_loc.detach().cpu().tolist()
        ranges = list(pairwise(offsets))
        sequences = [x[begin:end] for begin, end in ranges]

    for sequence, tokens in enumerate(sequences):
        state_offset = int(initial_state_idx[sequence].item()) if initial_state_idx is not None else 0
        cache_index = _cache_index(conv_state_indices, sequence, state_offset)
        if cache_index == pad_slot_id:
            continue
        state = conv_state[cache_index].clone()
        sequence_output = []
        for token in tokens:
            token_output, state = _conv_step(state, token, weight, bias, activation)
            sequence_output.append(token_output.to(output.dtype))
        conv_state[cache_index].copy_(state)
        stacked = torch.stack(sequence_output)
        if x.ndim == 3:
            output[sequence].copy_(stacked.transpose(0, 1))
        else:
            assert ranges is not None
            begin, end = ranges[sequence]
            output[begin:end].copy_(stacked)
    return output


def _expanded_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    value_heads: int,
    normalize: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    query_heads = q.shape[-2]
    if value_heads % query_heads:
        raise ValueError(f"value heads ({value_heads}) must be divisible by query heads ({query_heads})")
    if normalize:
        q = q * torch.rsqrt(torch.sum(q * q, dim=-1, keepdim=True) + 1.0e-6)
        k = k * torch.rsqrt(torch.sum(k * k, dim=-1, keepdim=True) + 1.0e-6)
    repeats = value_heads // query_heads
    return (
        q.repeat_interleave(repeats, dim=-2),
        k.repeat_interleave(repeats, dim=-2),
    )


def _step(
    state: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    state = state * torch.exp(g.float()).reshape(-1, 1, 1)
    predicted = torch.bmm(state, k.float().unsqueeze(-1)).squeeze(-1)
    residual = v.float() - predicted
    beta_f = beta.float()
    if beta_f.ndim == 1:
        beta_f = beta_f[:, None]
    residual = residual * beta_f
    state = state + residual.unsqueeze(-1) * k.float().unsqueeze(-2)
    output = torch.bmm(state, (q.float() * scale).unsqueeze(-1)).squeeze(-1)
    return output, state


def torch_chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = True,
):
    """Reference GDN prefill with the vLLM chunk-op signature."""
    batch, tokens, _, key_dim = q.shape
    value_heads = v.shape[-2]
    output = torch.empty_like(v)
    if cu_seqlens is None:
        ranges = [(index, 0, tokens) for index in range(batch)]
    else:
        offsets = cu_seqlens.detach().cpu().tolist()
        ranges = [(0, offsets[i], offsets[i + 1]) for i in range(len(offsets) - 1)]

    final_states: list[torch.Tensor] = []
    for sequence, (batch_index, begin, end) in enumerate(ranges):
        if initial_state is None:
            state = torch.zeros(
                value_heads,
                v.shape[-1],
                key_dim,
                dtype=torch.float32,
                device=q.device,
            )
        else:
            state = initial_state[sequence].float().clone()
        for token in range(begin, end):
            q_token, k_token = _expanded_qk(
                q[batch_index, token].float(),
                k[batch_index, token].float(),
                value_heads,
                use_qk_l2norm_in_kernel,
            )
            token_output, state = _step(
                state,
                q_token,
                k_token,
                v[batch_index, token],
                g[batch_index, token],
                beta[batch_index, token],
                key_dim**-0.5,
            )
            output[batch_index, token].copy_(token_output.to(output.dtype))
        final_states.append(state)

    final_state = None
    if output_final_state:
        final_state = torch.stack(final_states).to(initial_state.dtype)
    return output, final_state


def torch_fused_sigmoid_gating_delta_rule_update(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    inplace_final_state: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    is_kda: bool = False,
):
    """Reference decode recurrence with vLLM's fused-op signature."""
    if is_kda:
        raise NotImplementedError("The Darwin safety fallback currently covers GDN only")
    if initial_state is None:
        raise ValueError("initial_state is required for vLLM GDN inference")

    batch, tokens, _, key_dim = q.shape
    value_heads = v.shape[-2]
    scale = key_dim**-0.5 if scale is None else scale
    gate_input = a.reshape(-1, value_heads).float() + dt_bias.float()
    decay = -torch.exp(A_log.float()) * F.softplus(gate_input, beta=beta, threshold=threshold)
    update = torch.sigmoid(b.reshape(-1, value_heads).float())
    decay = decay.reshape(batch, tokens, value_heads)
    update = update.reshape(batch, tokens, value_heads)
    output = torch.zeros_like(v)

    if cu_seqlens is None:
        ranges = [(index, 0, tokens) for index in range(batch)]
    else:
        offsets = cu_seqlens.detach().cpu().tolist()
        ranges = [(0, offsets[i], offsets[i + 1]) for i in range(len(offsets) - 1)]

    if inplace_final_state:
        final_state = initial_state
    else:
        final_state = torch.empty(
            tokens,
            value_heads,
            v.shape[-1],
            key_dim,
            dtype=initial_state.dtype,
            device=initial_state.device,
        )

    def state_index(sequence: int, token_offset: int, initial: bool) -> int:
        if ssm_state_indices is None:
            return sequence
        indices = ssm_state_indices
        if indices.ndim == 1:
            return int(indices[sequence].item())
        if initial and num_accepted_tokens is not None:
            token_offset = int(num_accepted_tokens[sequence].item()) - 1
        return int(indices[sequence, token_offset].item())

    for sequence, (batch_index, begin, end) in enumerate(ranges):
        first_index = state_index(sequence, 0, True)
        if first_index < 0:
            continue
        state = initial_state[first_index].float().clone()
        for local_token, token in enumerate(range(begin, end)):
            q_token, k_token = _expanded_qk(
                q[batch_index, token].float(),
                k[batch_index, token].float(),
                value_heads,
                use_qk_l2norm_in_kernel,
            )
            token_output, state = _step(
                state,
                q_token,
                k_token,
                v[batch_index, token],
                decay[batch_index, token],
                update[batch_index, token],
                scale,
            )
            output[batch_index, token].copy_(token_output.to(output.dtype))
            if inplace_final_state:
                final_index = state_index(sequence, local_token, False)
                if final_index >= 0:
                    final_state[final_index].copy_(state.to(final_state.dtype))
            else:
                final_state[token].copy_(state.to(final_state.dtype))
    return output, final_state


def install_vllm_gdn_fallback() -> None:
    """Patch the unsafe Triton GDN entry points used by Qwen3.5."""
    global _INSTALLED
    if _INSTALLED:
        return
    import vllm.model_executor.layers.mamba.gdn_linear_attn as gdn

    gdn.causal_conv1d_fn = torch_causal_conv1d_fn
    gdn.causal_conv1d_update = torch_causal_conv1d_update
    gdn.fla_chunk_gated_delta_rule = torch_chunk_gated_delta_rule
    gdn.fused_sigmoid_gating_delta_rule_update = torch_fused_sigmoid_gating_delta_rule_update

    # The stock warmup exists only to JIT/autotune Triton/FlashInfer kernels.
    # Besides being unnecessary here, it calls torch.accelerator.empty_cache(),
    # which selects MPS on macOS even though this vLLM worker runs on CPU.
    def no_op_prefill_warmup(self, mixed_qkv: torch.Tensor) -> None:
        return None

    gdn.GatedDeltaNetAttention._warmup_prefill_kernels = no_op_prefill_warmup
    gdn._triton_cpu_qwen35_safe_gdn = True
    _INSTALLED = True


__all__ = [
    "install_vllm_gdn_fallback",
    "torch_causal_conv1d_fn",
    "torch_causal_conv1d_update",
    "torch_chunk_gated_delta_rule",
    "torch_fused_sigmoid_gating_delta_rule_update",
]
