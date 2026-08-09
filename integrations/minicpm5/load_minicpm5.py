"""Native PyTorch loaders for the local MiniCPM5 compressed checkpoints.

The checkpoint tensors are already unpacked signed INT4/INT8.  Loading them
directly avoids a dependency on compressed-tensors and, more importantly,
does not dequantize the matrices into temporary BF16 ``nn.Linear`` objects.
All matrix compute is routed to ordinary Triton Arm kernels.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from accelerate import init_empty_weights
from safetensors import safe_open
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

from flag_gems.runtime.backend._arm.fused.patch_llama_arch import (
    patch_llama_arch,
)
from flag_gems.runtime.backend._arm.int8.tle_int8_linear import TLEInt8Linear
from flag_gems.runtime.backend._arm.q4.aten_linear import AtenQ4Linear
from flag_gems.runtime.backend._arm.q4.optimize_qwen3 import (
    Q4Linear,
    _Q4FusedMLP,
    _Q4ProjectionView,
    _Q4QKVCoordinator,
    _install_decode_attention_fastpath,
)


class _CompressedTensorsQ4ReferenceLinear(torch.nn.Module):
    """Slow PyTorch reference for checkpoint-fidelity validation only."""

    def __init__(self, weight: torch.Tensor, scale: torch.Tensor) -> None:
        super().__init__()
        n, k = weight.shape
        group_size = k // scale.shape[1]
        dequantized = torch.empty((n, k), dtype=torch.bfloat16)
        for begin in range(0, n, 256):
            end = min(begin + 256, n)
            dequantized[begin:end].copy_(
                (
                    weight[begin:end]
                    .reshape(end - begin, -1, group_size)
                    .to(torch.bfloat16)
                    * scale[begin:end, :, None].to(torch.bfloat16)
                ).reshape(end - begin, k)
            )
        self.weight = torch.nn.Parameter(dequantized, requires_grad=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        minimum = torch.minimum(
            value.amin(dim=-1, keepdim=True), torch.zeros_like(value[..., :1])
        )
        maximum = torch.maximum(
            value.amax(dim=-1, keepdim=True), torch.zeros_like(value[..., :1])
        )
        scale = ((maximum - minimum) / 255.0).to(torch.bfloat16)
        scale = torch.where(
            scale == 0,
            torch.tensor(
                torch.finfo(torch.bfloat16).eps,
                dtype=torch.bfloat16,
                device=value.device,
            ),
            scale,
        )
        zero_point = torch.round(
            torch.clamp(
                torch.tensor(-128.0, dtype=torch.bfloat16) - minimum / scale,
                -128,
                127,
            )
        ).to(torch.int8)
        quantized = torch.round(
            torch.clamp(
                value / scale + zero_point.to(torch.bfloat16), -128, 127
            )
        )
        dequantized = (
            quantized.to(torch.bfloat16) - zero_point.to(torch.bfloat16)
        ) * scale
        return torch.nn.functional.linear(dequantized, self.weight)


class _AtenW8A8Linear(torch.nn.Module):
    """Strict checkpoint-semantics W8A8 baseline built from native ATen ops.

    PyTorch currently has no fused public CPU operator matching the
    compressed-tensors token-symmetric A8/per-channel-symmetric W8 contract.
    Keep every stage explicit here: per-token FP32 quantization,
    ``aten::_int_mm``, FP32 scale application, and BF16 output rounding.
    """

    def __init__(self, weight: torch.Tensor, scale: torch.Tensor) -> None:
        super().__init__()
        if weight.dtype != torch.int8 or weight.ndim != 2:
            raise ValueError("ATen W8A8 weight must be INT8 [N,K]")
        n, k = weight.shape
        scale = scale.reshape(-1)
        if scale.numel() != n:
            raise ValueError(f"ATen W8 scale must contain N={n} values")
        self.in_features = k
        self.out_features = n
        self.register_buffer(
            "weight_kn", weight.T.contiguous(), persistent=True
        )
        self.register_buffer(
            "weight_scale", scale.float().contiguous(), persistent=True
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled():
            raise RuntimeError("ATen W8A8 baseline is inference-only")
        shape = value.shape
        value_2d = value.reshape(-1, self.in_features).float()
        absmax = value_2d.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-8)
        activation_scale = absmax / 127.0
        # Stage this exactly like the compiler-generated decode/prefill
        # packers.  ``x / (absmax / 127)`` is mathematically equivalent but
        # can cross an exact FP32 half-tie after reassociation.
        quantized = torch.round(value_2d * (127.0 / absmax)).clamp(
            -127, 127
        ).to(torch.int8)
        output = (
            torch.ops.aten._int_mm(quantized, self.weight_kn).float()
            * activation_scale
            * self.weight_scale.reshape(1, -1)
        ).to(torch.bfloat16)
        return output.reshape(*shape[:-1], self.out_features)


class _AtenW8WeightOnlyLinear(torch.nn.Module):
    """Optimized native ATen A16W8 control; not W8A8-equivalent."""

    def __init__(self, weight: torch.Tensor, scale: torch.Tensor) -> None:
        super().__init__()
        if weight.dtype != torch.int8 or weight.ndim != 2:
            raise ValueError("ATen A16W8 weight must be INT8 [N,K]")
        n, k = weight.shape
        scale = scale.reshape(-1)
        if scale.numel() != n:
            raise ValueError(f"ATen W8 scale must contain N={n} values")
        self.in_features = k
        self.out_features = n
        self.register_buffer("weight", weight.contiguous(), persistent=True)
        self.register_buffer(
            "weight_scale_bf16",
            scale.to(torch.bfloat16).contiguous(),
            persistent=True,
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled():
            raise RuntimeError("ATen A16W8 baseline is inference-only")
        shape = value.shape
        value_2d = value.reshape(-1, self.in_features).to(torch.bfloat16)
        output = torch.ops.aten._weight_int8pack_mm(
            value_2d, self.weight, self.weight_scale_bf16
        )
        return output.reshape(*shape[:-1], self.out_features)


def _config_without_loader_quantization(model_dir: Path) -> LlamaConfig:
    payload = json.loads((model_dir / "config.json").read_text())
    payload.pop("quantization_config", None)
    # Older checkpoints use rope_theta while recent Transformers serializes
    # rope_parameters. LlamaConfig accepts either after normalization.
    return LlamaConfig.from_dict(payload)


def _empty_llama(config: LlamaConfig) -> LlamaForCausalLM:
    with init_empty_weights(include_buffers=True):
        model = LlamaForCausalLM(config)
    # The non-persistent rotary buffer is not present in safetensors.
    model.model.rotary_emb = LlamaRotaryEmbedding(config, device="cpu")
    return model


def _parent_and_leaf(model: torch.nn.Module, name: str):
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _assign_parameter(
    model: torch.nn.Module, name: str, tensor: torch.Tensor
) -> None:
    parent, leaf = _parent_and_leaf(model, name)
    if leaf not in parent._parameters:
        raise KeyError(f"{name} is not a model parameter")
    parent._parameters[leaf] = torch.nn.Parameter(
        tensor.contiguous(), requires_grad=False
    )


def _materialize_remaining_parameters(model, checkpoint) -> list[str]:
    loaded = []
    checkpoint_keys = set(checkpoint.keys())
    for name, parameter in list(model.named_parameters()):
        if not parameter.is_meta:
            continue
        if name not in checkpoint_keys:
            raise KeyError(f"checkpoint does not contain meta parameter {name}")
        _assign_parameter(model, name, checkpoint.get_tensor(name))
        loaded.append(name)
    remaining = [name for name, item in model.named_parameters() if item.is_meta]
    remaining += [name for name, item in model.named_buffers() if item.is_meta]
    if remaining:
        raise RuntimeError(f"MiniCPM loader left meta tensors: {remaining}")
    return loaded


def _replace(parent: torch.nn.Module, name: str, module: torch.nn.Module) -> None:
    setattr(parent, name, module)


def _q4_from_keys(
    checkpoint,
    weight_keys: tuple[str, ...],
    *,
    profile_name: str,
) -> Q4Linear:
    weights = tuple(checkpoint.get_tensor(f"{key}.weight") for key in weight_keys)
    scales = tuple(
        checkpoint.get_tensor(f"{key}.weight_scale") for key in weight_keys
    )
    weight = weights[0] if len(weights) == 1 else torch.cat(weights, dim=0)
    scale = scales[0] if len(scales) == 1 else torch.cat(scales, dim=0)
    return Q4Linear.from_grouped_int4(
        weight,
        scale,
        group_size=128,
        profile_name=profile_name,
    )


def _quantize_w8_per_channel_chunked(
    weight: torch.Tensor, chunk_rows: int = 512
) -> tuple[torch.Tensor, torch.Tensor]:
    n, k = weight.shape
    quantized = torch.empty((n, k), dtype=torch.int8)
    scales = torch.empty((n,), dtype=torch.float32)
    for begin in range(0, n, chunk_rows):
        end = min(begin + chunk_rows, n)
        values = weight[begin:end].float()
        scale = values.abs().amax(dim=1) / 127.0
        reciprocal = torch.where(scale != 0.0, 1.0 / scale, 0.0)
        quantized[begin:end].copy_(
            torch.round(values * reciprocal[:, None])
            .clamp_(-127, 127)
            .to(torch.int8)
        )
        scales[begin:end].copy_(scale)
    return quantized, scales


def _apply_llama_arm_patches(model: torch.nn.Module) -> dict[str, object]:
    patches: dict[str, object] = {"llama": patch_llama_arch()}
    from flag_gems.runtime.backend._arm.fused.patch_qwen3_mlp import (
        patch_qwen3_mlp,
    )
    from flag_gems.runtime.backend._arm.ops import apply_arm_overrides
    from flag_gems.runtime.backend._arm.ops.argmax import (
        set_argmax_vocab_assume_finite,
    )

    patches["mlp_modules"] = patch_qwen3_mlp(model)
    set_argmax_vocab_assume_finite(True)
    apply_arm_overrides(include=("scaled_dot_product_attention", "argmax"))
    return patches


def load_minicpm5_q4(
    model_dir: str | Path,
    *,
    quantize_lm_head: bool = False,
) -> tuple[LlamaForCausalLM, dict[str, object]]:
    """Load GPTQ-G128 signed INT4 directly into the ordinary-Triton Q4 ABI."""
    model_dir = Path(model_dir)
    config = _config_without_loader_quantization(model_dir)
    model = _empty_llama(config)
    checkpoint_path = model_dir / "model.safetensors"
    with safe_open(checkpoint_path, framework="pt", device="cpu") as checkpoint:
        for index, layer in enumerate(model.model.layers):
            prefix = f"model.layers.{index}"
            attention = layer.self_attn
            qkv = _q4_from_keys(
                checkpoint,
                (
                    f"{prefix}.self_attn.q_proj",
                    f"{prefix}.self_attn.k_proj",
                    f"{prefix}.self_attn.v_proj",
                ),
                profile_name="triton::minicpm_q4_qkv",
            )
            logical_sizes = (
                config.num_attention_heads * config.head_dim,
                config.num_key_value_heads * config.head_dim,
                config.num_key_value_heads * config.head_dim,
            )
            coordinator = _Q4QKVCoordinator.from_projection(
                qkv, logical_sizes, config.hidden_size
            )
            attention._triton_qkv_coordinator = coordinator
            attention.q_proj = _Q4ProjectionView(coordinator, 0)
            attention.k_proj = _Q4ProjectionView(coordinator, 1)
            attention.v_proj = _Q4ProjectionView(coordinator, 2)
            _install_decode_attention_fastpath(attention, coordinator)
            attention.o_proj = _q4_from_keys(
                checkpoint,
                (f"{prefix}.self_attn.o_proj",),
                profile_name="triton::minicpm_q4_o",
            )

            gate_up = _q4_from_keys(
                checkpoint,
                (f"{prefix}.mlp.gate_proj", f"{prefix}.mlp.up_proj"),
                profile_name="triton::minicpm_q4_gate_up",
            )
            down = _q4_from_keys(
                checkpoint,
                (f"{prefix}.mlp.down_proj",),
                profile_name="triton::minicpm_q4_down",
            )
            layer.mlp = _Q4FusedMLP.from_projections(
                gate_up, down, config.intermediate_size
            )

        if quantize_lm_head:
            model.lm_head = Q4Linear.from_weight(
                checkpoint.get_tensor("lm_head.weight"),
                profile_name="triton::minicpm_q4_lm_head",
                chunk_rows=512,
            )
        _materialize_remaining_parameters(model, checkpoint)

    patches = _apply_llama_arm_patches(model)
    model.eval()
    setup = {
        "quantization": (
            "compressed_tensors_signed_q4_g128_"
            "token_asymmetric_a8_bf16_scale"
        ),
        "decoder_layers": len(model.model.layers),
        "physical_q4_matrices": len(model.model.layers) * 4
        + int(quantize_lm_head),
        "lm_head_q4": quantize_lm_head,
        "patches": patches,
        "arm_overrides": ["scaled_dot_product_attention", "argmax"],
    }
    return model, setup


def load_minicpm5_q4_reference(
    model_dir: str | Path,
) -> tuple[LlamaForCausalLM, dict[str, object]]:
    """Load the documented compressed-tensors Q4 QDQ graph in PyTorch."""
    model_dir = Path(model_dir)
    config = _config_without_loader_quantization(model_dir)
    model = _empty_llama(config)
    checkpoint_path = model_dir / "model.safetensors"
    replaced = 0
    with safe_open(checkpoint_path, framework="pt", device="cpu") as checkpoint:
        keys = set(checkpoint.keys())
        for name, module in list(model.named_modules()):
            if not isinstance(module, torch.nn.Linear) or name == "lm_head":
                continue
            weight_key = f"{name}.weight"
            scale_key = f"{name}.weight_scale"
            if weight_key not in keys or scale_key not in keys:
                continue
            parent, leaf = _parent_and_leaf(model, name)
            _replace(
                parent,
                leaf,
                _CompressedTensorsQ4ReferenceLinear(
                    checkpoint.get_tensor(weight_key),
                    checkpoint.get_tensor(scale_key),
                ),
            )
            replaced += 1
        _materialize_remaining_parameters(model, checkpoint)
    model.eval()
    return model, {
        "quantization": "compressed_tensors_q4_qdq_pytorch_reference",
        "reference_linears": replaced,
        "lm_head_q4": False,
    }


def load_minicpm5_q4_aten(
    model_dir: str | Path,
    *,
    quantize_lm_head: bool = False,
) -> tuple[LlamaForCausalLM, dict[str, object]]:
    """Load the Q4/G128 checkpoint into native ATen/KleidiAI operators."""
    model_dir = Path(model_dir)
    config = _config_without_loader_quantization(model_dir)
    model = _empty_llama(config)
    checkpoint_path = model_dir / "model.safetensors"
    replaced = 0
    with safe_open(checkpoint_path, framework="pt", device="cpu") as checkpoint:
        keys = set(checkpoint.keys())
        for name, module in list(model.named_modules()):
            if not isinstance(module, torch.nn.Linear) or name == "lm_head":
                continue
            weight_key = f"{name}.weight"
            scale_key = f"{name}.weight_scale"
            if weight_key not in keys or scale_key not in keys:
                continue
            parent, leaf = _parent_and_leaf(model, name)
            _replace(
                parent,
                leaf,
                AtenQ4Linear.from_grouped_int4(
                    checkpoint.get_tensor(weight_key),
                    checkpoint.get_tensor(scale_key),
                    group_size=128,
                    profile_name=f"aten_q4::{name}",
                ),
            )
            replaced += 1
        if quantize_lm_head:
            model.lm_head = AtenQ4Linear.from_weight(
                checkpoint.get_tensor("lm_head.weight"),
                profile_name="aten_q4::lm_head",
                chunk_rows=512,
            )
            replaced += 1
        _materialize_remaining_parameters(model, checkpoint)
    model.eval()
    return model, {
        "quantization": "aten_kleidiai_dynamic_a8_signed_q4_g128",
        "aten_q4_linears": replaced,
        "lm_head_q4": quantize_lm_head,
        "model_level_triton_fusions": False,
    }


def load_minicpm5_q4_aten_fused(
    model_dir: str | Path,
    *,
    quantize_lm_head: bool = False,
) -> tuple[LlamaForCausalLM, dict[str, object]]:
    """Controlled hybrid: joined ATen/KleidiAI matrices plus Arm fusions.

    This is not reported as Triton Q4 codegen.  It holds the packed QKV and
    gate/up model structure constant while replacing the slower generated
    matrix body with native ATen/KleidiAI, isolating backend cost from fusion
    and Python dispatcher cost.
    """
    model_dir = Path(model_dir)
    config = _config_without_loader_quantization(model_dir)
    model = _empty_llama(config)
    checkpoint_path = model_dir / "model.safetensors"
    with safe_open(checkpoint_path, framework="pt", device="cpu") as checkpoint:
        for index, layer in enumerate(model.model.layers):
            prefix = f"model.layers.{index}"
            attention = layer.self_attn
            weights = tuple(
                checkpoint.get_tensor(
                    f"{prefix}.self_attn.{name}.weight"
                )
                for name in ("q_proj", "k_proj", "v_proj")
            )
            scales = tuple(
                checkpoint.get_tensor(
                    f"{prefix}.self_attn.{name}.weight_scale"
                )
                for name in ("q_proj", "k_proj", "v_proj")
            )
            qkv = AtenQ4Linear.from_grouped_int4(
                torch.cat(weights, dim=0),
                torch.cat(scales, dim=0),
                group_size=128,
                profile_name="aten_q4::minicpm_joined_qkv",
            )
            logical_sizes = (
                config.num_attention_heads * config.head_dim,
                config.num_key_value_heads * config.head_dim,
                config.num_key_value_heads * config.head_dim,
            )
            coordinator = _Q4QKVCoordinator.from_projection(
                qkv, logical_sizes, config.hidden_size
            )
            attention._triton_qkv_coordinator = coordinator
            attention.q_proj = _Q4ProjectionView(coordinator, 0)
            attention.k_proj = _Q4ProjectionView(coordinator, 1)
            attention.v_proj = _Q4ProjectionView(coordinator, 2)
            attention.o_proj = AtenQ4Linear.from_grouped_int4(
                checkpoint.get_tensor(f"{prefix}.self_attn.o_proj.weight"),
                checkpoint.get_tensor(
                    f"{prefix}.self_attn.o_proj.weight_scale"
                ),
                group_size=128,
                profile_name="aten_q4::minicpm_o",
            )

            gate_up = AtenQ4Linear.from_grouped_int4(
                torch.cat(
                    (
                        checkpoint.get_tensor(
                            f"{prefix}.mlp.gate_proj.weight"
                        ),
                        checkpoint.get_tensor(
                            f"{prefix}.mlp.up_proj.weight"
                        ),
                    ),
                    dim=0,
                ),
                torch.cat(
                    (
                        checkpoint.get_tensor(
                            f"{prefix}.mlp.gate_proj.weight_scale"
                        ),
                        checkpoint.get_tensor(
                            f"{prefix}.mlp.up_proj.weight_scale"
                        ),
                    ),
                    dim=0,
                ),
                group_size=128,
                profile_name="aten_q4::minicpm_joined_gate_up",
            )
            down = AtenQ4Linear.from_grouped_int4(
                checkpoint.get_tensor(f"{prefix}.mlp.down_proj.weight"),
                checkpoint.get_tensor(f"{prefix}.mlp.down_proj.weight_scale"),
                group_size=128,
                profile_name="aten_q4::minicpm_down",
            )
            layer.mlp = _Q4FusedMLP.from_projections(
                gate_up, down, config.intermediate_size
            )

        if quantize_lm_head:
            model.lm_head = AtenQ4Linear.from_weight(
                checkpoint.get_tensor("lm_head.weight"),
                profile_name="aten_q4::lm_head",
                chunk_rows=512,
            )
        _materialize_remaining_parameters(model, checkpoint)

    patches = _apply_llama_arm_patches(model)
    model.eval()
    return model, {
        "quantization": "hybrid_joined_aten_kleidiai_q4_g128",
        "matrix_backend": "aten_kleidiai",
        "joined_qkv_matrices": len(model.model.layers),
        "joined_gate_up_matrices": len(model.model.layers),
        "lm_head_q4": quantize_lm_head,
        "non_matrix_arm_patches": patches,
        "reported_as_triton_q4_codegen": False,
    }


def load_minicpm5_w8(
    model_dir: str | Path,
    *,
    quantize_lm_head: bool = False,
) -> tuple[LlamaForCausalLM, dict[str, object]]:
    """Load frozen per-channel W8 directly into SDOT/I8MM Triton modules."""
    model_dir = Path(model_dir)
    config = _config_without_loader_quantization(model_dir)
    model = _empty_llama(config)
    checkpoint_path = model_dir / "model.safetensors"
    replaced = 0
    with safe_open(checkpoint_path, framework="pt", device="cpu") as checkpoint:
        keys = set(checkpoint.keys())
        for name, module in list(model.named_modules()):
            if not isinstance(module, torch.nn.Linear) or name == "lm_head":
                continue
            weight_key = f"{name}.weight"
            scale_key = f"{name}.weight_scale"
            if weight_key not in keys or scale_key not in keys:
                continue
            parent, leaf = _parent_and_leaf(model, name)
            _replace(
                parent,
                leaf,
                TLEInt8Linear(
                    checkpoint.get_tensor(weight_key),
                    checkpoint.get_tensor(scale_key),
                ),
            )
            replaced += 1
        if quantize_lm_head:
            quantized, scale = _quantize_w8_per_channel_chunked(
                checkpoint.get_tensor("lm_head.weight")
            )
            model.lm_head = TLEInt8Linear(quantized, scale)
            replaced += 1
        _materialize_remaining_parameters(model, checkpoint)

    from flag_gems.runtime.backend._arm.fused.patch_qwen3_qkv import (
        patch_qwen3_qkv,
    )

    patches = _apply_llama_arm_patches(model)
    patches["qkv_modules"] = patch_qwen3_qkv(model)
    model.eval()
    setup = {
        "quantization": "checkpoint_per_channel_symmetric_w8",
        "decoder_layers": len(model.model.layers),
        "w8_linears": replaced,
        "lm_head_w8": quantize_lm_head,
        "patches": patches,
        "arm_overrides": ["scaled_dot_product_attention", "argmax"],
    }
    return model, setup


def load_minicpm5_w8_aten(
    model_dir: str | Path,
    *,
    weight_only: bool = False,
    quantize_lm_head: bool = False,
) -> tuple[LlamaForCausalLM, dict[str, object]]:
    """Load W8 using native ATen, without any Triton model patches.

    The default is the exact W8A8 checkpoint contract.  ``weight_only`` is a
    deliberately stronger native-kernel control using A16W8; it is reported
    separately because it skips activation quantization and is not
    numerically equivalent to the checkpoint.
    """
    model_dir = Path(model_dir)
    config = _config_without_loader_quantization(model_dir)
    model = _empty_llama(config)
    checkpoint_path = model_dir / "model.safetensors"
    linear_cls = _AtenW8WeightOnlyLinear if weight_only else _AtenW8A8Linear
    replaced = 0
    with safe_open(checkpoint_path, framework="pt", device="cpu") as checkpoint:
        keys = set(checkpoint.keys())
        for name, module in list(model.named_modules()):
            if not isinstance(module, torch.nn.Linear) or name == "lm_head":
                continue
            weight_key = f"{name}.weight"
            scale_key = f"{name}.weight_scale"
            if weight_key not in keys or scale_key not in keys:
                continue
            parent, leaf = _parent_and_leaf(model, name)
            _replace(
                parent,
                leaf,
                linear_cls(
                    checkpoint.get_tensor(weight_key),
                    checkpoint.get_tensor(scale_key),
                ),
            )
            replaced += 1
        if quantize_lm_head:
            quantized, scale = _quantize_w8_per_channel_chunked(
                checkpoint.get_tensor("lm_head.weight")
            )
            model.lm_head = linear_cls(quantized, scale)
            replaced += 1
        _materialize_remaining_parameters(model, checkpoint)
    model.eval()
    return model, {
        "quantization": (
            "aten_a16w8_weight_only_control"
            if weight_only
            else "aten_eager_token_symmetric_a8_per_channel_w8"
        ),
        "aten_w8_linears": replaced,
        "lm_head_w8": quantize_lm_head,
        "same_w8a8_semantics": not weight_only,
        "model_level_triton_fusions": False,
    }


__all__ = [
    "load_minicpm5_q4",
    "load_minicpm5_q4_aten",
    "load_minicpm5_q4_aten_fused",
    "load_minicpm5_q4_reference",
    "load_minicpm5_w8",
    "load_minicpm5_w8_aten",
]
