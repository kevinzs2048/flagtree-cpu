# Copyright 2026 FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import ctypes
import platform
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

from triton.language.extra.cpu import kleidiai


EXPECTED_REVISION = "a2abc4d3fb4669dcd30bcad8cd6fd4c232f64924"


def test_kleidiai_revision_is_pinned():
    assert kleidiai.upstream_revision() == EXPECTED_REVISION


def test_source_package_contains_no_native_build_artifacts():
    package_root = Path(kleidiai.__file__).resolve().parent
    artifacts = [
        path
        for pattern in ("*.o", "*.so")
        for path in package_root.rglob(pattern)
    ]
    assert artifacts == []


def test_w4a8_runtime_does_not_force_openmp_team_size():
    source = (
        Path(kleidiai.__file__).resolve().parent
        / "_native"
        / "kleidiai"
        / "w4a8_runtime.c"
    ).read_text(encoding="utf-8")
    assert "#pragma omp parallel num_threads" not in source


@pytest.mark.skipif(
    platform.machine().lower() not in {"aarch64", "arm64"},
    reason="direct KleidiAI runtimes require AArch64",
)
@pytest.mark.parametrize(
    ("variant", "expected"),
    [
        (
            "w4a8",
            {
                "flagtree_kai_w4a8_linear",
                "flagtree_kai_w4a8_pack_rhs",
                "flagtree_kai_w4a8_profile_count",
                "flagtree_kai_w4a8_profile_get",
                "flagtree_kai_w4a8_profile_reset",
                "flagtree_kai_w4a8_rhs_packed_size",
            },
        ),
        (
            "w8a8",
            {
                "flagtree_kai_w8a8_linear",
                "flagtree_kai_w8a8_pack_rhs",
                "flagtree_kai_w8a8_rhs_packed_size",
            },
        ),
    ],
)
def test_runtime_exports_only_public_abi(variant, expected):
    nm = shutil.which("nm")
    if nm is None:
        pytest.skip("nm is required for the native ABI test")
    result = subprocess.run(
        [nm, "-D", "--defined-only", kleidiai.build_runtime(variant)],
        check=True,
        text=True,
        capture_output=True,
    )
    exports = {
        line.split()[-1] for line in result.stdout.splitlines() if line.split()
    }
    assert exports == expected


@pytest.mark.skipif(
    platform.machine().lower() not in {"aarch64", "arm64"},
    reason="direct KleidiAI runtimes require AArch64",
)
@pytest.mark.parametrize("m", [1, 2, 4, 8, 129])
def test_w4a8_runtime_consumes_channelwise_checkpoint_weights(m):
    library = ctypes.CDLL(str(kleidiai.build_runtime("w4a8")))
    library.flagtree_kai_w4a8_rhs_packed_size.restype = ctypes.c_size_t
    library.flagtree_kai_w4a8_rhs_packed_size.argtypes = [ctypes.c_size_t] * 2
    library.flagtree_kai_w4a8_pack_rhs.restype = None
    library.flagtree_kai_w4a8_pack_rhs.argtypes = (
        [ctypes.c_size_t] * 2 + [ctypes.c_void_p] * 3
    )
    library.flagtree_kai_w4a8_linear.restype = None
    library.flagtree_kai_w4a8_linear.argtypes = (
        [ctypes.c_void_p] * 3 + [ctypes.c_int64] * 3
    )

    def pointer(tensor):
        return ctypes.c_void_p(tensor.data_ptr())

    torch.manual_seed(0)
    n, k = 64, 130
    weight = torch.randint(-8, 8, (n, k), dtype=torch.int8)
    unsigned = weight.add(8).to(torch.uint8)
    native = (
        unsigned[:, 0::2] | (unsigned[:, 1::2] << 4)
    ).contiguous()
    weight_scale = (torch.rand(n, dtype=torch.float32) * 0.1 + 1e-4).contiguous()
    packed = torch.empty(
        library.flagtree_kai_w4a8_rhs_packed_size(n, k), dtype=torch.uint8
    )
    library.flagtree_kai_w4a8_pack_rhs(
        n, k, pointer(native), pointer(weight_scale), pointer(packed)
    )

    activation = (torch.rand(m, k) * 4.0 - 1.0).to(torch.bfloat16).contiguous()
    activation[0].zero_()
    output = torch.empty((m, n), dtype=torch.bfloat16)
    library.flagtree_kai_w4a8_linear(
        pointer(activation), pointer(packed), pointer(output), m, k, n
    )

    torch_packed = torch.ops.aten._dyn_quant_pack_4bit_weight(
        native, weight_scale.view(-1, 1), None, k, k, n
    )
    torch_output = torch.ops.aten._dyn_quant_matmul_4bit(
        activation, torch_packed, k, k, n
    )
    torch.testing.assert_close(output, torch_output, rtol=0, atol=0)

    activation_scale = (activation.float().abs().amax(dim=1) / 127.0).clamp(
        min=1e-8
    )
    activation_int8 = (
        (activation.float() / activation_scale[:, None])
        .round()
        .clamp(-127, 127)
    )
    expected = (activation_int8 * activation_scale[:, None]) @ (
        weight.float() * weight_scale[:, None]
    ).T
    relative_error = torch.linalg.vector_norm(output.float() - expected) / (
        torch.linalg.vector_norm(expected) + 1e-12
    )
    assert float(relative_error) < 0.006
    assert torch.count_nonzero(output[0]) == 0


@pytest.mark.skipif(
    platform.machine().lower() not in {"aarch64", "arm64"},
    reason="direct KleidiAI runtimes require AArch64",
)
@pytest.mark.parametrize("m", [1, 7, 129])
def test_w8a8_runtime_uses_symmetric_per_token_activations(m):
    library = ctypes.CDLL(str(kleidiai.build_runtime("w8a8")))
    library.flagtree_kai_w8a8_rhs_packed_size.restype = ctypes.c_size_t
    library.flagtree_kai_w8a8_rhs_packed_size.argtypes = [ctypes.c_size_t] * 2
    library.flagtree_kai_w8a8_pack_rhs.restype = None
    library.flagtree_kai_w8a8_pack_rhs.argtypes = (
        [ctypes.c_size_t] * 2 + [ctypes.c_void_p] * 3
    )
    library.flagtree_kai_w8a8_linear.restype = None
    library.flagtree_kai_w8a8_linear.argtypes = (
        [ctypes.c_void_p] * 3 + [ctypes.c_int64] * 3
    )

    def pointer(tensor):
        return ctypes.c_void_p(tensor.data_ptr())

    torch.manual_seed(0)
    n, k = 64, 130
    weight = torch.randint(-128, 128, (n, k), dtype=torch.int8)
    weight_scale = (torch.rand(n, dtype=torch.float32) * 0.02 + 1e-4).contiguous()
    packed = torch.empty(
        library.flagtree_kai_w8a8_rhs_packed_size(n, k), dtype=torch.uint8
    )
    library.flagtree_kai_w8a8_pack_rhs(
        n, k, pointer(weight), pointer(weight_scale), pointer(packed)
    )

    activation = (torch.rand(m, k) * 4.0 - 1.0).to(torch.bfloat16).contiguous()
    activation[0].zero_()
    output = torch.empty((m, n), dtype=torch.bfloat16)
    library.flagtree_kai_w8a8_linear(
        pointer(activation), pointer(packed), pointer(output), m, k, n
    )

    activation_scale = (activation.float().abs().amax(dim=1) / 127.0).clamp(
        min=1e-8
    )
    activation_int8 = (
        (activation.float() / activation_scale[:, None])
        .round()
        .clamp(-127, 127)
    )
    expected = (activation_int8 * activation_scale[:, None]) @ (
        weight.float() * weight_scale[:, None]
    ).T
    relative_error = torch.linalg.vector_norm(output.float() - expected) / (
        torch.linalg.vector_norm(expected) + 1e-12
    )
    assert float(relative_error) < 0.003
    assert torch.count_nonzero(output[0]) == 0
