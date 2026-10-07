import os
import pytest
import torch

import triton
import triton.language as tl
"""
This test is only for the CPU backend.
"""


def is_interpreter():
    return os.environ.get('TRITON_INTERPRET', '0') == '1'


def is_cpu():
    return not is_interpreter() and \
        triton.runtime.driver.active.get_current_target().backend == "cpu"


def is_x86():
    return is_cpu() and \
        triton.runtime.driver.active.get_current_target().arch == "x86_64"


def test_scalar_pointer_arith(device):

    @triton.jit
    def kernel(src, dst, BLOCK_SIZE: tl.constexpr):
        offs = tl.arange(0, BLOCK_SIZE)
        x = tl.load(src + offs)
        tl.store(dst + offs, x)

    src = torch.rand((128, ), dtype=torch.float32, device=device)
    res = torch.empty_like(src)
    meta = kernel[(1, )](src, res, BLOCK_SIZE=128)
    assert (src == res).all()

    # Check TTCIR doesn't have pointer extraction from a tensor.
    ttcir = meta.asm["ttcir"]
    assert ttcir.count("extract") == 0


@pytest.mark.parametrize("size", [32, 128, 135])
@pytest.mark.parametrize("tile_size", [16])
def test_optimize_tile_mask(size, tile_size, device):
    pytest.xfail("masked vector ops no longer eliminated by upstream canonicalizers")

    @triton.jit
    def kernel(src, dst, size, TILE_SIZE: tl.constexpr):
        for i in range(0, tl.cdiv(size, TILE_SIZE)):
            offs = tl.arange(0, TILE_SIZE) + i * TILE_SIZE
            mask = offs < size
            x = tl.load(src + offs, mask=mask, other=0)
            tl.store(dst + offs, x, mask=mask)

    src = torch.rand((size, ), dtype=torch.float32, device='cpu')
    res = torch.empty_like(src)
    meta = kernel[(1, )](src, res, size, TILE_SIZE=tile_size)
    assert (src == res).all()

    # Check number of masked loads and stores.
    tttcir = meta.asm["tttcir"]
    masked_loads = tttcir.count("maskedload")
    masked_stores = tttcir.count("maskedstore")
    if size % tile_size == 0:
        assert masked_loads == 0
        assert masked_stores == 0
    else:
        assert masked_loads == 1
        assert masked_stores == 1


# Regression test for compilation failure in masks optimization
def test_vec_cdiv(device):

    @triton.jit
    def kernel(in_ptr, out_ptr):
        offs = tl.arange(0, 16)
        x = tl.load(in_ptr + offs)
        res = (x + 15) // 16
        tl.store(out_ptr + offs, res)

    arg0 = torch.zeros((16, ), dtype=torch.int32)
    arg1 = torch.empty_like(arg0)
    kernel[(1, )](arg0, arg1)


@pytest.mark.parametrize("enabled", [False, True])
def test_single_program_c_entry_preserves_grid_launcher(enabled):
    import ctypes

    @triton.jit
    def kernel(out, value):
        x, y, z = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        nx, ny, nz = tl.num_programs(0), tl.num_programs(1), tl.num_programs(2)
        tl.store(out + x + nx * (y + ny * z), value + 100 * nx + 10 * ny + nz)

    output = torch.full((8,), -1, dtype=torch.int32)
    # Warmup exports metadata but never executes the kernel.
    compiled = kernel.warmup(output, 7, grid=(2, 2, 2), enable_single_program_entry=enabled)
    assert torch.equal(output, torch.full_like(output, -1))
    entry_name = getattr(compiled.metadata, "single_program_name", None)
    if enabled:
        assert entry_name == compiled.name + "__single_program"
        library = ctypes.CDLL(compiled.metadata_group[compiled.name + ".so"])
        entry = getattr(library, entry_name)
        entry.argtypes = [ctypes.c_void_p, ctypes.c_int32]
        entry.restype = None
        entry(output.data_ptr(), 7)
        assert output.tolist() == [118, -1, -1, -1, -1, -1, -1, -1]
    else:
        assert entry_name is None
    # Enabling the extra entry must not specialize the ordinary grid to one.
    kernel[(2, 2, 2)](output, 7, enable_single_program_entry=enabled)
    assert output.tolist() == [229] * 8
