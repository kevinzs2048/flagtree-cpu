# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""CPU TLE native leaves, reusing the existing ARM runtime-call architecture.

This scalar C ABI entry is an extension of the local ARM TLE implementation,
not the separate main-branch raw.dialect source-region interface. The caller
loads the native operator library before the compiled kernel is loaded.
"""
from triton._C.libtriton import cpu
from triton.language import core

if not hasattr(cpu, "create_raw_call"):
    raise ImportError("Rebuild the CPU compiler with the TLE native-call binding first")


@core.builtin
def raw_call(symbol, *args, result_dtype=core.int32, _semantic=None):
    """Call a scalar integer C ABI leaf with conservative buffer effects.

    Pass pointer addresses as uint64 on 64-bit hosts. Values retain their
    explicit integer widths. This call is retained when its result is unused.
    """
    symbol = core._unwrap_if_constexpr(symbol)
    result_dtype = core._unwrap_if_constexpr(result_dtype)
    values = [_semantic.to_tensor(x) for x in args]
    if not isinstance(symbol, str) or not symbol:
        raise TypeError("CPU Raw requires a nonempty compile-time C ABI symbol")
    if any(x.type.is_block() or not x.dtype.is_int() for x in values):
        raise TypeError("CPU Raw native-call arguments must be scalar integers")
    if not result_dtype.is_int():
        raise TypeError("CPU Raw native-call result must be a scalar integer")
    handle = cpu.create_raw_call(_semantic.builder, symbol,
                                 [x.handle for x in values],
                                 result_dtype.to_ir(_semantic.builder))
    return core.tensor(handle, result_dtype)
