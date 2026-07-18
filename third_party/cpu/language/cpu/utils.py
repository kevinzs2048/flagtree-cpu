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

from triton import jit
import triton.language as tl
from triton.language.core import builtin


@jit
def _vnni_decode(arg0):
    tl.static_assert(len(arg0.shape) == 2)
    tmp = arg0.reshape((arg0.shape[0], arg0.shape[1] // 2, 2))
    tmp1, tmp2 = tl.split(tmp)
    return tl.join(tmp1.T, tmp2.T).reshape((arg0.shape[1] // 2, arg0.shape[0] * 2)).T


@builtin
def vnni_decode(arg0, _builder=None, _generator=None):
    bitwidth = arg0.dtype.primitive_bitwidth
    if bitwidth > 16:
        raise ValueError("Expected 8-bit or 16-bit values for vnni_decode")
    decoded = _generator.call_JitFunction(_vnni_decode, (arg0, ), kwargs={})
    if bitwidth == 8:
        decoded = _generator.call_JitFunction(_vnni_decode, (decoded, ), kwargs={})
    return decoded
