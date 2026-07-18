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

import triton
import re


def test_triton_reproducer_path(monkeypatch, tmp_path):
    # If we get a cache hit there will be no reproducer generated
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")

    @triton.jit
    def triton_():
        return

    # We need an temp empty file for MLIR to write the reproducer to, and then
    # the TRITON_REPRODUCER_PATH env var enables crash the reproduction
    # generation in MLIR.
    repro_path = tmp_path / "repro.mlir"
    repro_path.touch()
    monkeypatch.setenv("TRITON_REPRODUCER_PATH", str(repro_path))

    # Run the kernel so MLIR will generate a crash reproducer. It doesn't really
    # matter what the kernel does, just that the PassManager runs its passes.
    triton_[(1, )]()

    repro = repro_path.read_text()
    assert "mlir_reproducer" in repro, f"Expected MLIR reproducer in {repro_path}. Got:\n{repro}"
    m = re.search(r"pipeline: \"(.*)\"", repro)
    assert m, "Expected to match pass pipeline after \"pipeline:\" in MLIR reproducer"
    pipeline_str = m.group(1)
    assert pipeline_str, "Expected non-empty pass pipeline in MLIR reproducer"
