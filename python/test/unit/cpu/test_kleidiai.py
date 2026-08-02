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

import platform
import shutil
import subprocess
from pathlib import Path

import pytest

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
