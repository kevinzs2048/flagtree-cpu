#!/usr/bin/env python3
"""Report the exact GGUF tensor coverage of the llama Triton W4 route."""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "third_party/llama.cpp-w4/gguf-py"))

from gguf import GGUFReader  # noqa: E402


Q4_0 = 2
Q4_1 = 3
Q4_0_SHAPES = {
    (2560, 1024),
    (2560, 2560),
    (2560, 4096),
    (2560, 9728),
    (4096, 2560),
    (9728, 2560),
}


def type_id(tensor) -> int:
    try:
        return int(tensor.tensor_type)
    except TypeError:
        return tensor.tensor_type.value


def shape(tensor) -> tuple[int, ...]:
    return tuple(int(value) for value in tensor.shape)


def is_routed_projection(tensor) -> bool:
    tensor_shape = shape(tensor)
    tensor_type = type_id(tensor)
    return (tensor_type == Q4_0 and tensor_shape in Q4_0_SHAPES) or (
        tensor_type == Q4_1 and tensor_shape == (9728, 2560)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    args = parser.parse_args()

    reader = GGUFReader(args.model)
    covered = [tensor for tensor in reader.tensors if is_routed_projection(tensor)]
    total_params = sum(math.prod(shape(tensor)) for tensor in reader.tensors)
    covered_params = sum(math.prod(shape(tensor)) for tensor in covered)
    total_bytes = sum(tensor.n_bytes for tensor in reader.tensors)
    covered_bytes = sum(tensor.n_bytes for tensor in covered)
    buckets = Counter((type_id(tensor), shape(tensor)) for tensor in covered)

    print(f"model={args.model}")
    print(f"total_tensors={len(reader.tensors)}")
    print(f"routed_tensors={len(covered)}")
    print(f"total_params={total_params}")
    print(f"routed_params={covered_params}")
    print(f"routed_param_percent={100.0 * covered_params / total_params:.6f}")
    print(f"total_tensor_bytes={total_bytes}")
    print(f"routed_tensor_bytes={covered_bytes}")
    print(f"routed_byte_percent={100.0 * covered_bytes / total_bytes:.6f}")
    for (tensor_type, tensor_shape), count in sorted(buckets.items()):
        print(f"type={tensor_type} shape={tensor_shape} tensors={count}")


if __name__ == "__main__":
    main()
