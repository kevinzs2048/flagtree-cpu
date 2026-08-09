import torch
from vllm_triton_cpu_qwen35.gdn_fallback import (
    torch_causal_conv1d_fn,
    torch_causal_conv1d_update,
    torch_chunk_gated_delta_rule,
)


def _causal_conv_reference(x, state, weight, bias):
    outputs = []
    for token in x:
        window = torch.cat((state, token[:, None]), dim=1)
        outputs.append(torch.sum(window * weight, dim=1) + bias)
        state = window[:, 1:]
    return torch.stack(outputs), state


def test_torch_causal_conv1d_prefill_updates_selected_cache():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    weight = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    bias = torch.tensor([0.25, -0.5])
    states = torch.full((4, 2, 2), -7.0)
    states[2] = torch.tensor([[9.0, 10.0], [11.0, 12.0]])
    expected0, final0 = _causal_conv_reference(x[:, :2].T, torch.zeros(2, 2), weight, bias)
    expected1, final1 = _causal_conv_reference(x[:, 2:].T, states[2].clone(), weight, bias)

    actual = torch_causal_conv1d_fn(
        x,
        weight,
        bias,
        states,
        query_start_loc=torch.tensor([0, 2, 4]),
        cache_indices=torch.tensor([1, 2]),
        has_initial_state=torch.tensor([False, True]),
        activation=None,
    )

    torch.testing.assert_close(actual[:, :2].T, expected0)
    torch.testing.assert_close(actual[:, 2:].T, expected1)
    torch.testing.assert_close(states[1], final0)
    torch.testing.assert_close(states[2], final1)


def test_torch_causal_conv1d_decode_continues_prefill_state():
    weight = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    bias = torch.tensor([0.25, -0.5])
    states = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    tokens = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    expected, final = _causal_conv_reference(tokens, states[0].clone(), weight, bias)

    first = torch_causal_conv1d_update(tokens[:1], states, weight, bias, activation=None)
    second = torch_causal_conv1d_update(tokens[1:], states, weight, bias, activation=None)

    torch.testing.assert_close(torch.cat((first, second)), expected)
    torch.testing.assert_close(states[0], final)


def test_torch_chunk_gated_delta_rule_matches_scalar_reference():
    torch.manual_seed(20260810)
    q = torch.randn(1, 3, 1, 2)
    k = torch.randn(1, 3, 1, 2)
    v = torch.randn(1, 3, 2, 3)
    g = -torch.rand(1, 3, 2)
    beta = torch.sigmoid(torch.randn(1, 3, 2))
    initial = torch.randn(1, 2, 3, 2)

    actual, final = torch_chunk_gated_delta_rule(q, k, v, g, beta, initial, True, use_qk_l2norm_in_kernel=True)

    state = initial[0].float().clone()
    expected = torch.empty_like(v)
    for token in range(3):
        query = q[0, token, 0].float()
        key = k[0, token, 0].float()
        query = query / torch.sqrt(torch.sum(query * query) + 1.0e-6)
        key = key / torch.sqrt(torch.sum(key * key) + 1.0e-6)
        for head in range(2):
            state[head] *= torch.exp(g[0, token, head])
            residual = v[0, token, head] - state[head] @ key
            residual *= beta[0, token, head]
            state[head] += residual[:, None] * key[None, :]
            expected[0, token, head] = state[head] @ (query * (2**-0.5))

    torch.testing.assert_close(actual, expected, rtol=0, atol=1.0e-6)
    torch.testing.assert_close(final[0], state, rtol=0, atol=1.0e-6)
