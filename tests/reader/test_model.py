"""The reader computes its four lines, ignores padding, and its probe is one linear map."""
import math

import pytest
import torch

from check2.reader.model import OUTPUTS, VARIANTS, Reader

D = 16


def _reader(layers=(0, 1, 2), **kw) -> Reader:
    torch.manual_seed(0)
    reader = Reader(list(layers), D, **kw).eval()
    if reader.pooled and not reader.linear_probe:
        with torch.no_grad():
            reader.u.normal_()        # a non-uniform pooling, so the test sees the weights act
    return reader


def _window(n_layers: int, length: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, n_layers, length, D, generator=g)


def test_output_is_three_probabilities():
    reader = _reader()
    p = reader(_window(3, 5, 0), torch.ones(1, 5, dtype=torch.bool))
    assert p.shape == (1, len(OUTPUTS))
    assert torch.allclose(p.sum(-1), torch.ones(1))


def test_the_four_lines_computed_directly():
    reader = _reader()
    h = _window(3, 6, 1)[0]                                         # layers, W, d
    x = torch.cat([reader.projections[i](reader.norms[i](h[i])) for i in range(3)], dim=-1)
    a = torch.softmax(x @ reader.u / math.sqrt(64 * 3), dim=0)
    z = torch.cat([(a[:, None] * x).sum(0), x[-1]])
    expected = torch.softmax(reader.linear_2(torch.nn.functional.gelu(reader.linear_1(z))), dim=-1)
    got = reader(h[None], torch.ones(1, 6, dtype=torch.bool))[0]
    torch.testing.assert_close(got, expected, rtol=1e-6, atol=1e-7)


def test_padded_batch_equals_alone():
    reader = _reader()
    lengths = [3, 7, 5, 1]
    windows = [_window(3, n, 10 + i) for i, n in enumerate(lengths)]
    width = max(lengths)
    h = torch.full((len(lengths), 3, width, D), 123.0)               # padding holds large values
    mask = torch.zeros(len(lengths), width, dtype=torch.bool)
    for b, (w, n) in enumerate(zip(windows, lengths)):
        h[b, :, width - n:] = w[0]
        mask[b, width - n:] = True
    batched = reader(h, mask)
    for b, (w, n) in enumerate(zip(windows, lengths)):
        alone = reader(w, torch.ones(1, n, dtype=torch.bool))[0]
        # equal up to float32 rounding: a matrix product over another batch shape rounds differently
        torch.testing.assert_close(batched[b], alone, rtol=1e-6, atol=1e-7)


def test_padding_values_never_reach_the_output():
    reader = _reader()
    w = _window(3, 4, 3)
    mask = torch.tensor([[False, False, True, True, True, True]])
    outputs = []
    for fill in (0.0, 1e4, float("nan")):
        h = torch.full((1, 3, 6, D), fill)
        h[:, :, 2:] = w[0]
        outputs.append(reader(h, mask))
    assert torch.equal(outputs[0], outputs[1]) and torch.equal(outputs[0], outputs[2])


def test_linear_probe_is_one_linear_map():
    reader = _reader(layers=[18], linear_probe=True)
    assert [name for name, _ in reader.named_parameters()] == ["probe.weight", "probe.bias"]
    h = _window(1, 5, 4)
    mask = torch.ones(1, 5, dtype=torch.bool)
    logits = reader.logits(h, mask)
    torch.testing.assert_close(logits, h[:, 0, -1] @ reader.probe.weight.T + reader.probe.bias)
    # only the read token matters
    h2 = h.clone()
    h2[:, :, :-1] = 7.0
    assert torch.equal(reader.logits(h2, mask), logits)
    # and the map is linear: logits(a h + b g) - bias = a (logits(h) - bias) + b (logits(g) - bias)
    g = _window(1, 5, 5)
    bias = reader.probe.bias
    combined = reader.logits(2.0 * h - 3.0 * g, mask) - bias
    torch.testing.assert_close(combined, 2.0 * (logits - bias) - 3.0 * (reader.logits(g, mask) - bias))


def test_unpooled_reader_reads_the_read_token_alone():
    reader = _reader(pooled=False)
    h = _window(3, 5, 6)
    mask = torch.ones(1, 5, dtype=torch.bool)
    h2 = h.clone()
    h2[:, :, :-1] = -5.0
    assert torch.equal(reader(h, mask), reader(h2, mask))


def test_bf16_states_are_read_in_float32():
    reader = _reader()
    h = _window(3, 4, 7).to(torch.bfloat16)
    mask = torch.ones(1, 4, dtype=torch.bool)
    assert reader(h, mask).dtype == torch.float32
    assert torch.equal(reader(h, mask), reader(h.float(), mask))


def test_wrong_number_of_layers_is_refused():
    with pytest.raises(ValueError):
        _reader()(_window(2, 4, 8), torch.ones(1, 4, dtype=torch.bool))


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_each_variant_builds_at_full_width(name):
    reader = Reader(hidden_size=2560, **VARIANTS[name])
    n = sum(p.numel() for p in reader.parameters())
    assert n > 0
    if name == "six_layers":
        assert 1_000_000 < n < 1_200_000
