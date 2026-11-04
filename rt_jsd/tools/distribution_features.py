"""Detached distribution features and the quality-only pilot's small MLP.

This module has no detector, dataset, matching, or score-filtering side effects.
The caller controls initialization RNG and supplies already decoded candidate
rows in their original order. Only the auxiliary MLP receives gradients.
"""
from __future__ import annotations

import argparse
import json
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class QualityMLP(nn.Module):
    """One shared 20 -> 64 -> 1 quality-logit predictor.

    The first layer uses PyTorch's ordinary caller-seeded initialization. A
    zero final weight makes the initial output exactly the declared prior,
    independent of candidate features. No RNG is reseeded inside this class.
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, math.log(0.01 / 0.99))

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 2 or features.shape[1] != 20:
            raise ValueError("QualityMLP expects a [candidates, 20] feature tensor")
        # Explicit detachment also protects a future noncached caller from
        # accidentally optimizing the detector through this auxiliary loss.
        return self.net(features.detach().float()).squeeze(-1)


@torch.no_grad()
def quality_features(
    raw_image: Tensor, bins: int = 25, chunk_size: int = 32768
) -> Tensor:
    """Return detached FP32 top-four probability statistics for each row.

    For L, T, R, B, concatenate the four largest softmax probabilities in
    descending order and their arithmetic mean, then flatten side-major to
    [N, 20]. Objectness and class columns after the 4*bins logits are ignored.
    Chunking does not alter the candidate-row order or the feature definition.
    """
    if not isinstance(bins, int) or isinstance(bins, bool) or bins < 4:
        raise ValueError("bins must be an integer of at least four")
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if raw_image.ndim != 2 or raw_image.shape[1] < 4 * bins:
        raise ValueError("raw_image must be [candidates, at least 4*bins logits]")
    if not raw_image.is_floating_point():
        raise TypeError("raw_image must contain floating-point logits")
    if raw_image.shape[0] == 0:
        return torch.empty((0, 20), dtype=torch.float32, device=raw_image.device)

    result = []
    # Disabling autocast explicitly keeps the cached statistics definition
    # stable if this function is called inside a base-detector AMP context.
    with torch.autocast(device_type=raw_image.device.type, enabled=False):
        for start in range(0, raw_image.shape[0], chunk_size):
            logits = raw_image[start : start + chunk_size, : 4 * bins]
            probability = logits.detach().float().reshape(-1, 4, bins).softmax(-1)
            top = probability.topk(4, dim=-1, largest=True, sorted=True).values
            result.append(torch.cat((top, top.mean(-1, keepdim=True)), dim=-1).flatten(1))
    return torch.cat(result, dim=0)


def _self_test() -> dict:
    """CPU-only checks of feature semantics and gradient isolation."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2026)
        n, bins = 7, 25
        row = torch.arange(1, n + 1, dtype=torch.float32)[:, None, None]
        side = torch.arange(1, 5, dtype=torch.float32)[None, :, None]
        position = torch.arange(2, bins + 2, dtype=torch.float32)[None, None, :]
        logits = torch.sin(row * side * position * 0.37) * (0.2 + row * 0.31)
        raw = torch.cat((logits.flatten(1), torch.randn(n, 7) * 1000), dim=1)
        raw.requires_grad_(True)

        actual = quality_features(raw)
        probability = raw.detach()[:, :100].reshape(n, 4, bins).softmax(-1)
        top_reference = probability.sort(dim=-1, descending=True).values[..., :4]
        reference = torch.cat((top_reference, top_reference.mean(-1, keepdim=True)), -1).flatten(1)
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        assert actual.shape == (n, 20) and actual.dtype == torch.float32
        assert not actual.requires_grad and actual.grad_fn is None
        for chunk in (1, 2, 3, 32768):
            torch.testing.assert_close(quality_features(raw, chunk_size=chunk), reference, rtol=0, atol=0)
        permutation = torch.tensor([6, 0, 4, 1, 5, 3, 2])
        torch.testing.assert_close(quality_features(raw[permutation], chunk_size=2), actual[permutation], rtol=0, atol=0)
        assert not torch.equal(actual[0], actual[-1]), "Row-mapping fixture must distinguish rows"

        changed_tail = raw.detach().clone()
        changed_tail[:, 100:] *= -7
        torch.testing.assert_close(quality_features(changed_tail), actual, rtol=0, atol=0)
        empty = quality_features(raw[:0])
        assert empty.shape == (0, 20) and empty.dtype == torch.float32
        half = quality_features(raw.detach().half(), chunk_size=2)
        half_reference = quality_features(raw.detach().half().float(), chunk_size=3)
        torch.testing.assert_close(half, half_reference, rtol=0, atol=0)
        assert half.dtype == torch.float32

        # Separate caller-seeded constructors must reproduce identical state.
        torch.manual_seed(2026)
        predictor = QualityMLP()
        torch.manual_seed(2026)
        twin = QualityMLP()
        assert all(torch.equal(v, twin.state_dict()[k]) for k, v in predictor.state_dict().items())
        before_rng = torch.random.get_rng_state().clone()
        QualityMLP()
        assert not torch.equal(before_rng, torch.random.get_rng_state()), "Constructor must use, not reseed, caller RNG"

        output = predictor(actual)
        assert output.shape == (n,) and torch.isfinite(output).all()
        torch.testing.assert_close(output.sigmoid(), torch.full((n,), 0.01), rtol=1e-6, atol=1e-9)
        assert predictor(torch.empty(0, 20)).shape == (0,)
        features_with_grad = actual.clone().requires_grad_(True)
        target = torch.linspace(0.1, 0.9, n)
        optimizer = torch.optim.Adam(predictor.parameters(), lr=0.001)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(predictor(features_with_grad), target)
            loss.backward()
            assert raw.grad is None and features_with_grad.grad is None
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in predictor.parameters())
            if step == 1:
                assert predictor.net[0].weight.grad.abs().sum() > 0
            optimizer.step()
        assert all(torch.isfinite(p).all() for p in predictor.parameters())

        for bad_raw, kwargs in ((raw[0], {}), (raw[:, :99], {}), (raw, {"bins": 3}), (raw, {"chunk_size": 0})):
            try:
                quality_features(bad_raw, **kwargs)
            except (ValueError, TypeError):
                pass
            else:
                raise AssertionError("Invalid feature input was accepted")

    return {
        "status": "passed",
        "device": "cpu",
        "tests": [
            "side_major_sorted_top4_and_top4_mean",
            "candidate_row_and_chunk_identity",
            "objectness_class_columns_ignored",
            "empty_and_half_inputs",
            "caller_controlled_reproducible_initialization",
            "initial_output_prior_0p01",
            "detached_inputs_and_quality_only_finite_backward",
            "invalid_shapes_and_configuration_rejected",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test:
        parser.error("Use --self-test for the standalone CPU checks")
    print(json.dumps(_self_test(), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
