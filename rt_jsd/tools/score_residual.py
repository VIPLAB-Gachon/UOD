"""Detached final-score residuals with six original class-probability inputs.

Stage2 keeps the stage1 sparse targets, BCE objective and final-score offset.
Only detached original class probabilities are appended to the twenty detached
boundary statistics and original objectness. Six new hidden-input columns start
at zero; the original first21 weights and bias and constructor RNG consumption
match the fresh stage1 model exactly. No trained weights are loaded.

Training uses finite logit(clamp(original_score)) + delta. Inference preserves
all original scores exactly when delta is zero, including fixed zero/one
endpoints. Six independent sigmoid class scores need not sum to one, and their
class-specific corrections may change the final predicted class argmax.
"""
from __future__ import annotations

import argparse
import hashlib
import json

import torch
from torch import Tensor, nn
import torch.nn.functional as F


FLOAT32_MIN_PROBABILITY = torch.finfo(torch.float32).tiny
FLOAT32_MAX_PROBABILITY = 1.0 - torch.finfo(torch.float32).eps


def finite_old_logit(old_probability: Tensor) -> Tensor:
    """Detached FP32 finite-logit surrogate, matching the previous residual."""
    probability = old_probability.detach().float().clamp(
        min=FLOAT32_MIN_PROBABILITY, max=FLOAT32_MAX_PROBABILITY
    )
    return probability.log() - torch.log1p(-probability)


@torch.no_grad()
def corrected_probability(old_probability: Tensor, delta: Tensor) -> Tensor:
    """Stable odds correction with exact zero-residual and endpoint identity."""
    if old_probability.shape != delta.shape:
        raise ValueError("old_probability and delta must have the same shape")
    if old_probability.device != delta.device:
        raise ValueError("old_probability and delta must share a device")
    probability = old_probability.detach().float()
    offset = delta.detach().float()
    corrected = (finite_old_logit(probability) + offset).sigmoid()
    interior = (probability > 0) & (probability < 1)
    actual_logit = probability.log() - torch.log1p(-probability)
    corrected = torch.where(interior, (actual_logit + offset).sigmoid(), corrected)
    corrected = torch.where(probability == 0, probability, corrected)
    corrected = torch.where(probability == 1, probability, corrected)
    return torch.where(offset == 0, probability, corrected)


@torch.no_grad()
def clamp_report(original_scores: Tensor) -> dict:
    """Count score entries, not sixfold-duplicated candidate objects."""
    value = original_scores.detach().float()
    finite = torch.isfinite(value)
    valid = finite & (value >= 0) & (value <= 1)
    return dict(
        score_entries=int(value.numel()),
        candidate_rows=int(value.shape[0]) if value.ndim == 2 else None,
        classes=int(value.shape[1]) if value.ndim == 2 else None,
        zero_endpoints=int((value == 0).sum()),
        one_endpoints=int((value == 1).sum()),
        clamped_low=int((valid & (value < FLOAT32_MIN_PROBABILITY)).sum()),
        clamped_high=int((valid & (value > FLOAT32_MAX_PROBABILITY)).sum()),
        invalid=int((~valid).sum()),
        lower_bound=FLOAT32_MIN_PROBABILITY,
        upper_bound=FLOAT32_MAX_PROBABILITY,
        inference_endpoints_preserved=True,
        training_endpoint_surrogate_is_finite=True,
    )


def _validate_features(features: Tensor, objectness: Tensor) -> None:
    if features.ndim != 2 or features.shape[1] != 20:
        raise ValueError("features must have shape [candidates, 20]")
    if objectness.ndim != 1 or len(objectness) != len(features):
        raise ValueError("objectness must have shape [candidates]")
    if features.device != objectness.device:
        raise ValueError("features and objectness must share a device")
    if not features.is_floating_point() or not objectness.is_floating_point():
        raise TypeError("features and objectness must be floating-point tensors")


def _validate_scores(features: Tensor, objectness: Tensor, scores: Tensor) -> None:
    _validate_features(features, objectness)
    if scores.ndim != 2 or scores.shape != (len(features), 6):
        raise ValueError("original final scores must have shape [candidates, 6]")
    if scores.device != features.device:
        raise ValueError("scores and feature inputs must share a device")
    if not scores.is_floating_point():
        raise TypeError("scores must contain floating-point probabilities")


def _validate_class_probabilities(features: Tensor, class_probabilities: Tensor) -> None:
    if class_probabilities.ndim != 2 or class_probabilities.shape != (len(features), 6):
        raise ValueError("class_probabilities must have shape [candidates, 6]")
    if class_probabilities.device != features.device:
        raise ValueError("class_probabilities and features must share a device")
    if not class_probabilities.is_floating_point():
        raise TypeError("class_probabilities must contain floating-point probabilities")


class FinalClassScoreMLP(nn.Module):
    """Shared 27 -> 64(ReLU) -> 6 signed final-score logit correction.

    Input order is twenty distribution statistics, original objectness, then
    six original class probabilities. The four public inputs detach. Caller
    seed 2026 initializes the original first21 weights and bias exactly as the
    stage1 fresh model; six added hidden-input columns are zero but trainable.
    Output weights/bias are zero. Widening preserves the stage1 constructor's
    caller RNG consumption and does not load any trained checkpoint.
    """

    def __init__(self) -> None:
        super().__init__()
        # Consume caller RNG in exactly the original stage1 order.
        original_hidden = nn.Linear(21, 64)
        output = nn.Linear(64, 6)
        # The new storage needs no additional draws from the caller RNG stream.
        with torch.random.fork_rng(devices=[]):
            widened = nn.Linear(27, 64)
        with torch.no_grad():
            widened.weight.zero_()
            widened.weight[:, :21].copy_(original_hidden.weight)
            widened.bias.copy_(original_hidden.bias)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        self.net = nn.Sequential(widened, nn.ReLU(), output)

    def delta(self, features: Tensor, objectness: Tensor, class_probabilities: Tensor) -> Tensor:
        _validate_features(features, objectness)
        _validate_class_probabilities(features, class_probabilities)
        joined = torch.cat(
            (features.detach().float(), objectness.detach().float().unsqueeze(-1),
             class_probabilities.detach().float()), dim=-1,
        )
        with torch.autocast(device_type=joined.device.type, enabled=False):
            return self.net(joined)

    def logits(self, features: Tensor, objectness: Tensor, scores: Tensor,
               class_probabilities: Tensor) -> Tensor:
        _validate_scores(features, objectness, scores)
        return finite_old_logit(scores) + self.delta(features, objectness, class_probabilities)

    def forward(self, features: Tensor, objectness: Tensor, scores: Tensor,
                class_probabilities: Tensor) -> Tensor:
        return self.logits(features, objectness, scores, class_probabilities)

    @torch.no_grad()
    def probability(self, features: Tensor, objectness: Tensor, scores: Tensor,
                    class_probabilities: Tensor) -> Tensor:
        _validate_scores(features, objectness, scores)
        return corrected_probability(scores, self.delta(features, objectness, class_probabilities))


def _state_digest(values) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str((array.shape, str(array.dtype))).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _self_test() -> dict:
    """CPU tests of controlled widening, conditioning and input isolation."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2026)
        model = FinalClassScoreMLP()
        rng_after_new = torch.random.get_rng_state().clone()
        initial_digest = _state_digest(model.state_dict())
        torch.manual_seed(2026)
        # The previous 21-input model's constructor is fully captured here:
        # Linear(21,64), ReLU, Linear(64,6), then zero the output parameters.
        # This independent fixture keeps the widening/RNG check portable and
        # never imports an earlier experiment or reads a trained checkpoint.
        prior = nn.Sequential(nn.Linear(21, 64), nn.ReLU(), nn.Linear(64, 6))
        nn.init.zeros_(prior[-1].weight)
        nn.init.zeros_(prior[-1].bias)
        rng_after_prior = torch.random.get_rng_state().clone()
        assert torch.equal(rng_after_new, rng_after_prior)
        assert torch.equal(model.net[0].weight[:, :21], prior[0].weight)
        assert torch.equal(model.net[0].bias, prior[0].bias)
        assert torch.count_nonzero(model.net[0].weight[:, 21:]) == 0
        assert torch.equal(model.net[-1].weight, prior[-1].weight)
        assert torch.equal(model.net[-1].bias, prior[-1].bias)
        assert sum(p.numel() for p in model.parameters()) == 2182
        torch.manual_seed(2026)
        twin = FinalClassScoreMLP()
        assert all(torch.equal(value, twin.state_dict()[key]) for key, value in model.state_dict().items())
        before_rng = torch.random.get_rng_state().clone()
        FinalClassScoreMLP()
        assert not torch.equal(before_rng, torch.random.get_rng_state())

        n = 10
        features = torch.randn(n, 20, requires_grad=True)
        objectness = torch.linspace(.01, .99, n, requires_grad=True)
        class_probabilities = torch.rand(n, 6, requires_grad=True)
        scores = torch.linspace(.001, .9, n * 6).reshape(n, 6)
        scores[0, 0] = 0.0
        scores[-1, -1] = 1.0
        scores[0, 1] = 1e-35
        scores.requires_grad_(True)
        delta = model.delta(features, objectness, class_probabilities)
        assert delta.shape == (n, 6) and torch.equal(delta, torch.zeros_like(delta))
        predicted = model.probability(features, objectness, scores, class_probabilities)
        assert predicted.dtype == torch.float32 and torch.equal(predicted, scores.detach())
        assert not predicted.requires_grad
        assert torch.equal(predicted.argmax(1), scores.detach().argmax(1))
        assert torch.isfinite(model(features, objectness, scores, class_probabilities)).all()
        empty = (torch.empty(0, 20), torch.empty(0), torch.empty(0, 6), torch.empty(0, 6))
        assert model(*empty).shape == (0, 6) and model.probability(*empty).shape == (0, 6)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            assert torch.equal(model.probability(features, objectness, scores, class_probabilities), scores.detach())
            assert model(features, objectness, scores, class_probabilities).dtype == torch.float32

        offsets = torch.tensor([-2., -1., -.5, .5, 1., 2.])
        with torch.no_grad():
            model.net[-1].bias.copy_(offsets)
        adjusted = model.probability(features, objectness, scores, class_probabilities)
        interior = (scores.detach() > 0) & (scores.detach() < 1)
        direction = offsets.expand(n, -1)
        assert (adjusted[interior & (direction < 0)] < scores.detach()[interior & (direction < 0)]).all()
        assert (adjusted[interior & (direction > 0)] > scores.detach()[interior & (direction > 0)]).all()
        assert adjusted[0, 0].item() == 0 and adjusted[-1, -1].item() == 1
        reference = (torch.logit(scores.detach().double()) + offsets.double()).sigmoid().float()
        torch.testing.assert_close(adjusted, reference, rtol=3e-6, atol=1e-7)
        for sign in (-1., 1.):
            extreme = corrected_probability(scores, torch.full_like(scores, sign * 1e6))
            assert torch.isfinite(extreme).all() and ((extreme >= 0) & (extreme <= 1)).all()
            assert extreme[0, 0].item() == 0 and extreme[-1, -1].item() == 1
        with torch.no_grad():
            model.net[-1].bias.zero_()

        # Isolate the new input route while holding old score offsets fixed.
        # This fixture uses manual weights, not a trained checkpoint.
        with torch.random.fork_rng(devices=[]):
            fixture = FinalClassScoreMLP()
        with torch.no_grad():
            for parameter in fixture.parameters():
                parameter.zero_()
            fixture.net[0].weight[0, 21 + 2] = 1.0
            fixture.net[-1].weight[4, 0] = 2.0
        class_a = torch.zeros(1, 6)
        class_b = class_a.clone(); class_b[0, 2] = .5
        f = torch.zeros(1, 20); obj = torch.full((1,), .2)
        a = fixture.delta(f, obj, class_a)
        b = fixture.delta(f, obj, class_b)
        assert torch.equal(a, torch.zeros(1, 6))
        expected = torch.zeros(1, 6); expected[0, 4] = 1.0
        torch.testing.assert_close(b, expected, rtol=0, atol=0)
        same_offsets = torch.full((1, 6), .1)
        qa = fixture.probability(f, obj, same_offsets, class_a)
        qb = fixture.probability(f, obj, same_offsets, class_b)
        assert qb[0, 4] > qa[0, 4] and torch.equal(qb[0, :4], qa[0, :4])
        assert qb[0, 5] == qa[0, 5]

        fixture_clamps = clamp_report(scores)
        assert fixture_clamps["score_entries"] == n * 6 and fixture_clamps["candidate_rows"] == n
        assert fixture_clamps["classes"] == 6
        assert fixture_clamps["zero_endpoints"] == fixture_clamps["one_endpoints"] == 1
        assert fixture_clamps["clamped_low"] == fixture_clamps["clamped_high"] == 1
        assert fixture_clamps["invalid"] == 0

        target = torch.linspace(0., 1., n * 6).reshape(n, 6)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(
                model(features, objectness, scores, class_probabilities), target)
            loss.backward()
            assert all(value.grad is None for value in (features, objectness, scores, class_probabilities))
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            assert model.net[-1].weight.grad.abs().sum().item() > 0
            if step == 0:
                assert model.net[0].weight.grad.abs().sum().item() == 0
            else:
                assert model.net[0].weight.grad.abs().sum().item() > 0
                assert model.net[0].weight.grad[:, 21:].abs().sum().item() > 0
            optimizer.step()
        assert model.net[0].weight[:, 21:].detach().abs().sum().item() > 0

        for bad in ((features[:, :19], objectness, scores, class_probabilities),
                    (features, objectness[:, None], scores, class_probabilities),
                    (features, objectness, scores[:, :5], class_probabilities),
                    (features, objectness, scores[0], class_probabilities),
                    (features, objectness, scores, class_probabilities[:, :5]),
                    (features, objectness, scores, class_probabilities[0]),
                    (features, objectness, scores, class_probabilities.long()),
                    (features.long(), objectness, scores, class_probabilities),
                    (features, objectness, scores.long(), class_probabilities)):
            try:
                model(*bad)
            except (ValueError, TypeError):
                pass
            else:
                raise AssertionError("Invalid input shape or dtype accepted")

    return dict(status="passed", device="cpu", parameters=2182,
        architecture=[27, 64, 6], activation="ReLU", seed_for_fixture=2026,
        initial_state_digest=initial_digest,
        initial_hidden_first21_columns_and_bias_equal_stage1=True,
        extra_six_hidden_input_columns_zero_and_trainable=True,
        constructor_RNG_consumption_equal_stage1=True,
        original_class_probabilities_are_module_inputs=True,
        training_clamp_fixture=fixture_clamps,
        tests=["paired_initialization_and_caller_RNG", "stage1_first21_weight_and_bias_identity",
            "constructor_RNG_state_matches_stage1", "new_six_columns_initially_zero",
            "all_six_scores_bitwise_identity_at_zero_residual", "signed_per_class_logit_corrections",
            "class_probability_input_conditioning_fixture", "zero_one_endpoints_and_extreme_offsets",
            "finite_training_logits_and_clamp_report", "empty_Nx6_output", "FP32_inside_outer_autocast",
            "all_four_inputs_detached", "new_zero_columns_learn_after_two_updates",
            "invalid_shapes_and_dtypes_rejected"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test:
        parser.error("Use --self-test for the standalone CPU checks")
    print(json.dumps(_self_test(), sort_keys=True))


if __name__ == "__main__":
    main()
