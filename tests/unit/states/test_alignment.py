from __future__ import annotations

from dataclasses import replace
from math import log, nan

import pytest

from market_regime_engine.models.artifacts import GaussianHMMArtifact
from market_regime_engine.states.alignment import (
    ALIGNMENT_AMBIGUITY_ABS_TOLERANCE,
    StateAlignment,
    StateAlignmentAmbiguityError,
    align_first_fold,
    align_to_reference,
)
from market_regime_engine.states.signatures import (
    rms_distance,
    signature_sort_key,
    state_signatures,
)


def artifact(
    means: tuple[tuple[float, ...], ...],
    covariances: tuple[tuple[tuple[float, ...], ...], ...],
) -> GaussianHMMArtifact:
    state_count = len(means)
    dimension = len(means[0])
    if state_count == 2:
        transition = ((0.9, 0.1), (0.2, 0.8))
        start = (0.6, 0.4)
    elif state_count == 3:
        transition = (
            (0.8, 0.1, 0.1),
            (0.1, 0.8, 0.1),
            (0.1, 0.1, 0.8),
        )
        start = (0.4, 0.3, 0.3)
    else:
        raise AssertionError("test helper only needs K=2 or K=3")
    return GaussianHMMArtifact(
        state_count=state_count,
        feature_order=tuple(f"f{index}" for index in range(dimension)),
        start_probabilities=start,
        transition_matrix=transition,
        means=means,
        full_covariances=covariances,
    )


def test_signature_exact_components_and_rms() -> None:
    model = artifact(
        means=((1.0, -2.0), (3.0, 4.0)),
        covariances=(
            ((4.0, 3.0), (3.0, 9.0)),
            ((1.0, 0.0), (0.0, 1.0)),
        ),
    )
    signatures = state_signatures(model)
    assert signatures[0] == pytest.approx((1.0, -2.0, log(2.0), log(3.0), 0.5))
    assert signatures[1] == pytest.approx((3.0, 4.0, 0.0, 0.0, 0.0))
    assert rms_distance(signatures[0], signatures[0]) == 0.0
    assert signature_sort_key((1.00000000004, -2.0)) == (1.0, -2.0)


def test_first_fold_uses_rounded_lexicographic_sort_and_canonical_ids() -> None:
    model = artifact(
        means=((2.0,), (-1.0,)),
        covariances=(((1.0,),), ((1.0,),)),
    )
    aligned = align_first_fold(model)
    assert aligned.persistent_state_ids == ("state_0", "state_1")
    assert aligned.persistent_to_fitted == (1, 0)
    assert aligned.aligned_signatures[0][0] == -1.0
    assert aligned.matched_rms == (0.0, 0.0)
    assert aligned.max_drift == 0.0
    assert aligned.initial_alignment is True


def test_first_fold_identical_rounded_keys_are_ambiguous() -> None:
    model = artifact(
        means=((1.000000000001,), (1.000000000002,)),
        covariances=(((1.0,),), ((1.0,),)),
    )
    with pytest.raises(StateAlignmentAmbiguityError, match="rounded-10-decimal"):
        align_first_fold(model)


def test_later_fold_exhaustive_mapping_recovers_swapped_states() -> None:
    reference_model = artifact(
        means=((-2.0,), (3.0,)),
        covariances=(((1.0,),), ((4.0,),)),
    )
    reference = align_first_fold(reference_model).aligned_signatures
    swapped = artifact(
        means=((3.0,), (-2.0,)),
        covariances=(((4.0,),), ((1.0,),)),
    )
    aligned = align_to_reference(swapped, reference)
    assert aligned.persistent_to_fitted == (1, 0)
    assert aligned.total_cost == pytest.approx(0.0)
    assert aligned.max_drift == pytest.approx(0.0)
    assert aligned.initial_alignment is False


def test_equal_best_and_second_best_cost_is_ambiguous() -> None:
    reference = ((-1.0, 0.0), (1.0, 0.0))
    current = artifact(
        means=((0.0,), (0.0,)),
        covariances=(((1.0,),), ((1.0,),)),
    )
    assert ALIGNMENT_AMBIGUITY_ABS_TOLERANCE == 1e-10
    with pytest.raises(StateAlignmentAmbiguityError, match="1e-10"):
        align_to_reference(current, reference)


def test_large_unique_drift_is_recorded_but_not_gated() -> None:
    reference_model = artifact(
        means=((-1.0,), (1.0,)),
        covariances=(((1.0,),), ((1.0,),)),
    )
    reference = align_first_fold(reference_model).aligned_signatures
    far_model = artifact(
        means=((-100.0,), (200.0,)),
        covariances=(((1.0,),), ((1.0,),)),
    )
    aligned = align_to_reference(far_model, reference)
    assert aligned.max_drift > 99.0
    assert aligned.total_cost > aligned.max_drift


def test_reference_shape_and_signature_helpers_fail_closed() -> None:
    model = artifact(
        means=((-1.0,), (1.0,)),
        covariances=(((1.0,),), ((1.0,),)),
    )
    with pytest.raises(ValueError, match="count"):
        align_to_reference(model, ((0.0, 0.0),))
    with pytest.raises(ValueError, match="dimension"):
        align_to_reference(model, ((0.0,), (1.0, 2.0)))
    with pytest.raises(ValueError, match="same non-zero"):
        rms_distance((1.0,), (1.0, 2.0))
    with pytest.raises(ValueError, match="finite"):
        rms_distance((nan,), (1.0,))
    with pytest.raises(ValueError, match="non-empty"):
        signature_sort_key(())
    with pytest.raises(ValueError, match="finite"):
        signature_sort_key((nan,))


def test_alignment_evidence_validation_paths() -> None:
    valid = StateAlignment(
        persistent_state_ids=("state_0", "state_1"),
        persistent_to_fitted=(0, 1),
        aligned_signatures=((0.0,), (1.0,)),
        matched_rms=(0.0, 0.0),
        total_cost=0.0,
        max_drift=0.0,
        initial_alignment=True,
    )
    cases = (
        ({"persistent_state_ids": ("bad", "state_1")}, "canonical"),
        ({"persistent_to_fitted": (0, 0)}, "one-to-one"),
        ({"aligned_signatures": ((0.0,),)}, "one entry"),
        ({"matched_rms": (0.0,)}, "one entry"),
        ({"matched_rms": (-1.0, 1.0)}, "finite and non-negative"),
        ({"matched_rms": (nan, 0.0)}, "finite and non-negative"),
        ({"total_cost": -1.0}, "total cost"),
        ({"total_cost": nan}, "total cost"),
        ({"max_drift": -1.0}, "max drift"),
        ({"max_drift": nan}, "max drift"),
        ({"total_cost": 1.0}, "summed matched RMS"),
        ({"max_drift": 1.0}, "maximum matched RMS"),
    )
    for changes, match in cases:
        with pytest.raises(ValueError, match=match):
            replace(valid, **changes)
