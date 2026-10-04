"""Regression test for the L1 probe's kept-index/original-index confusion.

`recovery.forensics.build_target_day_leak` builds its ``date -> feature
vector`` map from the ORIGIN dates of the samples, so a sample whose target day
is not itself some sample's origin is genuinely unavailable and is dropped
rather than faked. Once ANY earlier sample has been dropped, the position of a
kept sample in the OUTPUT list (``j``) no longer equals its position in the
INPUT list (``i``).

The verification loop used ``targets[j]``, so every kept sample after the first
dropped one was audited against the WRONG target date. These tests pin the
correct indexing with fixtures whose missing target-date vector sits in the
MIDDLE of the sample list, guaranteeing ``j != i`` for the samples that follow.

These are unit tests on the probe builder only; no forensic experiment is
re-run and no forensic result artifact is touched.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agentic_forecaster.recovery import forensics as fx

SEQ_LEN, N_FEAT = 3, 2


def _days(n: int = 15) -> list[str]:
    return [str(d.date()) for d in pd.bdate_range("2020-01-01", periods=n)]


def _encode(origin_idx: list[int], n_days: int = 15) -> np.ndarray:
    """One sample per origin, whose every feature value IS its day index.

    The probe registers ``X[i, -1, :]`` under sample ``i``'s origin date, so
    encoding the DAY index (not the row index) keeps the vector a faithful
    identifier of its date even when the origin list skips days.
    """
    return np.stack(
        [np.full((SEQ_LEN, N_FEAT), float(day), dtype=np.float32) for day in origin_idx],
        axis=0,
    )


def test_l1_mid_list_drop_is_audited_against_its_own_target_date():
    """A target-day vector missing from the MIDDLE shifts every later index.

    ``days[5]`` and ``days[13]`` are used only as TARGETS, so no sample
    registers a feature vector for them and those two samples are dropped. The
    first drop is in the middle of the list, which is the case the old
    ``targets[j]`` indexing got wrong: for every later kept sample the output
    position ``j`` is smaller than the input position ``i``.
    """
    days = _days()
    origin_idx = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12]
    target_idx = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13]
    X = _encode(origin_idx)
    origins = [days[i] for i in origin_idx]
    targets = [days[i] for i in target_idx]
    y = np.zeros(len(origins), dtype=int)

    out = fx.build_target_day_leak(X, y, origins, targets)

    dropped = [i for i in range(len(origins)) if target_idx[i] not in origin_idx]
    assert dropped == [4, 11], "fixture must drop one sample mid-list"
    assert out["n_dropped"] == len(dropped)
    assert out["n_kept"] == len(origins) - len(dropped)

    keep = [i for i in range(len(origins)) if i not in dropped]
    assert [(j, i) for j, i in enumerate(keep) if j != i], "fixture must shift indices"

    for j, i in enumerate(keep):
        # the returned metadata is indexed by the ORIGINAL sample index
        assert out["origin_dates"][j] == origins[i]
        assert out["target_dates"][j] == targets[i]
        # the realised final timestep is THIS sample's target date...
        assert out["final_timestep_dates"][j] == targets[i]
        # ...and the appended vector really is that date's vector
        assert np.allclose(out["X_leaked"][j, -1, :], float(target_idx[i]))
        assert targets[i] > origins[i]
        if j != i:
            # the old buggy indexing would have audited a DIFFERENT date
            assert targets[j] != targets[i]

    # The verification loop must therefore have been driven by targets[i].
    # Reproduce the old buggy assertion and require it to fail on the first
    # post-drop sample: if it did not fail, this fixture would not discriminate.
    def _buggy_assertion(kept: list[int]) -> None:
        for j, i in enumerate(kept):
            t = targets[j]
            if out["final_timestep_dates"][j] != t:
                raise fx.L1AlignmentError(
                    f"final timestep {out['final_timestep_dates'][j]} != target_date {t}")

    with pytest.raises(fx.L1AlignmentError):
        _buggy_assertion(keep)


def test_l1_every_kept_row_still_carries_its_own_target_vector():
    """Independent of the helper above: assert the alignment directly.

    ``days[9]`` and ``days[5]`` are used only as targets and are not origins,
    so those samples are dropped from the middle and the end of the list.
    """
    days = _days()
    origin_idx = [0, 1, 2, 3, 4]
    target_idx = [1, 2, 9, 4, 5]
    X = _encode(origin_idx)
    origins = [days[i] for i in origin_idx]
    targets = [days[i] for i in target_idx]

    out = fx.build_target_day_leak(X, np.zeros(len(origins), int), origins, targets)

    assert out["n_dropped"] == 2
    assert out["n_kept"] == 3
    vec_by_date = {o: X[i, -1, :] for i, o in enumerate(origins)}
    for j, t in enumerate(out["target_dates"]):
        assert t in vec_by_date
        assert np.allclose(out["X_leaked"][j, -1, :], vec_by_date[t])
        assert out["final_timestep_dates"][j] == t