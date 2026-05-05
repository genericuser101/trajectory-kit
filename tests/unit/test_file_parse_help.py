"""
tests/unit/test_file_parse_help.py
==================================
Unit tests for trajectory_kit._file_parse_help.

Functions under test
--------------------
resolve_frame_interval   pure function — no fixture needed
iter_records             file-based — tested via tmp_path text fixtures
iter_records_sample      file-based — tested via tmp_path text fixtures
"""

from __future__ import annotations

from pathlib import Path

import pytest

from trajectory_kit._file_parse_help import (
    iter_records,
    iter_records_sample,
    resolve_frame_interval,
)


# ===========================================================================
# resolve_frame_interval — pure, no fixtures
# ===========================================================================

class TestResolveFrameInterval:

    def test_empty_returns_full_default(self):
        assert resolve_frame_interval(()) == (0, None, 1)

    def test_none_returns_full_default(self):
        # falsy check at top of function — None should also yield the default
        assert resolve_frame_interval(None) == (0, None, 1)

    def test_two_tuple_inclusive_to_exclusive(self):
        # Public contract: stop is inclusive on input, +1 to make Python's
        # exclusive stop.
        assert resolve_frame_interval((0, 10)) == (0, 11, 1)

    def test_three_tuple_with_step(self):
        assert resolve_frame_interval((0, 10, 2)) == (0, 11, 2)

    def test_open_low(self):
        assert resolve_frame_interval((None, 10)) == (0, 11, 1)

    def test_open_high(self):
        # When stop is None it must remain None, not become "None+1".
        assert resolve_frame_interval((5, None)) == (5, None, 1)

    def test_step_one_explicit(self):
        assert resolve_frame_interval((2, 5, 1)) == (2, 6, 1)

    def test_zero_step_raises(self):
        with pytest.raises(ValueError, match="step must be"):
            resolve_frame_interval((0, 10, 0))

    def test_negative_step_raises(self):
        with pytest.raises(ValueError, match="step must be"):
            resolve_frame_interval((0, 10, -1))

    def test_one_element_raises(self):
        with pytest.raises(ValueError, match="2 or 3 elements"):
            resolve_frame_interval((5,))

    def test_four_element_raises(self):
        with pytest.raises(ValueError, match="2 or 3 elements"):
            resolve_frame_interval((0, 10, 1, 5))


# ===========================================================================
# iter_records — counted mode
# ===========================================================================

@pytest.fixture
def counted_file(tmp_path) -> Path:
    """A file with a header line followed by N data rows."""
    p = tmp_path / "counted.txt"
    p.write_text(
        "noise line one\n"
        "       3 !COUNT\n"
        "alpha 1\n"
        "beta 2\n"
        "gamma 3\n"
        "trailing line\n"
    )
    return p


class TestIterRecordsCounted:

    def test_yields_n_records(self, counted_file):
        rows = list(iter_records(
            counted_file, mode="counted",
            header_pred=lambda line: "!COUNT" in line,
            count_from_header=lambda h: int(h.split()[0]),
            parse_row=lambda line, i: (i, line.strip()),
        ))
        assert len(rows) == 3

    def test_indices_start_at_zero(self, counted_file):
        rows = list(iter_records(
            counted_file, mode="counted",
            header_pred=lambda line: "!COUNT" in line,
            count_from_header=lambda h: int(h.split()[0]),
            parse_row=lambda line, i: i,
        ))
        assert rows == [0, 1, 2]

    def test_start_index_offset(self, counted_file):
        rows = list(iter_records(
            counted_file, mode="counted",
            header_pred=lambda line: "!COUNT" in line,
            count_from_header=lambda h: int(h.split()[0]),
            parse_row=lambda line, i: i,
            start_index=10,
        ))
        assert rows == [10, 11, 12]

    def test_parse_row_receives_line(self, counted_file):
        rows = list(iter_records(
            counted_file, mode="counted",
            header_pred=lambda line: "!COUNT" in line,
            count_from_header=lambda h: int(h.split()[0]),
            parse_row=lambda line, i: line.split()[0],
        ))
        assert rows == ["alpha", "beta", "gamma"]

    def test_missing_header_raises(self, tmp_path):
        p = tmp_path / "no_header.txt"
        p.write_text("just text\nno marker\n")
        with pytest.raises(ValueError, match="Header not found"):
            list(iter_records(
                p, mode="counted",
                header_pred=lambda line: "!COUNT" in line,
                count_from_header=lambda h: 0,
                parse_row=lambda line, i: line,
            ))

    def test_counted_requires_predicates(self, counted_file):
        with pytest.raises(ValueError, match="counted mode requires"):
            list(iter_records(
                counted_file, mode="counted",
                parse_row=lambda line, i: line,
            ))


# ===========================================================================
# iter_records — predicate mode
# ===========================================================================

@pytest.fixture
def predicate_file(tmp_path) -> Path:
    """A file mixing 'KEEP' and other lines."""
    p = tmp_path / "predicate.txt"
    p.write_text(
        "REMARK ignore me\n"
        "KEEP one\n"
        "REMARK ignore again\n"
        "KEEP two\n"
        "KEEP three\n"
        "end\n"
    )
    return p


class TestIterRecordsPredicate:

    def test_yields_only_matching_lines(self, predicate_file):
        rows = list(iter_records(
            predicate_file, mode="predicate",
            record_pred=lambda line: line.startswith("KEEP"),
            parse_row=lambda line, i: line.strip(),
        ))
        assert rows == ["KEEP one", "KEEP two", "KEEP three"]

    def test_indices_only_count_matched(self, predicate_file):
        rows = list(iter_records(
            predicate_file, mode="predicate",
            record_pred=lambda line: line.startswith("KEEP"),
            parse_row=lambda line, i: i,
        ))
        assert rows == [0, 1, 2]

    def test_predicate_requires_record_pred(self, predicate_file):
        with pytest.raises(ValueError, match="predicate mode requires"):
            list(iter_records(
                predicate_file, mode="predicate",
                parse_row=lambda line, i: line,
            ))

    def test_invalid_mode_raises(self, predicate_file):
        with pytest.raises(ValueError, match="counted.*predicate"):
            list(iter_records(
                predicate_file, mode="bogus",
                parse_row=lambda line, i: line,
            ))


# ===========================================================================
# iter_records_sample — Bernoulli sampler with target size
# ===========================================================================

class TestIterRecordsSampleFullScan:
    """When file has fewer lines than target_sample_size, we get the
    full-sample (probability 1) path."""

    def test_empty_file_returns_zeros(self, tmp_path):
        p = tmp_path / "empty.txt"
        p.write_text("")
        info = iter_records_sample(
            p,
            record_pred=lambda l: True,
            parse_row=lambda l, i: l,
        )
        assert info["number_of_lines"] == 0
        assert info["number_of_sampled_lines"] == 0
        assert info["sampled_records"] == []

    def test_short_file_full_sample(self, tmp_path):
        p = tmp_path / "short.txt"
        p.write_text("a\nb\nc\nd\n")
        info = iter_records_sample(
            p,
            record_pred=lambda l: True,
            parse_row=lambda l, i: l.strip(),
            target_sample_size=10,
        )
        assert info["number_of_lines"] == 4
        assert info["number_of_sampled_lines"] == 4
        assert info["sampling_metadata"]["sample_probability"] == 1.0
        assert info["sampled_records"] == ["a", "b", "c", "d"]

    def test_predicate_filters_records(self, tmp_path):
        p = tmp_path / "mixed.txt"
        p.write_text("KEEP 1\nDROP 1\nKEEP 2\nDROP 2\n")
        info = iter_records_sample(
            p,
            record_pred=lambda l: l.startswith("KEEP"),
            parse_row=lambda l, i: l.strip(),
            target_sample_size=10,
        )
        # All 4 lines sampled, but only 2 are eligible
        assert info["number_of_sampled_lines"] == 4
        assert info["number_of_sampled_eligible_records"] == 2
        assert info["sampled_records"] == ["KEEP 1", "KEEP 2"]


class TestIterRecordsSampleSubsample:
    """When file is larger than target, we get the geometric-skip path.
    Tests are statistical but bounded by a fixed seed."""

    def test_long_file_subsamples(self, tmp_path):
        p = tmp_path / "long.txt"
        p.write_text("".join(f"row{k}\n" for k in range(1000)))
        info = iter_records_sample(
            p,
            record_pred=lambda l: True,
            parse_row=lambda l, i: l.strip(),
            target_sample_size=100,
            rng_seed=42,
        )
        assert info["number_of_lines"] == 1000
        # With p ≈ 0.1 and seed=42, sampled count is approximately 100,
        # but the geometric process is not exact — give a generous band.
        assert 50 <= info["number_of_sampled_lines"] <= 200
        assert info["sampling_metadata"]["sample_probability"] == 0.1

    def test_seed_makes_run_deterministic(self, tmp_path):
        p = tmp_path / "long.txt"
        p.write_text("".join(f"row{k}\n" for k in range(1000)))
        a = iter_records_sample(
            p, record_pred=lambda l: True,
            parse_row=lambda l, i: l.strip(),
            target_sample_size=100, rng_seed=123,
        )
        b = iter_records_sample(
            p, record_pred=lambda l: True,
            parse_row=lambda l, i: l.strip(),
            target_sample_size=100, rng_seed=123,
        )
        assert a["sampled_records"] == b["sampled_records"]
