#!/usr/bin/env python3

# Copyright LLM.build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for job -> lineage row decomposition."""

import pytest

from gbserver.lineage.decompose import (
    JOB_METADATA_KEYS,
    LineageDecomposeError,
    LineageRowDraft,
    group_by_job,
    to_lineage_rows,
)


def identify(artifact: dict) -> str:
    """Stand-in for canonical_id: the artifact's name is its identifier."""
    return artifact["name"]


def artifact(name: str, **extra) -> dict:
    return {"name": name, **extra}


def job(job_id: str = "J", sources=None, targets=None, **extra) -> dict:
    entry: dict = {"job_id": job_id}
    if sources is not None:
        entry["sources"] = sources
    if targets is not None:
        entry["targets"] = targets
    entry.update(extra)
    return entry


def pairs(rows) -> list[tuple]:
    return [(row.source, row.target) for row in rows]


class TestTerminals:
    """source/target None is real information, not a missing value."""

    def test_no_targets_is_a_deletion_terminal(self):
        rows = to_lineage_rows(
            job(sources=[artifact("a"), artifact("b")], targets=[]), identify
        )
        assert pairs(rows) == [("a", None), ("b", None)]

    def test_no_sources_is_a_creation_terminal(self):
        rows = to_lineage_rows(job(sources=[], targets=[artifact("out")]), identify)
        assert pairs(rows) == [(None, "out")]

    def test_creation_with_several_outputs(self):
        rows = to_lineage_rows(job(targets=[artifact("x"), artifact("y")]), identify)
        assert pairs(rows) == [(None, "x"), (None, "y")]

    def test_missing_keys_behave_like_empty_lists(self):
        rows = to_lineage_rows(job(targets=[artifact("out")]), identify)
        assert pairs(rows) == [(None, "out")]


class TestFanShapes:
    """The prototype's cases 2b and 3, with no special-casing."""

    def test_many_sources_one_target(self):
        rows = to_lineage_rows(
            job(
                sources=[artifact("a"), artifact("b"), artifact("c")],
                targets=[artifact("out")],
            ),
            identify,
        )
        assert pairs(rows) == [("a", "out"), ("b", "out"), ("c", "out")]

    def test_one_source_many_targets(self):
        rows = to_lineage_rows(
            job(sources=[artifact("in")], targets=[artifact("x"), artifact("y")]),
            identify,
        )
        assert pairs(rows) == [("in", "x"), ("in", "y")]

    def test_one_to_one(self):
        rows = to_lineage_rows(
            job(sources=[artifact("in")], targets=[artifact("out")]), identify
        )
        assert pairs(rows) == [("in", "out")]


class TestCartesianCase:
    """N sources AND M targets -- what the prototype rejects outright.

    Its guard raises when ``len(sources) > 1 and len(targets) > 1``, but a
    granite.build target run with 3 inputs and 2 outputs is routine, and the W&B
    sink already records it today. Rejecting it would drop real lineage.
    """

    def test_three_inputs_two_outputs_emits_six_rows(self):
        rows = to_lineage_rows(
            job(
                sources=[artifact("i1"), artifact("i2"), artifact("i3")],
                targets=[artifact("o1"), artifact("o2")],
            ),
            identify,
        )
        assert len(rows) == 6
        assert pairs(rows) == [
            ("i1", "o1"),
            ("i1", "o2"),
            ("i2", "o1"),
            ("i2", "o2"),
            ("i3", "o1"),
            ("i3", "o2"),
        ]

    def test_does_not_raise(self):
        """The one behavioral divergence from the prototype, stated as a test."""
        rows = to_lineage_rows(
            job(
                sources=[artifact("a"), artifact("b")],
                targets=[artifact("x"), artifact("y")],
            ),
            identify,
        )
        assert len(rows) == 4

    def test_every_row_shares_the_job_id(self):
        rows = to_lineage_rows(
            job(
                job_id="run-7",
                sources=[artifact("a"), artifact("b")],
                targets=[artifact("x"), artifact("y")],
            ),
            identify,
        )
        assert {row.job_id for row in rows} == {"run-7"}


class TestRegrouping:
    """Why N*M is a projection and not a loss of information.

    This is the property that justifies dropping the guard: the flat rows still
    say which inputs and which outputs one execution had.
    """

    def test_inputs_and_outputs_are_recoverable(self):
        rows = to_lineage_rows(
            job(
                job_id="J",
                sources=[artifact("i1"), artifact("i2"), artifact("i3")],
                targets=[artifact("o1"), artifact("o2")],
            ),
            identify,
        )
        grouped = group_by_job(rows)
        assert grouped["J"]["sources"] == {"i1", "i2", "i3"}
        assert grouped["J"]["targets"] == {"o1", "o2"}

    def test_terminals_regroup_without_none(self):
        creation = to_lineage_rows(job("C", targets=[artifact("o")]), identify)
        deletion = to_lineage_rows(job("D", sources=[artifact("i")]), identify)
        grouped = group_by_job(creation + deletion)
        assert grouped["C"]["sources"] == set()
        assert grouped["C"]["targets"] == {"o"}
        assert grouped["D"]["sources"] == {"i"}
        assert grouped["D"]["targets"] == set()

    def test_several_jobs_stay_separate(self):
        rows = to_lineage_rows(
            job("J1", sources=[artifact("a")], targets=[artifact("x")]), identify
        ) + to_lineage_rows(
            job("J2", sources=[artifact("b")], targets=[artifact("y")]), identify
        )
        grouped = group_by_job(rows)
        assert set(grouped) == {"J1", "J2"}
        assert grouped["J1"]["sources"] == {"a"}
        assert grouped["J2"]["sources"] == {"b"}


class TestSelfLoop:
    """A job rewriting its own input -- legitimate for unversioned entities."""

    def test_same_artifact_in_and_out(self):
        rows = to_lineage_rows(
            job(sources=[artifact("tbl")], targets=[artifact("tbl")]), identify
        )
        assert pairs(rows) == [("tbl", "tbl")]


class TestMetadata:
    def test_metadata_is_copied_onto_every_row(self):
        rows = to_lineage_rows(
            job(
                sources=[artifact("a"), artifact("b")],
                targets=[artifact("x")],
                job_name="train",
                owner="someone",
                job_status="SUCCESS",
            ),
            identify,
        )
        for row in rows:
            assert row.metadata["job_name"] == "train"
            assert row.metadata["owner"] == "someone"
            assert row.metadata["job_status"] == "SUCCESS"

    def test_unknown_keys_are_not_carried(self):
        rows = to_lineage_rows(
            job(sources=[artifact("a")], targets=[artifact("x")], not_a_key="v"),
            identify,
        )
        assert "not_a_key" not in rows[0].metadata

    def test_absent_metadata_is_omitted_not_defaulted(self):
        rows = to_lineage_rows(
            job(sources=[artifact("a")], targets=[artifact("x")]), identify
        )
        assert rows[0].metadata == {"job_id": "J"}

    def test_each_row_owns_its_metadata(self):
        """Rows must not share one dict, or mutating one would edit them all."""
        rows = to_lineage_rows(
            job(
                sources=[artifact("a"), artifact("b")],
                targets=[artifact("x")],
                job_name="n",
            ),
            identify,
        )
        rows[0].metadata["job_name"] = "changed"
        assert rows[1].metadata["job_name"] == "n"

    def test_job_id_is_a_metadata_key(self):
        assert "job_id" in JOB_METADATA_KEYS


class TestFilters:
    """Partition filters are carried, though the traversal does not use them yet."""

    def test_filters_are_taken_from_the_artifacts(self):
        rows = to_lineage_rows(
            job(
                sources=[artifact("a", filter='{"dt":"2024"}')],
                targets=[artifact("x", filter='{"dt":"2025"}')],
            ),
            identify,
        )
        assert rows[0].source_filter == '{"dt":"2024"}'
        assert rows[0].target_filter == '{"dt":"2025"}'

    def test_absent_filter_is_none(self):
        rows = to_lineage_rows(
            job(sources=[artifact("a")], targets=[artifact("x")]), identify
        )
        assert rows[0].source_filter is None
        assert rows[0].target_filter is None


class TestArtifactsAreCarried:
    def test_source_and_target_artifacts_are_kept(self):
        source, target = artifact("a", type="model"), artifact("x", type="dataset")
        rows = to_lineage_rows(job(sources=[source], targets=[target]), identify)
        assert rows[0].source_artifact == source
        assert rows[0].target_artifact == target

    def test_terminal_rows_carry_only_one_side(self):
        rows = to_lineage_rows(job(targets=[artifact("x")]), identify)
        assert rows[0].source_artifact is None
        assert rows[0].target_artifact is not None


class TestRejectedInput:
    def test_missing_job_id_raises(self):
        with pytest.raises(LineageDecomposeError, match="job_id"):
            to_lineage_rows({"sources": [artifact("a")]}, identify)

    def test_empty_job_id_raises(self):
        with pytest.raises(LineageDecomposeError, match="job_id"):
            to_lineage_rows(job("", sources=[artifact("a")]), identify)

    def test_no_sources_and_no_targets_raises(self):
        with pytest.raises(LineageDecomposeError, match="neither sources nor targets"):
            to_lineage_rows(job(sources=[], targets=[]), identify)

    def test_identify_errors_propagate(self):
        """A rejected identifier must not be swallowed into a degraded row."""

        def failing(_artifact: dict) -> str:
            raise ValueError("bad identifier")

        with pytest.raises(ValueError, match="bad identifier"):
            to_lineage_rows(
                job(sources=[artifact("a")], targets=[artifact("x")]), failing
            )


class TestRowIdentity:
    def test_key_is_the_storage_unique_triple(self):
        row = LineageRowDraft(job_id="J", source="a", target="b")
        assert row.key() == ("J", "a", "b")

    def test_rows_of_one_job_have_distinct_keys(self):
        rows = to_lineage_rows(
            job(
                sources=[artifact("a"), artifact("b")],
                targets=[artifact("x"), artifact("y")],
            ),
            identify,
        )
        assert len({row.key() for row in rows}) == len(rows)
