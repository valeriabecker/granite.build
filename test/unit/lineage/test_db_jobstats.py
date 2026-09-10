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

"""Tests for the sink that writes lineage into the local index.

Run against a real SQLite row storage, not a fake: the properties worth pinning
here -- that a re-ingest does not duplicate, that an already-recorded target is
skipped -- are enforced by the composite unique index and by presence queries, and
a fake would just re-implement (and possibly contradict) both.
"""

import os
import uuid as uuid_module

import pytest

from gbserver.lineage.db_jobstats import DBLineageStore, _row_from_draft
from gbserver.lineage.decompose import LineageRowDraft
from gbserver.storage.sqlite.storage_factory import SqliteStorageFactory

pytestmark = pytest.mark.skipif(
    os.environ.get("SKIP_SQL_ADMIN_TESTS", "False").lower() == "true",
    reason="Don't want to run this in CICD.",
)


@pytest.fixture(name="rows")
def rows_fixture():
    """A lineage row storage on a table unique to this test."""
    table = f"t_sink_{uuid_module.uuid4().hex[:8]}"
    return SqliteStorageFactory().create_lineage_row_storage(table_name=table)


@pytest.fixture(name="sink")
def sink_fixture(rows):
    return DBLineageStore(storage=rows)


def job(job_id: str, sources: list, targets: list, **metadata) -> dict:
    """A job entry in the shape the shared builders emit."""
    return {
        "job_id": job_id,
        "sources": sources,
        "targets": targets,
        **metadata,
    }


def artifact(name: str, uri: str = "", space: str = "sp") -> dict:
    return {"name": name, "uri": uri, "space_name": space}


LH_TABLE = "lh://prod/ns/tables/raw_tbl"
LH_MODEL = "lh://prod/ns/models/mdl_tbl/trained/v1"


class TestDecomposition:
    """One job entry becomes N*M rows sharing a job_id."""

    def test_one_input_one_output_is_one_row(self, sink, rows):
        sink._write_job(
            job("J1", [artifact("a", LH_TABLE)], [artifact("b", LH_MODEL)]),
            build_id="BLD",
            target_run_uuid="TR",
        )
        assert len(rows.get_rows_by_job("J1")) == 1

    def test_two_by_three_is_six_rows(self, sink, rows):
        sink._write_job(
            job(
                "J1",
                [artifact("i1", LH_TABLE), artifact("i2", "s3://b/i2")],
                [
                    artifact("o1", LH_MODEL),
                    artifact("o2", "s3://b/o2"),
                    artifact("o3", "s3://b/o3"),
                ],
            ),
            build_id="BLD",
            target_run_uuid="TR",
        )
        assert len(rows.get_rows_by_job("J1")) == 6

    def test_every_row_of_a_job_shares_its_job_id(self, sink, rows):
        sink._write_job(
            job(
                "J1",
                [artifact("i1", LH_TABLE), artifact("i2", "s3://b/i2")],
                [artifact("o1", LH_MODEL)],
            ),
            build_id="BLD",
            target_run_uuid="TR",
        )
        assert {r.job_id for r in rows.get_rows_by_job("J1")} == {"J1"}

    def test_a_creation_records_a_terminal_source(self, sink, rows):
        sink._write_job(
            job("J1", [], [artifact("b", LH_MODEL)]),
            build_id="BLD",
            target_run_uuid="TR",
        )
        stored = rows.get_rows_by_job("J1")
        assert len(stored) == 1
        assert stored[0].is_creation()

    def test_a_deletion_records_a_terminal_target(self, sink, rows):
        sink._write_job(
            job("J1", [artifact("a", LH_TABLE)], []),
            build_id="BLD",
            target_run_uuid="TR",
        )
        stored = rows.get_rows_by_job("J1")
        assert len(stored) == 1
        assert stored[0].is_deletion()

    def test_an_undecomposable_job_is_skipped_not_raised(self, sink, rows):
        # One unrecordable entry must not abort the rest of a build's lineage.
        sink._write_job(
            job("", [artifact("a", LH_TABLE)], []),
            build_id="BLD",
            target_run_uuid="TR",
        )
        assert rows.get_rows_by_build("BLD") == []


class TestIdempotence:
    """Re-ingesting the same lineage must not duplicate rows."""

    def test_writing_the_same_job_twice_does_not_duplicate(self, sink, rows):
        entry = job("J1", [artifact("a", LH_TABLE)], [artifact("b", LH_MODEL)])
        for _ in range(2):
            sink._write_job(
                entry,
                build_id="BLD",
                target_run_uuid="TR",
            )
        assert len(rows.get_rows_by_job("J1")) == 1

    def test_a_creation_row_is_not_duplicated_either(self, sink, rows):
        # The least visible case: terminal endpoints are stored as "" rather than
        # NULL precisely so the unique index still catches them.
        entry = job("J1", [], [artifact("b", LH_MODEL)])
        for _ in range(2):
            sink._write_job(
                entry,
                build_id="BLD",
                target_run_uuid="TR",
            )
        assert len(rows.get_rows_by_job("J1")) == 1


class TestDedupByPresence:
    def test_filter_unrecorded_reports_targets_with_no_rows(self, sink):
        assert sink.filter_unrecorded({"t1", "t2"}) == {"t1", "t2"}

    def test_filter_unrecorded_drops_a_recorded_target(self, sink):
        sink._write_job(
            job("J1", [artifact("a", LH_TABLE)], [artifact("b", LH_MODEL)]),
            build_id="BLD",
            target_run_uuid="t1",
        )
        assert sink.filter_unrecorded({"t1", "t2"}) == {"t2"}

    def test_expected_counts_is_ignored(self, sink):
        # It counts one W&B run per output artifact -- a shape that never equals an
        # N*M row count. Honouring it would report every target unrecorded forever.
        sink._write_job(
            job("J1", [artifact("a", LH_TABLE)], [artifact("b", LH_MODEL)]),
            build_id="BLD",
            target_run_uuid="t1",
        )
        assert sink.filter_unrecorded({"t1"}, expected_counts={"t1": 999}) == set()

    def test_an_empty_candidate_set_is_empty(self, sink):
        assert sink.filter_unrecorded(set()) == set()

    def test_a_query_failure_fails_open_and_reports_the_error(self, rows):
        # Fails OPEN here (every candidate re-recorded, which is idempotent) but
        # must invoke the callback -- the reconciler is what fails closed on it.
        # Without the callback this silently duplicates work instead of skipping.
        class Failing:
            def get_recorded_target_runs(self, target_run_uuids):
                raise RuntimeError("db is down")

        seen = []
        sink = DBLineageStore(storage=Failing())
        result = sink.filter_unrecorded({"t1"}, on_query_error=seen.append)
        assert result == {"t1"}
        assert len(seen) == 1

    def test_a_query_failure_without_a_callback_still_does_not_raise(self):
        class Failing:
            def get_recorded_target_runs(self, target_run_uuids):
                raise RuntimeError("db is down")

        assert DBLineageStore(storage=Failing()).filter_unrecorded({"t1"}) == {"t1"}


class TestReleaseCounts:
    """release_id is a build_id; the UNIT is rows, not W&B runs."""

    def test_counts_rows_for_a_build(self, sink):
        sink._write_job(
            job(
                "J1",
                [artifact("i1", LH_TABLE), artifact("i2", "s3://b/i2")],
                [artifact("o1", LH_MODEL)],
            ),
            build_id="BLD",
            target_run_uuid="TR",
        )
        # Two inputs x one output = 2 rows. W&B would report 1 run (one per
        # output), which is exactly the shape mismatch the docstring warns about.
        assert sink.count_release_ids("BLD") == 2

    def test_narrows_by_target_run(self, sink):
        for target_run in ("t1", "t2"):
            sink._write_job(
                job(
                    f"J-{target_run}",
                    [artifact("a", LH_TABLE)],
                    [artifact("b", LH_MODEL)],
                ),
                build_id="BLD",
                target_run_uuid=target_run,
            )
        assert sink.count_release_ids("BLD") == 2
        assert sink.count_release_ids("BLD", target_id="t1") == 1

    def test_an_unknown_release_is_zero(self, sink):
        assert sink.count_release_ids("nope") == 0

    def test_an_empty_release_id_is_zero(self, sink):
        assert sink.count_release_ids("") == 0

    def test_does_release_id_exist_compares_row_counts(self, sink):
        sink._write_job(
            job("J1", [artifact("a", LH_TABLE)], [artifact("b", LH_MODEL)]),
            build_id="BLD",
            target_run_uuid="TR",
        )
        assert sink.does_release_id_exist("BLD", 1) is True
        assert sink.does_release_id_exist("BLD", 5) is False


class TestRowContents:
    def test_promoted_pieces_come_from_the_identifier(self):
        # Parsed back out of the identifier rather than threaded separately, so a
        # column can never disagree with the identifier it describes.
        row = _row_from_draft(
            LineageRowDraft(
                job_id="J1",
                source="model:prod/ns::label|tbl",
                target=None,
            ),
            build_id="BLD",
            target_run_uuid="TR",
        )
        assert row.source_kind == "model"
        assert row.source_namespace == "prod/ns"
        assert row.source_name == "label"
        assert row.source_table == "tbl"

    def test_none_endpoints_become_the_terminal_marker(self):
        row = _row_from_draft(
            LineageRowDraft(job_id="J1", source=None, target="table:prod/ns::t"),
            build_id="BLD",
            target_run_uuid="TR",
        )
        # Not NULL: in SQL NULL never equals NULL, so NULL endpoints would slip
        # past the unique index.
        assert row.source == ""
        assert row.is_creation()

    def test_rows_are_marked_derivable(self):
        # A rebuild deletes only derivable rows, so imported lineage survives it.
        row = _row_from_draft(
            LineageRowDraft(job_id="J1", source="table:prod/ns::t", target=None),
            build_id="BLD",
            target_run_uuid="TR",
        )
        assert row.derivable is True
        assert row.source_system == "granite.build"

    def test_the_real_uri_is_carried_onto_the_row(self, sink, rows):
        # The scheme is not recoverable from a canonical identifier, so the URI has
        # to travel rather than be re-derived on read.
        sink._write_job(
            job(
                "J1",
                [artifact("a", "s3://bkt/raw")],
                [artifact("b", "hf:///org/repo")],
            ),
            build_id="BLD",
            target_run_uuid="TR",
        )
        stored = rows.get_rows_by_job("J1")[0]
        assert stored.source_uri == "s3://bkt/raw"
        assert stored.target_uri == "hf:///org/repo"

    def test_carried_metadata_survives(self, sink, rows):
        sink._write_job(
            job(
                "J1",
                [artifact("a", LH_TABLE)],
                [artifact("b", LH_MODEL)],
                job_name="train",
                owner="alice",
            ),
            build_id="BLD",
            target_run_uuid="TR",
        )
        stored = rows.get_rows_by_job("J1")[0]
        assert stored.metadata["job_name"] == "train"
        assert stored.metadata["owner"] == "alice"


class TestArtifactRegistrationRows:
    """The path with no target run (D11)."""

    def test_rows_carry_no_target_run(self):
        # build_id holds the artifact uuid so count_release_ids finds them, while
        # target_run_uuid stays empty: there is no target run.
        row = _row_from_draft(
            LineageRowDraft(job_id="ART-UUID", source="table:prod/ns::t", target=None),
            build_id="ART-UUID",
            target_run_uuid="",
        )
        assert row.target_run_uuid == ""
        assert row.build_id == "ART-UUID"
        # job_id still identifies the rows, so the composite unique keeps
        # protecting them without depending on a target run.
        assert row.job_id == "ART-UUID"


class TestWiring:
    def test_the_sink_records_centralized_lineage(self, sink):
        # lineage-watch exits immediately for a store reporting False, which would
        # leave the index permanently empty.
        assert sink.records_centralized_lineage is True

    def test_it_implements_the_store_interface(self, sink):
        from gbserver.lineage.jobstats import ILineageStore

        assert isinstance(sink, ILineageStore)

    def test_building_needs_no_wandb_instance(self):
        # The shared builders are pure; reusing them must not require wandb to be
        # installed or a W&B store to be constructed.
        import inspect

        from gbserver.lineage import db_jobstats

        source = inspect.getsource(db_jobstats)
        assert "WandBLineageStore()" not in source


class TestJobNormalization:
    """The builders nest the job identity; decomposition needs it at the top."""

    def test_job_id_is_lifted_out_of_job_details(self):
        from gbserver.lineage.db_jobstats import _normalized_job

        # This is what the shared builders actually emit: sources/targets are
        # mirrored to the top level, but job_id lives under job_details. Reading
        # only the top level yielded job_id=None, which made to_lineage_rows reject
        # EVERY event -- the index stayed empty while recording looked successful.
        event = {
            "sources": [],
            "targets": [],
            "job_details": {"job_id": "TR-UUID", "owner": "alice"},
        }
        normalized = _normalized_job(event)
        assert normalized["job_id"] == "TR-UUID"
        assert normalized["owner"] == "alice"

    def test_a_top_level_job_id_is_preserved(self):
        from gbserver.lineage.db_jobstats import _normalized_job

        event = {"job_id": "J1", "sources": [], "targets": []}
        assert _normalized_job(event)["job_id"] == "J1"

    def test_an_event_with_no_job_details_is_unchanged(self):
        from gbserver.lineage.db_jobstats import _normalized_job

        event = {"job_id": "J1", "sources": [], "targets": []}
        assert _normalized_job(event)["job_id"] == "J1"
