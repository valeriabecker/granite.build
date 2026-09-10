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

"""Storage-level tests for lineage rows against SQLite.

These exercise the parts that only a real table can prove: that the declared
indexes and the composite unique actually exist, that the unique protects
terminal rows too, and that a batched hop produces an ``IN`` rather than silently
degrading.
"""

import os
import sqlite3
import uuid as uuid_module

import pytest

from gbserver.storage.sqlite.storage_factory import SqliteStorageFactory
from gbserver.storage.stored_lineage_row import TERMINAL, StoredLineageRow

pytestmark = pytest.mark.skipif(
    os.environ.get("SKIP_SQL_ADMIN_TESTS", "False").lower() == "true",
    reason="Don't want to run this in CICD.",
)


@pytest.fixture(name="storage")
def storage_fixture():
    """A lineage row storage on a table unique to this test."""
    table = f"t_lin_{uuid_module.uuid4().hex[:8]}"
    return SqliteStorageFactory().create_lineage_row_storage(table_name=table)


def row(
    job_id: str = "J",
    source: str = "model:h/ns::a|t",
    target: str = "dataset:h/ns::b|t",
    **kwargs,
) -> StoredLineageRow:
    return StoredLineageRow(job_id=job_id, source=source, target=target, **kwargs)


class TestHops:
    """One indexed, batched query per traversal level."""

    def test_hop_by_source_finds_descendant_rows(self, storage):
        storage.add(row(source="a", target="b"))
        storage.add(row(job_id="J2", source="b", target="c"))
        found = storage.get_rows_by_source(["a"])
        assert [r.target for r in found] == ["b"]

    def test_hop_by_target_finds_ancestor_rows(self, storage):
        storage.add(row(source="a", target="b"))
        storage.add(row(job_id="J2", source="b", target="c"))
        found = storage.get_rows_by_target(["c"])
        assert [r.source for r in found] == ["b"]

    def test_hop_batches_a_whole_frontier(self, storage):
        for i in range(5):
            storage.add(row(job_id=f"J{i}", source=f"s{i}", target=f"t{i}"))
        found = storage.get_rows_by_source(["s0", "s2", "s4"])
        assert sorted(r.source for r in found) == ["s0", "s2", "s4"]

    def test_empty_frontier_queries_nothing(self, storage):
        storage.add(row())
        assert storage.get_rows_by_source([]) == []
        assert storage.get_rows_by_target([]) == []

    def test_terminal_marker_never_reaches_a_query(self, storage):
        """Every creation row shares the terminal marker, so a hop matching on it
        would treat unrelated creations as one node.
        """
        storage.add(row(job_id="C1", source=TERMINAL, target="x"))
        storage.add(row(job_id="C2", source=TERMINAL, target="y"))
        storage.add(row(job_id="N", source="real", target="z"))

        # Both creations would match on the marker, collapsing them into one node.
        assert storage.get_rows_by_source([TERMINAL]) == []
        # A real identifier alongside the marker still matches, and only itself.
        found = storage.get_rows_by_source([TERMINAL, "real"])
        assert [r.job_id for r in found] == ["N"]

    def test_frontier_is_deduplicated(self, storage):
        storage.add(row(source="a", target="b"))
        assert len(storage.get_rows_by_source(["a", "a", "a"])) == 1


class TestSeeding:
    def test_rows_by_build(self, storage):
        storage.add(row(job_id="J1", build_id="B1"))
        storage.add(row(job_id="J2", build_id="B1", target="dataset:h/ns::c|t"))
        storage.add(row(job_id="J3", build_id="B2"))
        assert len(storage.get_rows_by_build("B1")) == 2

    def test_empty_build_id_returns_nothing(self, storage):
        """Imported rows carry build_id="" and must not all match one query."""
        storage.add(row(job_id="J1", build_id=""))
        storage.add(row(job_id="J2", build_id="", target="dataset:h/ns::c|t"))
        assert storage.get_rows_by_build("") == []

    def test_rows_by_job_recovers_the_whole_execution(self, storage):
        """The N*M decomposition stays regroupable through storage."""
        for i, (src, tgt) in enumerate(
            [("i1", "o1"), ("i1", "o2"), ("i2", "o1"), ("i2", "o2")]
        ):
            storage.add(row(job_id="J", source=src, target=tgt))
        rows = storage.get_rows_by_job("J")
        assert len(rows) == 4
        assert {r.source for r in rows} == {"i1", "i2"}
        assert {r.target for r in rows} == {"o1", "o2"}


class TestDedupSupport:
    """Presence-based dedup, not count-based."""

    def test_has_rows_for_target_run(self, storage):
        storage.add(row(target_run_uuid="TR1"))
        assert storage.has_rows_for_target_run("TR1")
        assert not storage.has_rows_for_target_run("TR2")

    def test_empty_target_run_is_not_recorded(self, storage):
        storage.add(row(target_run_uuid=""))
        assert not storage.has_rows_for_target_run("")

    def test_get_recorded_target_runs_filters(self, storage):
        storage.add(row(job_id="J1", target_run_uuid="TR1"))
        storage.add(row(job_id="J2", target_run_uuid="TR3"))
        assert storage.get_recorded_target_runs(["TR1", "TR2", "TR3"]) == {"TR1", "TR3"}

    def test_presence_is_independent_of_row_count(self, storage):
        """A target run writing N*M rows is recorded once, however many rows.

        The reconciler's expected_counts is one W&B run per output artifact, which
        can never match an N*M row count -- comparing against it would report every
        target as unrecorded forever and re-record in a loop.
        """
        for i in range(6):
            storage.add(row(job_id="J", source=f"i{i}", target_run_uuid="TR1"))
        assert storage.get_recorded_target_runs(["TR1"]) == {"TR1"}


class TestUniqueConstraint:
    """Re-ingest idempotency -- a property added here, not inherited.

    The prototype has no key at all and executemany's without a guard, so it
    duplicates silently.
    """

    def test_same_triple_is_rejected(self, storage):
        storage.add(row(job_id="J", source="a", target="b"))
        with pytest.raises(Exception):
            storage.add(row(job_id="J", source="a", target="b"))

    def test_creation_terminal_is_protected(self, storage):
        """The reason source/target hold "" and not NULL: in SQL NULL never equals
        NULL, so a NULL endpoint would slip past the unique and leave creation rows
        as the only duplicable ones.
        """
        storage.add(row(job_id="C", source=TERMINAL, target="x"))
        with pytest.raises(Exception):
            storage.add(row(job_id="C", source=TERMINAL, target="x"))

    def test_deletion_terminal_is_protected(self, storage):
        storage.add(row(job_id="D", source="x", target=TERMINAL))
        with pytest.raises(Exception):
            storage.add(row(job_id="D", source="x", target=TERMINAL))

    def test_same_job_different_endpoints_is_allowed(self, storage):
        """An N*M job writes several rows under one job_id."""
        storage.add(row(job_id="J", source="i1", target="o1"))
        storage.add(row(job_id="J", source="i1", target="o2"))
        storage.add(row(job_id="J", source="i2", target="o1"))
        assert len(storage.get_rows_by_job("J")) == 3

    def test_same_endpoints_different_job_is_allowed(self, storage):
        """Two executions can relate the same pair of artifacts."""
        storage.add(row(job_id="J1", source="a", target="b"))
        storage.add(row(job_id="J2", source="a", target="b"))
        assert len(storage.get_rows_by_source(["a"])) == 2


class TestRebuild:
    def test_deletes_only_derivable_rows_of_that_system(self, storage):
        storage.add(row(job_id="J1", source_system="granite.build", derivable=True))
        storage.add(
            row(
                job_id="J2",
                source_system="granite.build",
                derivable=True,
                target="dataset:h/ns::c|t",
            )
        )
        storage.add(row(job_id="J3", source_system="lh", derivable=False))
        storage.add(
            row(
                job_id="J4",
                source_system="granite.build",
                derivable=False,
                target="dataset:h/ns::d|t",
            )
        )

        assert storage.delete_derivable_rows("granite.build") == 2

        remaining = {r.job_id for r in storage.get_by_where({})}
        assert remaining == {"J3", "J4"}

    def test_imported_rows_survive_a_rebuild(self, storage):
        """Imported lineage is not re-derivable once the upstream source is gone."""
        storage.add(row(job_id="IMP", source_system="lh", derivable=False))
        storage.delete_derivable_rows("granite.build")
        assert len(storage.get_rows_by_job("IMP")) == 1


class TestSchema:
    """The declared indexes and unique must actually exist in the table."""

    @staticmethod
    def _query_schema(storage, sql: str, parameters: tuple = ()) -> list:
        connection = sqlite3.connect(str(storage._get_db_file_path()))
        try:
            return connection.execute(sql, parameters).fetchall()
        finally:
            connection.close()

    @classmethod
    def _index_statements(cls, storage) -> list:
        return cls._query_schema(
            storage,
            "SELECT name, sql FROM sqlite_master WHERE type='index' AND tbl_name=?",
            (storage.table_name,),
        )

    @pytest.mark.parametrize(
        "column",
        ["source", "target", "job_id", "target_run_uuid", "build_id", "source_system"],
    )
    def test_column_is_indexed(self, storage, column):
        storage.add(row())
        statements = " ".join(sql or "" for _, sql in self._index_statements(storage))
        assert f"({column})" in statements, f"{column} is not indexed: {statements}"

    def test_composite_unique_exists(self, storage):
        """It can only be created with the table, never added later."""
        storage.add(row())
        statements = [sql or "" for _, sql in self._index_statements(storage)]
        unique = [s for s in statements if "UNIQUE" in s and "job_id" in s]
        assert unique, statements
        assert "(job_id, source, target)" in unique[0]

    def test_traversal_columns_are_text(self, storage):
        """get_by_where only builds an IN clause for string columns; anything else
        silently degrades to ``column == [list]``.
        """
        storage.add(row())
        columns = {
            name: kind
            for _, name, kind, *_ in self._query_schema(
                storage, f"PRAGMA table_info('{storage.table_name}')"
            )
        }
        for column in ("source", "target", "job_id", "target_run_uuid", "build_id"):
            assert columns[column].startswith("VARCHAR"), (column, columns[column])
        assert columns["derivable"] == "BOOLEAN"


class TestRoundTrip:
    def test_all_fields_survive_storage(self, storage):
        original = StoredLineageRow(
            job_id="J",
            source="model:h/ns::a|t",
            target="fileset:h/ns::b@v1|t",
            source_filter='{"dt":"2024"}',
            target_filter=None,
            source_kind="model",
            source_namespace="h/ns",
            source_name="a",
            source_table="t",
            source_revision="",
            target_kind="fileset",
            target_namespace="h/ns",
            target_name="b",
            target_table="t",
            target_revision="v1",
            source_system="lh",
            derivable=False,
            build_id="B",
            target_run_uuid="TR",
            metadata={"job_name": "train", "owner": "someone"},
        )
        storage.add(original)
        stored = storage.get_rows_by_job("J")[0]

        assert stored.source == original.source
        assert stored.target == original.target
        assert stored.source_filter == '{"dt":"2024"}'
        assert stored.target_filter is None
        assert stored.target_revision == "v1"
        assert stored.source_system == "lh"
        assert stored.derivable is False
        # metadata lives in the JSON blob, not a column
        assert stored.metadata == {"job_name": "train", "owner": "someone"}

    def test_terminal_helpers_survive_storage(self, storage):
        storage.add(row(job_id="C", source=TERMINAL, target="x"))
        storage.add(row(job_id="D", source="x", target=TERMINAL))
        storage.add(row(job_id="S", source="tbl", target="tbl"))

        assert storage.get_rows_by_job("C")[0].is_creation()
        assert storage.get_rows_by_job("D")[0].is_deletion()
        assert storage.get_rows_by_job("S")[0].is_self_loop()

    def test_creation_row_is_not_a_self_loop(self, storage):
        """Two terminal rows both have source == target == "" but are not loops."""
        storage.add(row(job_id="C", source=TERMINAL, target=TERMINAL))
        assert not storage.get_rows_by_job("C")[0].is_self_loop()


class TestWalkAgainstRealStorage:
    """The traversal over the real table, not a fake.

    The fake in test_walk.py proves the algorithm; this proves the algorithm and
    the storage agree -- that a batched hop really produces an IN over the indexed
    column, and that the terminal sentinel survives a round trip through SQL.
    """

    def test_walks_a_chain_stored_in_sqlite(self, storage):
        from gbserver.lineage.walk import Direction, walk_lineage

        storage.add(row(job_id="J1", source="a", target="b"))
        storage.add(row(job_id="J2", source="b", target="c"))

        graph = walk_lineage(storage, ["a"], Direction.DESCENDANTS)
        assert graph.depths == {"a": 0, "b": 1, "c": 2}

        back = walk_lineage(storage, ["c"], Direction.ANCESTORS)
        assert back.depths == {"c": 0, "b": 1, "a": 2}

    def test_terminal_survives_a_round_trip(self, storage):
        from gbserver.lineage.walk import Direction, walk_lineage

        storage.add(row(job_id="C", source=TERMINAL, target="a"))
        storage.add(row(job_id="J", source="a", target="b"))

        graph = walk_lineage(storage, ["b"], Direction.ANCESTORS)
        assert graph.depths == {"b": 0, "a": 1}
        assert TERMINAL not in graph.depths
        assert any(r.is_creation() for r in graph.rows)

    def test_cartesian_job_walks_from_any_endpoint(self, storage):
        from gbserver.lineage.walk import Direction, walk_lineage

        for src in ("i1", "i2", "i3"):
            for tgt in ("o1", "o2"):
                storage.add(row(job_id="J", source=src, target=tgt))

        graph = walk_lineage(storage, ["i1"], Direction.DESCENDANTS)
        assert graph.depths == {"i1": 0, "o1": 1, "o2": 1}

        # From an output, every input of that execution is one hop back.
        back = walk_lineage(storage, ["o1"], Direction.ANCESTORS)
        assert back.depths == {"o1": 0, "i1": 1, "i2": 1, "i3": 1}


class TestStoredUriColumns:
    """The URI columns exist as real columns and round-trip.

    They are what the read path hands the UI as node identity, and a canonical
    identifier does not encode a scheme -- so an ``hf://`` or ``s3://`` URI is only
    ever available because it was stored here.
    """

    def test_uri_columns_are_promoted(self, storage):
        storage.add(row())
        assert "source_uri" in storage.get_column_names()
        assert "target_uri" in storage.get_column_names()

    def test_uris_round_trip(self, storage):
        storage.add(row(source_uri="s3://bkt/in", target_uri="hf:///org/out"))
        stored = storage.get_by_where({"job_id": "J"})[0]
        assert stored.source_uri == "s3://bkt/in"
        assert stored.target_uri == "hf:///org/out"

    def test_uris_default_to_empty_for_rows_that_have_none(self, storage):
        storage.add(row())
        stored = storage.get_by_where({"job_id": "J"})[0]
        assert stored.source_uri == ""
        assert stored.target_uri == ""

    def test_uri_is_outside_the_unique_index(self, storage):
        # Row identity is (job_id, source, target). A differing URI must not let a
        # duplicate row in, or re-ingesting an artifact whose URI was rewritten
        # would double every edge it touches.
        storage.add(row(source_uri="s3://bkt/one"))
        with pytest.raises(Exception):
            storage.add(row(source_uri="s3://bkt/two"))
