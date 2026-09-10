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

"""Base storage interface and implementation for lineage rows.

The queries here are the ones the traversal and the sink need: one batched hop in
each direction, seeding by build, and a presence check for dedup. Each is a single
indexed query, so a graph walk costs one query per level rather than a scan.
"""

from typing import List

from gbserver.storage.storage import BaseItemStorage, IItemStorage
from gbserver.storage.stored_lineage_row import TERMINAL, StoredLineageRow
from gbserver.types.constants import GB_LINEAGE_TABLE_NAME


class ILineageRowStorage(IItemStorage[StoredLineageRow]):
    """Interface for lineage row storage implementations."""

    def get_rows_by_source(self, sources: List[str]) -> List[StoredLineageRow]:
        """Return rows whose ``source`` is one of ``sources`` (descendant hop)."""
        raise NotImplementedError

    def get_rows_by_target(self, targets: List[str]) -> List[StoredLineageRow]:
        """Return rows whose ``target`` is one of ``targets`` (ancestor hop)."""
        raise NotImplementedError

    def get_rows_by_build(self, build_id: str) -> List[StoredLineageRow]:
        """Return every row recorded for one build."""
        raise NotImplementedError

    def get_rows_by_job(self, job_id: str) -> List[StoredLineageRow]:
        """Return every row of one job execution."""
        raise NotImplementedError

    def has_rows_for_target_run(self, target_run_uuid: str) -> bool:
        """Whether any row is already recorded for a target run."""
        raise NotImplementedError

    def get_recorded_target_runs(self, target_run_uuids: List[str]) -> set:
        """Return which of ``target_run_uuids`` already have rows."""
        raise NotImplementedError

    def delete_derivable_rows(self, source_system: str) -> int:
        """Delete re-derivable rows of one source system. Returns the count."""
        raise NotImplementedError


class BaseLineageRowStorage(BaseItemStorage[StoredLineageRow], ILineageRowStorage):
    """Base storage implementation for lineage rows.

    Provides the shared query logic across backends (SQL, SQLite).
    """

    def __init__(self, **kwargs) -> None:
        kwargs["item_class"] = StoredLineageRow
        if kwargs.get("table_name") is None:
            kwargs["table_name"] = GB_LINEAGE_TABLE_NAME
        super().__init__(**kwargs)

    def _get_column_values(self, item: StoredLineageRow) -> dict:
        """Extract the queryable columns from a row.

        Every field returned here becomes a real column; everything else lives in
        the JSON blob, which is Text and therefore neither queryable nor
        indexable. The set is returned unconditionally -- a conditionally omitted
        key would be missing from the schema derived from the sample item.

        Note which columns are strings. ``get_by_where`` only builds an ``IN``
        clause for string-typed columns and silently degrades to
        ``column == [list]`` for anything else, so every column a batched query
        filters on (``source``, ``target``, ``target_run_uuid``, ``build_id``,
        ``job_id``, ``source_system``) must be text. ``derivable`` is a bool and is
        only ever compared for equality, never batched.
        """
        fields_to_include = {
            "job_id",
            "source",
            "target",
            "source_filter",
            "target_filter",
            "source_kind",
            "source_namespace",
            "source_name",
            "source_table",
            "source_revision",
            "target_kind",
            "target_namespace",
            "target_name",
            "target_table",
            "target_revision",
            "source_uri",
            "target_uri",
            "source_system",
            "derivable",
            "build_id",
            "target_run_uuid",
        }
        return item.model_dump(include=fields_to_include)

    @classmethod
    def _get_sample_item(cls) -> StoredLineageRow:
        """Return a sample row used to derive the table schema.

        Every column must be present and of its real type here: the SQL layer
        infers each column's type from this item's values. The filters are
        non-None so they are typed as text rather than being skipped.
        """
        return StoredLineageRow(
            job_id="sample-job",
            source="model:host/ns::sample|tbl",
            target="dataset:host/ns::sample|tbl",
            source_filter='{"dt":"2024-01-01"}',
            target_filter='{"dt":"2024-01-01"}',
            source_kind="model",
            source_namespace="host/ns",
            source_name="sample",
            source_table="tbl",
            source_revision="v1",
            target_kind="dataset",
            target_namespace="host/ns",
            target_name="sample",
            target_table="tbl",
            target_revision="",
            source_uri="lh://prod/ns/models/tbl/sample",
            target_uri="lh://prod/ns/datasets/tbl/sample",
            source_system="granite.build",
            derivable=True,
            build_id="sample-build",
            target_run_uuid="sample-target-run",
        )

    def get_rows_by_source(self, sources: List[str]) -> List[StoredLineageRow]:
        """Return rows whose ``source`` is one of ``sources``.

        One batched, indexed query -- the descendant hop of a level-order walk.

        Args:
            sources: canonical identifiers of the current frontier. The terminal
                marker is dropped: a creation row's source identifies no artifact,
                so matching on it would join unrelated creations together.

        Returns:
            The matching rows, or an empty list when nothing is left to match.
        """
        wanted = self._batchable(sources)
        if not wanted:
            return []
        return self.get_by_where({"source": wanted})

    def get_rows_by_target(self, targets: List[str]) -> List[StoredLineageRow]:
        """Return rows whose ``target`` is one of ``targets``.

        The ancestor hop; see :meth:`get_rows_by_source`.
        """
        wanted = self._batchable(targets)
        if not wanted:
            return []
        return self.get_by_where({"target": wanted})

    @staticmethod
    def _batchable(identifiers: List[str]) -> List[str]:
        """Drop empty identifiers and duplicates from a hop's frontier.

        The terminal marker must never reach a query: every creation row shares it,
        so a hop matching on it would treat unrelated creations as one node. The
        walk stops on a terminal instead of following it.
        """
        return sorted({value for value in identifiers if value and value != TERMINAL})

    def get_rows_by_build(self, build_id: str) -> List[StoredLineageRow]:
        """Return every row recorded for one build.

        Seeds a build-scoped graph with one indexed query, rather than loading the
        build's target runs and deriving the rows again.
        """
        if not build_id:
            return []
        return self.get_by_where({"build_id": build_id})

    def get_rows_by_job(self, job_id: str) -> List[StoredLineageRow]:
        """Return every row of one job execution.

        This is what makes the N*M decomposition lossless: the rows sharing a
        ``job_id`` still say which inputs and outputs that execution had.
        """
        if not job_id:
            return []
        return self.get_by_where({"job_id": job_id})

    def has_rows_for_target_run(self, target_run_uuid: str) -> bool:
        """Whether any row is already recorded for a target run."""
        if not target_run_uuid:
            return False
        return bool(self.get_by_where({"target_run_uuid": target_run_uuid}))

    def get_recorded_target_runs(self, target_run_uuids: List[str]) -> set:
        """Return which of ``target_run_uuids`` already have rows.

        The sink's dedup is presence-based: a target run either has its rows or it
        does not. It deliberately does not compare a row count against the
        reconciler's ``expected_counts``, which counts one W&B run per output
        artifact -- a shape that never matches an N*M row count, and would report
        every target as unrecorded forever.

        Args:
            target_run_uuids: the target runs to check.

        Returns:
            The subset that already has at least one row.
        """
        wanted = self._batchable(list(target_run_uuids))
        if not wanted:
            return set()
        rows = self.get_by_where({"target_run_uuid": wanted})
        return {row.target_run_uuid for row in rows if row.target_run_uuid}

    def delete_derivable_rows(self, source_system: str) -> int:
        """Delete the re-derivable rows of one source system.

        Imported rows are left untouched: they carry ``derivable=False`` and could
        not be regenerated, since the sources they came from are being switched
        off. Because the graph has no node table, deleting rows cannot orphan
        anything.

        Args:
            source_system: the system whose derivable rows to drop.

        Returns:
            How many rows were deleted.
        """
        rows = self.get_by_where({"source_system": source_system, "derivable": True})
        deleted = 0
        for row in rows:
            self.delete(row.uuid)
            deleted += 1
        return deleted
