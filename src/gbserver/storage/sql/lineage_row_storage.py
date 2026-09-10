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

"""SQL storage implementation for lineage rows."""

from gbserver.storage.lineage_row_storage import (
    BaseLineageRowStorage,
    ILineageRowStorage,
)
from gbserver.storage.sql.sql_storage import BaseSQLItemStorage
from gbserver.storage.stored_lineage_row import StoredLineageRow


class SQLLineageRowStorage(
    BaseSQLItemStorage[StoredLineageRow],
    BaseLineageRowStorage,
    ILineageRowStorage,
):
    """SQL-based storage implementation for lineage rows.

    Schema decisions, and why each column is indexed:

    - ``source`` / ``target`` carry the graph traversal. Every hop is
      ``WHERE source IN (frontier)`` or ``WHERE target IN (frontier)``, so without
      these two indexes a walk degrades to a full scan per level. The prototype
      indexes exactly these.
    - ``job_id`` groups the rows of one execution, which is what makes an N*M
      decomposition regroupable. The prototype joins and groups on it but leaves it
      unindexed -- a gap corrected here.
    - ``target_run_uuid`` backs the sink's presence-based dedup, run on every scan.
    - ``build_id`` seeds a build-scoped graph with one query instead of loading the
      build's target runs.
    - ``source_system`` scopes the rebuild delete.

    ``derivable`` is promoted but deliberately not indexed: it is a bool, only ever
    compared for equality in the rebuild, and never batched -- a list against a
    non-string column silently degrades to ``column == [list]``.
    """

    def __init__(self, **kwargs) -> None:
        kwargs["indexed_columns"] = [
            "source",
            "target",
            "job_id",
            "target_run_uuid",
            "build_id",
            "source_system",
        ]
        # One row per (job, input, output). A second source reporting the same
        # relation is a no-op, which is what makes re-ingest idempotent -- a
        # property the prototype lacks entirely (it has no key at all and
        # executemany's without a guard, so it duplicates silently).
        #
        # This is why source/target hold the "" sentinel rather than NULL for
        # terminals: in SQL, NULL never equals NULL, so NULL endpoints would slip
        # past this index and leave creation/deletion rows unprotected. Note also
        # that unique indexes are only created with the table, so this cannot be
        # added later without recreating it.
        kwargs["unique_columns"] = {("job_id", "source", "target"): None}
        kwargs["autoincr_column"] = "index"
        kwargs["default_pagination_sort_by_column"] = "index"
        super().__init__(**kwargs)
