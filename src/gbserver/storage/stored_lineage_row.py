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

"""Storage model for one flat lineage row.

A row is a ``(source, job_id, target)`` triple: the artifacts are identified by
canonical identifier strings, not by uuid, so lineage imported from a source with
no per-artifact uuid (Lakehouse, dmf-ng) is a first-class citizen rather than
something an alias table has to compensate for.

The graph has no node table. Its nodes are the distinct ``source``/``target``
values, and the traversal walks by matching one row's identifier against
another's -- which is why those two columns are indexed and why their length is
validated before a row is ever built.
"""

from typing import Any, Dict, Optional

from pydantic import Field

from gbserver.storage.stored_build import BaseStoredItem

# Terminal marker for source/target. NOT NULL: in SQL, NULL never equals NULL, so
# NULL endpoints would slip past the (job_id, source, target) unique index and
# leave creation/deletion rows -- the least visible ones -- as the only rows a
# re-ingest could duplicate. The prototype's NULL exists to serve a recursive-SQL
# traversal over a table with no unique key; here the traversal runs in Python and
# the unique key exists, so the sentinel costs nothing: the walk stops on a falsy
# endpoint exactly as it would on None.
#
# canonical_id never returns "" (it requires a namespace, plus name and/or table
# by type, raising otherwise), so "" in these columns can only mean "terminal".
TERMINAL = ""


class StoredLineageRow(BaseStoredItem):
    """One lineage row: an input, the job that ran, and an output.

    Attributes:
        job_id: identity of the job execution. The same value on every row of one
            job, which is what keeps an N*M decomposition regroupable: the rows of
            a job with 3 inputs and 2 outputs still say which inputs and which
            outputs that execution had.
        source: canonical identifier of the input artifact, or
            :data:`TERMINAL` when the job had no input (a creation).
        target: canonical identifier of the output artifact, or
            :data:`TERMINAL` when the job produced none (a deletion).
        source_filter: partition filter scoping the input, or ``None`` for a
            whole-entity job. ``None`` is meaningful here -- it is not a terminal
            marker -- and this column is outside the unique index, so it keeps SQL
            NULL rather than a sentinel.
        target_filter: partition filter scoping the output, or ``None``.
        source_kind: artifact type of the input, promoted for filtering and the UI.
        source_namespace: namespace or organization of the input.
        source_name: name piece of the input.
        source_table: table piece of the input; empty for types that have none.
        source_revision: revision piece of the input; only filesets carry one.
        target_kind: artifact type of the output.
        target_namespace: namespace or organization of the output.
        target_name: name piece of the output.
        target_table: table piece of the output.
        target_revision: revision piece of the output.
        source_uri: the input artifact's real URI, as registered, or empty when
            the source had none or the row predates this column. Stored rather
            than derived: a canonical identifier keeps only type, namespace, name,
            table and revision and does NOT encode the scheme, so an ``hf://`` or
            ``s3://`` URI cannot be recovered from it -- only ``lh://`` can. This
            is what lets the read path hand the UI a real URI instead of a
            synthesized one. Outside the unique index, so it never affects row
            identity.
        target_uri: the output artifact's real URI; see ``source_uri``.
        source_system: which system this row came from -- ``"granite.build"`` for
            rows the scan derives, or an importer's name.
        derivable: whether the row can be re-derived by re-scanning. A rebuild
            deletes only derivable rows, so imported lineage survives it. That
            matters because imported lineage will not be re-derivable at all once
            the upstream sources are switched off.
        build_id: the build this row came from; empty for lineage that has no
            build, which includes every imported row.
        target_run_uuid: the target run this row came from; empty when there is
            none. The sink's dedup checks presence of rows for a given value here.
        metadata: carried job metadata (name, status, owner, timestamps, the
            originating artifact dicts). Lives in the JSON blob, so nothing here
            is queryable -- anything that needs filtering has its own column.
    """

    job_id: str = Field(..., description="Identity of the job execution")
    source: str = Field(
        default=TERMINAL,
        description="Canonical identifier of the input artifact; TERMINAL if none",
    )
    target: str = Field(
        default=TERMINAL,
        description="Canonical identifier of the output artifact; TERMINAL if none",
    )

    source_filter: Optional[str] = Field(
        default=None,
        description="Partition filter scoping the input; None means whole entity",
    )
    target_filter: Optional[str] = Field(
        default=None,
        description="Partition filter scoping the output; None means whole entity",
    )

    source_kind: str = Field(default="", description="Artifact type of the input")
    source_namespace: str = Field(
        default="", description="Namespace or organization of the input"
    )
    source_name: str = Field(default="", description="Name piece of the input")
    source_table: str = Field(default="", description="Table piece of the input")
    source_revision: str = Field(default="", description="Revision piece of the input")

    target_kind: str = Field(default="", description="Artifact type of the output")
    target_namespace: str = Field(
        default="", description="Namespace or organization of the output"
    )
    target_name: str = Field(default="", description="Name piece of the output")
    target_table: str = Field(default="", description="Table piece of the output")
    target_revision: str = Field(default="", description="Revision piece of the output")

    source_uri: str = Field(
        default="", description="Real URI of the input artifact, as registered"
    )
    target_uri: str = Field(
        default="", description="Real URI of the output artifact, as registered"
    )

    source_system: str = Field(
        default="granite.build", description="System this row came from"
    )
    derivable: bool = Field(
        default=True, description="Whether a re-scan can regenerate this row"
    )

    build_id: str = Field(default="", description="Build this row came from, if any")
    target_run_uuid: str = Field(
        default="", description="Target run this row came from, if any"
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Carried job metadata"
    )

    def is_creation(self) -> bool:
        """Whether the job produced this output with no recorded input."""
        return self.source == TERMINAL

    def is_deletion(self) -> bool:
        """Whether the job consumed this input and produced nothing."""
        return self.target == TERMINAL

    def is_self_loop(self) -> bool:
        """Whether the job rewrote its own input.

        Legitimate for unversioned entities: several runs rewriting one table
        converge on a single node. The traversal includes such a row but does not
        chain through it.
        """
        return self.source == self.target and self.source != TERMINAL
