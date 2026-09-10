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

"""Turn walked lineage rows into the graph dict the API already speaks.

The read path's shape is not ours to choose: ``api/lineage.get_artifact_graph``
consumes ``{root_id, nodes, edges, truncated}`` with artifact and run nodes and
directed edges between them, and re-projects that into the run-centred
``ArtifactGraphResponse`` the frontend renders. Serving the index through the same
dict is what lets a second provider land with no frontend change.

The shape conversion is the real work here, because the storage model and the wire
model disagree about what a node is. A stored row is one flat
``(source, job_id, target)`` triple -- the job is a *column*, not a node. The wire
graph is **bipartite**: artifacts and runs are both nodes, and every edge joins one
of each (``artifact -> run`` for a consumed input, ``run -> artifact`` for a
produced output). So one row expands into up to two edges through a shared run
node, and the rows of one ``job_id`` converge on that single run node -- which is
what reassembles the N*M decomposition into "this execution had these inputs and
these outputs".

Terminals do not become nodes. A creation row's source and a deletion row's target
are the empty marker, identifying no artifact; each contributes only the half-edge
it does have. Emitting a node for them would collapse every creation in the graph
into one shared "nothing" node.
"""

import logging
from typing import Optional

from gbserver.lineage.identity import LineageIdentityError, parse_canonical_id
from gbserver.lineage.node_uri import node_uri
from gbserver.lineage.walk import LineageGraph
from gbserver.storage.stored_lineage_row import TERMINAL, StoredLineageRow

logger = logging.getLogger(__name__)

# Wire node_type values, matching what the W&B provider emits and what
# api/lineage.get_artifact_graph branches on.
NODE_TYPE_ARTIFACT = "artifact"
NODE_TYPE_RUN = "run"

# Run-node metadata keys the API handler reads off a run node to build an
# ArtifactRunEntry. Listed so a row's carried metadata is copied under the names
# the handler expects rather than whatever the sink happened to store.
_RUN_METADATA_KEYS = (
    "job_name",
    "job_namespace",
    "job_type",
    "run_id",
    "created_at",
    "state",
    "job_id",
    "job_status",
    "job_started_at",
    "job_completed_at",
    "release_id",
    "category",
    "owner",
    "source_code_details",
    "job_input_params",
    "execution_stats",
    "job_output_stats",
)


def build_graph_dict(
    graph: LineageGraph,
    root_identifier: str,
    root_uri: str = "",
    root_is_artifact: bool = True,
) -> dict:
    """Project a walked graph into the API's graph dict.

    Args:
        graph: the walk result, as returned by
            :func:`~gbserver.lineage.walk.walk_lineage`.
        root_identifier: what the graph is *about*. It becomes ``root_id``.
        root_uri: the root artifact's real URI when the caller knows it, e.g.
            because the request supplied one. Rows carry their own URIs, so this is
            only a fallback for a root that appears in no row -- an artifact with no
            lineage yet.
        root_is_artifact: whether ``root_identifier`` names an artifact. ``False``
            for a build-seeded graph, where it names a build: a build is not a node,
            so nothing is flagged ``is_root`` and no node is synthesized for it.
            Defaulting this to ``True`` would mint a bogus artifact node named after
            the build and flag it as the root artifact.

    Returns:
        ``{root_id, nodes, edges, truncated}``. An empty artifact graph yields the
        root node alone with no edges: "nothing recorded" is a real answer and must
        not read as an error.
    """
    artifact_nodes: dict[str, dict] = {}
    run_nodes: dict[str, dict] = {}
    edges: list[dict] = []
    edge_keys: set[tuple[str, str]] = set()

    def add_edge(source: str, target: str) -> None:
        key = (source, target)
        if key in edge_keys:
            return
        edge_keys.add(key)
        edges.append({"source": source, "target": target})

    for row in graph.rows:
        run_id = _run_node_id(row)
        if run_id not in run_nodes:
            run_nodes[run_id] = _run_node(row, run_id)

        if row.source != TERMINAL:
            _ensure_artifact_node(
                artifact_nodes,
                identifier=row.source,
                uri=row.source_uri,
                kind=row.source_kind,
                name=row.source_name,
            )
            add_edge(row.source, run_id)

        if row.target != TERMINAL:
            _ensure_artifact_node(
                artifact_nodes,
                identifier=row.target,
                uri=row.target_uri,
                kind=row.target_kind,
                name=row.target_name,
            )
            add_edge(run_id, row.target)

    if root_is_artifact and root_identifier:
        # The root may appear in no row -- an artifact with no lineage recorded yet.
        # It still has to be in the graph, or the response would describe a
        # different artifact than the one that was asked about.
        if root_identifier not in artifact_nodes:
            _ensure_artifact_node(
                artifact_nodes,
                identifier=root_identifier,
                uri=root_uri,
                kind="",
                name="",
            )
        artifact_nodes[root_identifier]["is_root"] = True

    return {
        "root_id": root_identifier,
        "nodes": list(artifact_nodes.values()) + list(run_nodes.values()),
        "edges": edges,
        "truncated": graph.truncated,
    }


def _ensure_artifact_node(
    nodes: dict[str, dict],
    identifier: str,
    uri: str,
    kind: str,
    name: str,
) -> None:
    """Add an artifact node for ``identifier`` if it is not already present.

    Keyed by canonical identifier, so the same artifact reached along several paths
    is one node -- the deduplication the graph depends on, done on identity rather
    than on the URI. The URI still matters because it is what the *frontend*
    deduplicates by downstream (see :mod:`gbserver.lineage.node_uri`).

    First writer wins. Two rows describing one artifact carry the same promoted
    pieces by construction (they are derived from its identifier), so there is
    nothing to reconcile; re-deriving on every mention would only cost parses.
    """
    if identifier in nodes:
        return

    display_name = name
    artifact_type: Optional[str] = kind or None
    if not display_name or not artifact_type:
        # Rows written by an importer may not have promoted every piece. The
        # identifier always carries them, so fall back to parsing it rather than
        # showing a node with no name.
        try:
            identity = parse_canonical_id(identifier)
        except LineageIdentityError:
            logger.debug("Unparseable lineage identifier in graph: %r", identifier)
        else:
            display_name = display_name or identity.name or identity.table
            artifact_type = artifact_type or (identity.artifact_type.value or None)

    nodes[identifier] = {
        "id": identifier,
        "node_type": NODE_TYPE_ARTIFACT,
        "name": display_name or identifier,
        "artifact_type": artifact_type,
        "is_root": False,
        "metadata": {"uri": node_uri(identifier, uri)},
    }


def _run_node_id(row: StoredLineageRow) -> str:
    """Identity of the run node a row hangs off.

    ``job_id`` is the same value on every row of one execution, which is exactly
    the grouping a run node needs: the N*M rows of a job with 3 inputs and 2
    outputs converge on one run with 3 inbound and 2 outbound edges.

    Prefixed so a run node id can never equal an artifact node id. They share one
    id space in the wire graph -- edges reference plain strings -- and a job_id that
    happened to look like a canonical identifier would otherwise fuse a run and an
    artifact into one node.
    """
    return f"run:{row.job_id}"


def _run_node(row: StoredLineageRow, run_id: str) -> dict:
    """Build the run node for a row's job execution.

    The metadata a row carries lives in its JSON blob; the API handler reads
    specific keys off a run node's metadata to assemble an ``ArtifactRunEntry``, so
    only those are copied, under those names.

    ``job_namespace`` is load-bearing beyond display: the handler splits it on the
    first ``/`` to recover the space name and drops runs the caller cannot see. A
    run with neither namespace nor owner therefore fails closed, which is the
    intended behaviour for a row whose provenance is unknown.
    """
    metadata = row.metadata or {}
    node_metadata = {
        key: metadata[key] for key in _RUN_METADATA_KEYS if key in metadata
    }

    # Identity of the execution, for a client correlating back to the index.
    node_metadata.setdefault("job_id", row.job_id)
    if row.build_id:
        node_metadata.setdefault("gb_build_id", row.build_id)
    if row.target_run_uuid:
        node_metadata.setdefault("gb_target_run_uuid", row.target_run_uuid)
    node_metadata.setdefault("source_system", row.source_system)

    return {
        "id": run_id,
        "node_type": NODE_TYPE_RUN,
        "name": metadata.get("job_name") or row.job_id,
        "artifact_type": None,
        "is_root": False,
        "metadata": node_metadata,
    }
