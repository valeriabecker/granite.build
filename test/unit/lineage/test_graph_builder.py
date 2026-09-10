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

"""Tests for projecting walked rows into the API's graph dict.

The conversion is between two disagreeing models: a stored row keeps the job in a
*column*, while the wire graph is bipartite and the job is a *node*. Most of what
is checked here is that the disagreement is resolved without inventing or losing
structure -- especially that terminals do not become nodes and that one job's rows
converge on one run node.
"""

from gbserver.lineage.graph_builder import build_graph_dict
from gbserver.lineage.walk import LineageGraph
from gbserver.storage.stored_lineage_row import TERMINAL, StoredLineageRow


def row(
    job_id: str,
    source: str,
    target: str,
    source_uri: str = "",
    target_uri: str = "",
    **kwargs,
) -> StoredLineageRow:
    return StoredLineageRow(
        job_id=job_id,
        source=source,
        target=target,
        source_uri=source_uri,
        target_uri=target_uri,
        **kwargs,
    )


def graph_of(*rows) -> LineageGraph:
    depths = {}
    for r in rows:
        for endpoint in (r.source, r.target):
            if endpoint and endpoint != TERMINAL:
                depths.setdefault(endpoint, 1)
    return LineageGraph(rows=list(rows), depths=depths)


def nodes_by_type(result: dict, node_type: str) -> list:
    return [n for n in result["nodes"] if n["node_type"] == node_type]


def edge_pairs(result: dict) -> set:
    return {(e["source"], e["target"]) for e in result["edges"]}


A = "table:prod/ns::a"
B = "table:prod/ns::b"
C = "table:prod/ns::c"


class TestBipartiteShape:
    def test_one_row_becomes_two_edges_through_a_run(self):
        result = build_graph_dict(graph_of(row("J1", A, B)), A)
        assert edge_pairs(result) == {(A, "run:J1"), ("run:J1", B)}

    def test_artifact_and_run_nodes_are_both_present(self):
        result = build_graph_dict(graph_of(row("J1", A, B)), A)
        assert {n["id"] for n in nodes_by_type(result, "artifact")} == {A, B}
        assert {n["id"] for n in nodes_by_type(result, "run")} == {"run:J1"}

    def test_run_node_id_cannot_collide_with_an_artifact_id(self):
        # A job_id shaped like a canonical identifier would otherwise fuse a run
        # node and an artifact node into one.
        result = build_graph_dict(graph_of(row(A, A, B)), A)
        run_ids = {n["id"] for n in nodes_by_type(result, "run")}
        artifact_ids = {n["id"] for n in nodes_by_type(result, "artifact")}
        assert not run_ids & artifact_ids


class TestJobRegrouping:
    def test_n_by_m_rows_converge_on_one_run_node(self):
        # The N*M decomposition reassembled: 2 inputs x 2 outputs is 4 rows, but
        # one execution, so one run node with 2 in and 2 out.
        rows = [
            row("J1", "table:prod/ns::i1", "table:prod/ns::o1"),
            row("J1", "table:prod/ns::i1", "table:prod/ns::o2"),
            row("J1", "table:prod/ns::i2", "table:prod/ns::o1"),
            row("J1", "table:prod/ns::i2", "table:prod/ns::o2"),
        ]
        result = build_graph_dict(graph_of(*rows), "table:prod/ns::i1")
        assert len(nodes_by_type(result, "run")) == 1
        inbound = {e for e in edge_pairs(result) if e[1] == "run:J1"}
        outbound = {e for e in edge_pairs(result) if e[0] == "run:J1"}
        assert len(inbound) == 2 and len(outbound) == 2

    def test_different_jobs_get_different_run_nodes(self):
        result = build_graph_dict(graph_of(row("J1", A, B), row("J2", B, C)), A)
        assert len(nodes_by_type(result, "run")) == 2

    def test_edges_are_deduplicated(self):
        result = build_graph_dict(graph_of(row("J1", A, B), row("J1", A, B)), A)
        assert len(result["edges"]) == len(edge_pairs(result))


class TestTerminals:
    def test_a_creation_contributes_no_source_node(self):
        # Every creation row shares the terminal marker; emitting a node for it
        # would collapse all creations in the graph into one shared node.
        result = build_graph_dict(graph_of(row("J1", TERMINAL, B)), B)
        assert {n["id"] for n in nodes_by_type(result, "artifact")} == {B}
        assert edge_pairs(result) == {("run:J1", B)}

    def test_a_deletion_contributes_no_target_node(self):
        result = build_graph_dict(graph_of(row("J1", A, TERMINAL)), A)
        assert {n["id"] for n in nodes_by_type(result, "artifact")} == {A}
        assert edge_pairs(result) == {(A, "run:J1")}

    def test_two_creations_do_not_share_a_node(self):
        result = build_graph_dict(
            graph_of(row("J1", TERMINAL, B), row("J2", TERMINAL, C)), B
        )
        assert {n["id"] for n in nodes_by_type(result, "artifact")} == {B, C}


class TestRoot:
    def test_root_is_flagged(self):
        result = build_graph_dict(graph_of(row("J1", A, B)), A)
        roots = [n for n in result["nodes"] if n["is_root"]]
        assert [n["id"] for n in roots] == [A]

    def test_root_id_is_echoed(self):
        result = build_graph_dict(graph_of(row("J1", A, B)), A)
        assert result["root_id"] == A

    def test_an_artifact_with_no_lineage_still_yields_its_own_node(self):
        # "Nothing recorded" is a real answer and must not read as an error, nor
        # describe a different artifact than the one asked about.
        result = build_graph_dict(LineageGraph(), A)
        assert result["edges"] == []
        assert [n["id"] for n in result["nodes"]] == [A]
        assert result["nodes"][0]["is_root"] is True

    def test_only_one_node_is_root(self):
        result = build_graph_dict(graph_of(row("J1", A, B), row("J2", B, C)), B)
        assert sum(1 for n in result["nodes"] if n["is_root"]) == 1


class TestNodeIdentity:
    def test_an_artifact_reached_twice_is_one_node(self):
        # A diamond: two paths converge on C.
        rows = [
            row("J1", A, B),
            row("J2", A, C),
            row("J3", B, "table:prod/ns::d"),
            row("J4", C, "table:prod/ns::d"),
        ]
        result = build_graph_dict(graph_of(*rows), A)
        ids = [n["id"] for n in nodes_by_type(result, "artifact")]
        assert len(ids) == len(set(ids))

    def test_stored_uri_reaches_the_node_metadata(self):
        result = build_graph_dict(graph_of(row("J1", A, B, source_uri="s3://bkt/a")), A)
        node = next(n for n in result["nodes"] if n["id"] == A)
        assert node["metadata"]["uri"] == "s3://bkt/a"

    def test_nodes_without_a_stored_uri_still_differ(self):
        # What the frontend deduplicates by, so equal URIs would merge them.
        result = build_graph_dict(graph_of(row("J1", A, B)), A)
        uris = [n["metadata"]["uri"] for n in nodes_by_type(result, "artifact")]
        assert len(uris) == len(set(uris))
        assert all(u for u in uris)

    def test_name_falls_back_to_the_identifier_pieces(self):
        # An importer may not promote every column; the identifier always has them.
        result = build_graph_dict(graph_of(row("J1", A, B)), A)
        node = next(n for n in result["nodes"] if n["id"] == A)
        assert node["name"] == "a"
        assert node["artifact_type"] == "table"

    def test_promoted_columns_are_preferred_over_parsing(self):
        result = build_graph_dict(
            graph_of(row("J1", A, B, source_name="Display Name", source_kind="model")),
            A,
        )
        node = next(n for n in result["nodes"] if n["id"] == A)
        assert node["name"] == "Display Name"
        assert node["artifact_type"] == "model"


class TestRunMetadata:
    def test_job_namespace_survives_for_the_authorization_filter(self):
        # The API handler splits this on "/" to recover the space and drop runs the
        # caller cannot see, so losing it would fail every run closed.
        result = build_graph_dict(
            graph_of(
                row(
                    "J1",
                    A,
                    B,
                    metadata={"job_namespace": "my-space/my-build", "owner": "alice"},
                )
            ),
            A,
        )
        run = nodes_by_type(result, "run")[0]
        assert run["metadata"]["job_namespace"] == "my-space/my-build"
        assert run["metadata"]["owner"] == "alice"

    def test_run_name_comes_from_job_name_when_present(self):
        result = build_graph_dict(
            graph_of(row("J1", A, B, metadata={"job_name": "train"})), A
        )
        assert nodes_by_type(result, "run")[0]["name"] == "train"

    def test_run_name_falls_back_to_job_id(self):
        result = build_graph_dict(graph_of(row("J1", A, B)), A)
        assert nodes_by_type(result, "run")[0]["name"] == "J1"

    def test_unknown_metadata_keys_are_not_copied(self):
        result = build_graph_dict(
            graph_of(row("J1", A, B, metadata={"surprise": "x"})), A
        )
        assert "surprise" not in nodes_by_type(result, "run")[0]["metadata"]

    def test_build_provenance_is_exposed(self):
        result = build_graph_dict(
            graph_of(row("J1", A, B, build_id="BLD", target_run_uuid="TR")), A
        )
        metadata = nodes_by_type(result, "run")[0]["metadata"]
        assert metadata["gb_build_id"] == "BLD"
        assert metadata["gb_target_run_uuid"] == "TR"


class TestTruncation:
    def test_truncated_is_propagated(self):
        graph = graph_of(row("J1", A, B))
        graph.truncated = True
        assert build_graph_dict(graph, A)["truncated"] is True

    def test_untruncated_is_propagated(self):
        assert build_graph_dict(graph_of(row("J1", A, B)), A)["truncated"] is False
