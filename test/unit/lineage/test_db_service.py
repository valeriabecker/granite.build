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

"""Tests for serving artifact lineage out of the local index.

The distinction under most of these: ``None`` becomes an HTTP 404, which the
frontend renders as "Lineage is not available". So ``None`` has to mean "this
artifact is unknown here" and never "this artifact has no lineage yet" -- the
second is a real, successful answer and must come back as a graph.
"""

import pytest

from gbserver.lineage.db_service import DBLineageService
from gbserver.lineage.openlineage_service import (
    LineageServiceFactory,
    NoopLineageService,
)
from gbserver.storage.stored_lineage_row import TERMINAL, StoredLineageRow

A = "table:prod/ns::a"
B = "table:prod/ns::b"
C = "table:prod/ns::c"


def row(job_id: str, source: str, target: str, **kwargs) -> StoredLineageRow:
    return StoredLineageRow(job_id=job_id, source=source, target=target, **kwargs)


class FakeStorage:
    """In-memory storage implementing the methods the service uses."""

    def __init__(self, rows: list, fail: bool = False) -> None:
        self.rows = rows
        self.fail = fail
        self.pages = 0

    def get_paged(self, where=None, page_size: int = 200):
        if self.fail:
            raise RuntimeError("storage is down")
        self.pages += 1
        yield list(self.rows)

    def get_rows_by_source(self, sources: list) -> list:
        wanted = {s for s in sources if s and s != TERMINAL}
        return [r for r in self.rows if r.source in wanted]

    def get_rows_by_target(self, targets: list) -> list:
        wanted = {t for t in targets if t and t != TERMINAL}
        return [r for r in self.rows if r.target in wanted]

    def get_rows_by_build(self, build_id: str) -> list:
        if not build_id:
            return []
        return [r for r in self.rows if r.build_id == build_id]


def service(*rows, **kwargs) -> DBLineageService:
    return DBLineageService(storage=FakeStorage(list(rows), **kwargs))


class TestProviderRegistration:
    def test_db_provider_is_registered(self):
        assert isinstance(LineageServiceFactory.create("db"), DBLineageService)

    def test_none_still_yields_the_noop_service(self):
        assert isinstance(LineageServiceFactory.create("none"), NoopLineageService)

    def test_an_unknown_provider_still_raises(self):
        with pytest.raises(ValueError):
            LineageServiceFactory.create("not-a-provider")


class TestRootResolution:
    def test_resolves_by_canonical_identifier(self):
        result = service(row("J1", A, B)).get_artifact_graph(artifact_name=A)
        assert result is not None and result["root_id"] == A

    def test_resolves_by_stored_uri(self):
        svc = service(row("J1", A, B, source_uri="s3://bkt/a"))
        result = svc.get_artifact_graph(artifact_url="s3://bkt/a")
        assert result is not None and result["root_id"] == A

    def test_resolves_by_promoted_name(self):
        svc = service(row("J1", A, B, source_name="my-artifact"))
        result = svc.get_artifact_graph(artifact_name="my-artifact")
        assert result is not None and result["root_id"] == A

    def test_resolves_a_target_endpoint_too(self):
        result = service(row("J1", A, B)).get_artifact_graph(artifact_name=B)
        assert result is not None and result["root_id"] == B

    def test_an_unknown_artifact_is_none(self):
        # This is the 404 case: nothing in the index mentions it.
        assert service(row("J1", A, B)).get_artifact_graph(artifact_name="nope") is None

    def test_no_identifier_at_all_is_none(self):
        assert service(row("J1", A, B)).get_artifact_graph() is None

    def test_an_empty_index_is_none(self):
        assert service().get_artifact_graph(artifact_name=A) is None

    def test_a_storage_failure_reports_not_found_rather_than_raising(self):
        svc = service(row("J1", A, B), fail=True)
        assert svc.get_artifact_graph(artifact_name=A) is None

    def test_an_identifier_match_beats_a_name_match(self):
        # A row whose *name* happens to equal another row's identifier must not
        # shadow the exact hit.
        rows = [row("J1", A, B), row("J2", C, B, source_name=A)]
        result = service(*rows).get_artifact_graph(artifact_name=A)
        assert result is not None and result["root_id"] == A


class TestGraphContent:
    def test_descendants_walk_from_the_root(self):
        svc = service(row("J1", A, B), row("J2", B, C))
        result = svc.get_artifact_graph(artifact_name=A, direction="downstream")
        ids = {n["id"] for n in result["nodes"] if n["node_type"] == "artifact"}
        assert ids == {A, B, C}

    def test_upstream_walks_toward_ancestors(self):
        svc = service(row("J1", A, B), row("J2", B, C))
        result = svc.get_artifact_graph(artifact_name=C, direction="upstream")
        ids = {n["id"] for n in result["nodes"] if n["node_type"] == "artifact"}
        assert ids == {A, B, C}

    def test_downstream_does_not_walk_backward(self):
        # Pins the wire mapping: downstream is used_by(), toward descendants, which
        # is what the live W&B backend and the frontend already mean by it.
        svc = service(row("J1", A, B), row("J2", B, C))
        result = svc.get_artifact_graph(artifact_name=B, direction="downstream")
        ids = {n["id"] for n in result["nodes"] if n["node_type"] == "artifact"}
        assert ids == {B, C}
        assert A not in ids

    def test_upstream_does_not_walk_forward(self):
        svc = service(row("J1", A, B), row("J2", B, C))
        result = svc.get_artifact_graph(artifact_name=B, direction="upstream")
        ids = {n["id"] for n in result["nodes"] if n["node_type"] == "artifact"}
        assert ids == {A, B}
        assert C not in ids

    def test_both_reaches_each_side(self):
        svc = service(row("J1", A, B), row("J2", B, C))
        result = svc.get_artifact_graph(artifact_name=B, direction="both")
        ids = {n["id"] for n in result["nodes"] if n["node_type"] == "artifact"}
        assert ids == {A, B, C}

    def test_an_artifact_with_no_edges_is_a_graph_not_a_404(self):
        # It IS in the index -- as a creation with no input -- so the answer is a
        # graph with its own node, never None.
        svc = service(row("J1", TERMINAL, A))
        result = svc.get_artifact_graph(artifact_name=A, direction="upstream")
        assert result is not None
        assert result["root_id"] == A

    def test_max_depth_truncates_and_says_so(self):
        svc = service(row("J1", A, B), row("J2", B, C))
        result = svc.get_artifact_graph(
            artifact_name=A, direction="downstream", max_depth=1
        )
        assert result["truncated"] is True


class TestValidation:
    def test_an_unknown_direction_raises(self):
        # The API layer maps ValueError to a 400.
        with pytest.raises(ValueError):
            service(row("J1", A, B)).get_artifact_graph(
                artifact_name=A, direction="sideways"
            )

    def test_a_contradicted_artifact_type_raises(self):
        svc = service(row("J1", A, B, source_kind="table"))
        with pytest.raises(ValueError):
            svc.get_artifact_graph(artifact_name=A, artifact_type="model")

    def test_a_matching_artifact_type_is_accepted(self):
        svc = service(row("J1", A, B, source_kind="table"))
        assert svc.get_artifact_graph(artifact_name=A, artifact_type="table")

    def test_direction_is_validated_before_the_root_lookup(self):
        # Otherwise an unknown artifact would 404 on a request that is also a bad
        # request, hiding the client's actual mistake.
        with pytest.raises(ValueError):
            service().get_artifact_graph(artifact_name="nope", direction="sideways")


class TestBuildGraph:
    def test_seeds_from_a_builds_rows(self):
        svc = service(
            row("J1", A, B, build_id="BLD"), row("J2", B, C, build_id="OTHER")
        )
        result = svc.get_build_graph("BLD")
        assert result is not None
        assert result["root_id"] == "BLD"

    def test_crosses_into_other_builds_by_default(self):
        svc = service(
            row("J1", A, B, build_id="BLD"), row("J2", B, C, build_id="OTHER")
        )
        result = svc.get_build_graph("BLD", direction="downstream")
        ids = {n["id"] for n in result["nodes"] if n["node_type"] == "artifact"}
        assert C in ids

    def test_within_build_only_stays_inside(self):
        svc = service(
            row("J1", A, B, build_id="BLD"), row("J2", B, C, build_id="OTHER")
        )
        result = svc.get_build_graph(
            "BLD", direction="downstream", within_build_only=True
        )
        ids = {n["id"] for n in result["nodes"] if n["node_type"] == "artifact"}
        assert C not in ids

    def test_a_build_with_no_rows_is_none(self):
        assert service(row("J1", A, B, build_id="BLD")).get_build_graph("OTHER") is None

    def test_an_empty_build_id_is_none(self):
        assert service(row("J1", A, B, build_id="BLD")).get_build_graph("") is None

    def test_no_artifact_is_flagged_root_for_a_build(self):
        # root_id names a build, not an artifact; flagging one arbitrarily would
        # misreport which artifact was asked about.
        svc = service(row("J1", A, B, build_id="BLD"))
        result = svc.get_build_graph("BLD")
        assert not any(n["is_root"] for n in result["nodes"])

    def test_an_unknown_direction_raises_for_a_build(self):
        with pytest.raises(ValueError):
            service(row("J1", A, B, build_id="BLD")).get_build_graph(
                "BLD", direction="sideways"
            )


class TestWritePathIsInert:
    """The write-path methods degrade safely rather than pretending to work."""

    def test_emit_event_does_not_raise(self):
        assert service().emit_event({"anything": True}) is None

    def test_tag_searches_are_empty(self):
        svc = service(row("J1", A, B))
        assert svc.search_lineage_by_tags(["t"]) == (0, [])
        assert svc.count_events_by_tags(["t"]) == 0
        assert svc.count_runs_by_tags(["t"]) == 0

    def test_filter_unrecorded_fails_toward_rerecording(self):
        # The interface requires this: recording is idempotent, so returning the
        # candidates unchanged is the safe answer and never drops lineage.
        candidates = {"t1", "t2"}
        assert service().filter_unrecorded(candidates) == candidates
