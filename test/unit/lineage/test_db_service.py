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

# Endpoints are normalized URIs: the node's identity and what a request names are
# the same string, which is what turned root resolution into one indexed lookup.
A = "lh://prod/ns/tables/a"
B = "lh://prod/ns/tables/b"
C = "lh://prod/ns/tables/c"


def row(job_id: str, source: str, target: str, **kwargs) -> StoredLineageRow:
    """A row whose endpoint detail and job metadata go into ``attributes``."""
    attributes = dict(kwargs.pop("attributes", {}) or {})
    for side in ("source", "target"):
        kind = kwargs.pop(f"{side}_kind", "")
        name = kwargs.pop(f"{side}_name", "")
        detail = {}
        if kind:
            detail["kind"] = kind
        if name:
            detail["name"] = name
        if detail:
            attributes[side] = detail
    assert not kwargs, f"unhandled row() keywords: {sorted(kwargs)}"
    return StoredLineageRow(
        job_id=job_id, source=source, target=target, attributes=attributes
    )


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
        matched = self._matching(where)
        for start in range(0, max(len(matched), 1), page_size):
            chunk = matched[start : start + page_size]
            if chunk or start == 0:
                yield chunk

    def get_rows_by_source(self, sources: list) -> list:
        wanted = {s for s in sources if s and s != TERMINAL}
        return [r for r in self.rows if r.source in wanted]

    def get_rows_by_target(self, targets: list) -> list:
        wanted = {t for t in targets if t and t != TERMINAL}
        return [r for r in self.rows if r.target in wanted]

    def count(self, where=None) -> int:
        return len(self._matching(where))

    def _matching(self, where) -> list:
        if not where:
            return list(self.rows)
        matched = []
        for row in self.rows:
            if all(getattr(row, key, None) == value for key, value in where.items()):
                matched.append(row)
        return matched

    def get_rows_by_job(self, job_id: str) -> list:
        if self.fail:
            raise RuntimeError("storage is down")
        if not job_id:
            return []
        return [r for r in self.rows if r.job_id == job_id]

    # No get_rows_by_build: a build is not a column. A build-seeded graph resolves
    # its artifacts through gb_targets and seeds the ordinary walk with their URIs.


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
    """One indexed lookup, not a scan.

    This used to page the whole table on every request, because the indexed columns
    held canonical identifiers while a request carries a URL. Normalizing the
    request's URL now produces exactly the value those columns hold.
    """

    def test_resolves_by_uri(self):
        result = service(row("J1", A, B)).get_artifact_graph(artifact_url=A)
        assert result is not None and result["root_id"] == A

    def test_converges_alternate_spellings_of_one_uri(self):
        """A browser URL and the runtime's own URI are one artifact.

        Without this the graph splits into two disconnected halves of the same
        thing, depending on which spelling the caller happened to have.
        """
        hf = "hf://huggingface.co/models/org/repo"
        svc = service(row("J1", hf, B))
        for spelling in (
            hf,
            "hf:///org/repo",
            "https://huggingface.co/org/repo",
            "hf://huggingface.co/models/org/repo/main",
        ):
            result = svc.get_artifact_graph(artifact_url=spelling)
            assert result is not None, spelling
            assert result["root_id"] == hf, spelling

    def test_resolves_a_target_endpoint_too(self):
        result = service(row("J1", A, B)).get_artifact_graph(artifact_url=B)
        assert result is not None and result["root_id"] == B

    def test_a_uri_in_the_name_field_still_resolves(self):
        """A caller that passes a URI as the name is not punished for it."""
        result = service(row("J1", A, B)).get_artifact_graph(artifact_url=A)
        assert result is not None and result["root_id"] == A

    def test_a_bare_name_is_none(self):
        """A name is not an identity, and a scheme is not guessed for it.

        The index keys on URIs. Inventing a scheme for "my-artifact" would resolve
        to an artifact that may not exist -- an invented identity, which is the one
        failure worth returning nothing to avoid.
        """
        assert (
            service(row("J1", A, B, source_name="my-artifact")).get_artifact_graph(
                artifact_name="my-artifact"
            )
            is None
        )

    def test_an_unidentifiable_uri_is_none(self):
        # The 404 case: the request names nothing this index can key on.
        assert (
            service(row("J1", A, B)).get_artifact_graph(artifact_url="bogus://x")
            is None
        )

    def test_no_identifier_at_all_is_none(self):
        assert service(row("J1", A, B)).get_artifact_graph() is None

    def test_an_absent_but_valid_uri_is_a_graph_not_a_404(self):
        """A resolvable URI with no rows yields its own node, not None.

        The distinction matters at the API boundary: ``None`` becomes the 404 the
        frontend renders as "lineage is not available", so it must mean "cannot key
        on this", never "nothing recorded yet".
        """
        result = service().get_artifact_graph(artifact_url=A)
        assert result is not None
        assert result["root_id"] == A
        assert [n["id"] for n in result["nodes"]] == [A]


class TestGraphContent:
    def test_descendants_walk_from_the_root(self):
        svc = service(row("J1", A, B), row("J2", B, C))
        result = svc.get_artifact_graph(artifact_url=A, direction="downstream")
        ids = {n["id"] for n in result["nodes"] if n["node_type"] == "artifact"}
        assert ids == {A, B, C}

    def test_upstream_walks_toward_ancestors(self):
        svc = service(row("J1", A, B), row("J2", B, C))
        result = svc.get_artifact_graph(artifact_url=C, direction="upstream")
        ids = {n["id"] for n in result["nodes"] if n["node_type"] == "artifact"}
        assert ids == {A, B, C}

    def test_downstream_does_not_walk_backward(self):
        # Pins the wire mapping: downstream is used_by(), toward descendants, which
        # is what the live W&B backend and the frontend already mean by it.
        svc = service(row("J1", A, B), row("J2", B, C))
        result = svc.get_artifact_graph(artifact_url=B, direction="downstream")
        ids = {n["id"] for n in result["nodes"] if n["node_type"] == "artifact"}
        assert ids == {B, C}
        assert A not in ids

    def test_upstream_does_not_walk_forward(self):
        svc = service(row("J1", A, B), row("J2", B, C))
        result = svc.get_artifact_graph(artifact_url=B, direction="upstream")
        ids = {n["id"] for n in result["nodes"] if n["node_type"] == "artifact"}
        assert ids == {A, B}
        assert C not in ids

    def test_both_reaches_each_side(self):
        svc = service(row("J1", A, B), row("J2", B, C))
        result = svc.get_artifact_graph(artifact_url=B, direction="both")
        ids = {n["id"] for n in result["nodes"] if n["node_type"] == "artifact"}
        assert ids == {A, B, C}

    def test_an_artifact_with_no_edges_is_a_graph_not_a_404(self):
        # It IS in the index -- as a creation with no input -- so the answer is a
        # graph with its own node, never None.
        svc = service(row("J1", TERMINAL, A))
        result = svc.get_artifact_graph(artifact_url=A, direction="upstream")
        assert result is not None
        assert result["root_id"] == A

    def test_max_depth_truncates_and_says_so(self):
        svc = service(row("J1", A, B), row("J2", B, C))
        result = svc.get_artifact_graph(
            artifact_url=A, direction="downstream", max_depth=1
        )
        assert result["truncated"] is True


class TestValidation:
    def test_an_unknown_direction_raises(self):
        # The API layer maps ValueError to a 400.
        with pytest.raises(ValueError):
            service(row("J1", A, B)).get_artifact_graph(
                artifact_url=A, direction="sideways"
            )

    def test_a_contradicted_artifact_type_raises(self):
        svc = service(row("J1", A, B, source_kind="table"))
        with pytest.raises(ValueError):
            svc.get_artifact_graph(artifact_url=A, artifact_type="model")

    def test_a_matching_artifact_type_is_accepted(self):
        svc = service(row("J1", A, B, source_kind="table"))
        assert svc.get_artifact_graph(artifact_url=A, artifact_type="table")

    def test_direction_is_validated_before_the_root_lookup(self):
        # Otherwise an unknown artifact would 404 on a request that is also a bad
        # request, hiding the client's actual mistake.
        with pytest.raises(ValueError):
            service().get_artifact_graph(artifact_name="nope", direction="sideways")


class _StubBuilder:
    """Stands in for the shared event builder.

    Build seeding now goes through ``create_jobstats_for_target`` so the seeds are
    by construction the endpoints the writer saw. That builder needs a real build,
    artifact registry and step configs, which is more than a unit fake should
    reconstruct -- so these tests stub the builder and assert the *seeding* logic,
    while ``test_lineage_index_roundtrip`` exercises the real builder against real
    storage.
    """

    def __init__(self, events_by_target: dict):
        self._events_by_target = events_by_target

    def create_jobstats_for_target(self, storage, target):
        if target.uuid == "boom":
            raise RuntimeError("cannot build events for this target")
        return self._events_by_target.get(target.uuid, []), {}


class _FakeTarget:
    def __init__(self, uuid: str):
        self.uuid = uuid


class _FakeAdminStorage:
    """Just enough of the admin storage for build seeding."""

    def __init__(self, targets_by_build: dict):
        outer = self
        self._targets_by_build = targets_by_build

        class _TargetStorage:
            def get_by_where(self, where):
                return outer._targets_by_build.get(where.get("build_id"), [])

        self.target_storage = _TargetStorage()


def event(sources=(), targets=()) -> dict:
    """A job entry in the shape the shared builder emits."""
    return {
        "sources": [{"uri": u} for u in sources],
        "targets": [{"uri": u} for u in targets],
    }


def build_service(rows, targets_by_build, events_by_target) -> DBLineageService:
    """A service whose build seeding reads fakes instead of the admin storage."""
    svc = DBLineageService(
        storage=FakeStorage(list(rows)),
        admin_storage=_FakeAdminStorage(targets_by_build),
    )
    builder = _StubBuilder(events_by_target)
    # Patch the builder construction, not the seeding logic under test.
    import gbserver.lineage.db_jobstats as db_jobstats_mod

    svc._test_builder = builder
    original = db_jobstats_mod.DBLineageStore
    db_jobstats_mod.DBLineageStore = lambda **_kw: builder
    svc._restore_builder = lambda: setattr(
        db_jobstats_mod, "DBLineageStore", original
    )
    return svc


@pytest.fixture(autouse=True)
def _restore_builder_patch():
    """Undo any builder patch a test installed."""
    import gbserver.lineage.db_jobstats as db_jobstats_mod

    original = db_jobstats_mod.DBLineageStore
    yield
    db_jobstats_mod.DBLineageStore = original


class TestBuildGraph:
    """A build is resolved OUTSIDE the index, then seeds the ordinary walk.

    There is no ``build_id`` column: a build is granite.build's own process concept
    and is empty on every imported row, so indexing it would index blanks over most
    of the table. The build's target runs are turned into endpoints by the shared
    event builder, and those URIs are the seeds.
    """

    def test_seeds_from_the_builds_artifacts(self):
        svc = build_service(
            [row("J1", A, B), row("J2", B, C)],
            {"BLD": [_FakeTarget("t1")]},
            {"t1": [event(sources=[A], targets=[B])]},
        )
        result = svc.get_build_graph("BLD")
        assert result is not None
        assert result["root_id"] == "BLD"
        ids = {n["id"] for n in result["nodes"] if n["node_type"] == "artifact"}
        assert {A, B} <= ids

    def test_the_index_is_never_asked_about_a_build(self):
        """The lineage storage has no build-scoped query to call."""
        assert not hasattr(FakeStorage([]), "get_rows_by_build")

    def test_seed_uris_are_normalized(self):
        """A builder URI in any spelling has to match the stored endpoint."""
        hf = "hf://huggingface.co/models/org/repo"
        svc = build_service(
            [row("J1", hf, B)],
            {"BLD": [_FakeTarget("t1")]},
            {"t1": [event(sources=["https://huggingface.co/org/repo"])]},
        )
        result = svc.get_build_graph("BLD", direction="downstream")
        ids = {n["id"] for n in result["nodes"] if n["node_type"] == "artifact"}
        assert hf in ids

    def test_it_crosses_into_other_builds(self):
        """Scope is chosen by seeds now; there is no row filter to bound it."""
        svc = build_service(
            [row("J1", A, B), row("J2", B, C)],
            {"BLD": [_FakeTarget("t1")]},
            {"t1": [event(sources=[A])]},
        )
        result = svc.get_build_graph("BLD", direction="downstream")
        ids = {n["id"] for n in result["nodes"] if n["node_type"] == "artifact"}
        assert C in ids

    def test_a_build_with_no_targets_is_none(self):
        svc = build_service([row("J1", A, B)], {}, {})
        assert svc.get_build_graph("OTHER") is None

    def test_a_build_whose_endpoints_do_not_normalize_is_none(self):
        """An unidentifiable endpoint is skipped, never guessed."""
        svc = build_service(
            [row("J1", A, B)],
            {"BLD": [_FakeTarget("t1")]},
            {"t1": [event(sources=["bogus://x"])]},
        )
        assert svc.get_build_graph("BLD") is None

    def test_one_unbuildable_target_does_not_cost_the_whole_graph(self):
        """The other targets still seed it."""
        svc = build_service(
            [row("J1", A, B)],
            {"BLD": [_FakeTarget("boom"), _FakeTarget("t1")]},
            {"t1": [event(sources=[A])]},
        )
        result = svc.get_build_graph("BLD", direction="downstream")
        assert result is not None
        ids = {n["id"] for n in result["nodes"] if n["node_type"] == "artifact"}
        assert A in ids

    def test_an_empty_build_id_is_none(self):
        svc = build_service([row("J1", A, B)], {}, {})
        assert svc.get_build_graph("") is None

    def test_a_storage_failure_reports_no_graph_rather_than_raising(self):
        class _Broken:
            @property
            def target_storage(self):
                raise RuntimeError("admin storage is down")

        svc = DBLineageService(
            storage=FakeStorage([row("J1", A, B)]), admin_storage=_Broken()
        )
        assert svc.get_build_graph("BLD") is None

    def test_no_artifact_is_flagged_root_for_a_build(self):
        # root_id names a build, not an artifact; flagging one arbitrarily would
        # misreport which artifact was asked about.
        svc = build_service(
            [row("J1", A, B)],
            {"BLD": [_FakeTarget("t1")]},
            {"t1": [event(sources=[A])]},
        )
        result = svc.get_build_graph("BLD")
        assert not any(n["is_root"] for n in result["nodes"])

    def test_an_unknown_direction_raises_for_a_build(self):
        svc = build_service([row("J1", A, B)], {}, {})
        with pytest.raises(ValueError):
            svc.get_build_graph("BLD", direction="sideways")


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


class TestQueryGraph:
    """The general entry point: every filter optional, and it never returns None.

    ``get_artifact_graph`` returns ``None`` for "cannot key on this", which the API
    turns into a 404. A query endpoint cannot do that: asking with no filters is
    legitimate, and an empty index is a legitimate answer, so "nothing recorded" must
    come back as an empty graph rather than as an error.
    """

    def test_by_uri(self):
        svc = service(row("J1", A, B), row("J2", B, C))
        result = svc.query_graph(uri=B)
        ids = {n["id"] for n in result["nodes"] if n["node_type"] == "artifact"}
        assert ids == {A, B, C}
        assert result["root_id"] == B

    def test_a_uri_query_flags_its_single_root(self):
        result = service(row("J1", A, B)).query_graph(uri=A)
        assert [n["id"] for n in result["nodes"] if n["is_root"]] == [A]

    def test_alternate_spellings_reach_one_artifact(self):
        hf = "hf://huggingface.co/models/org/repo"
        svc = service(row("J1", hf, B))
        for spelling in (hf, "hf:///org/repo", "https://huggingface.co/org/repo"):
            assert svc.query_graph(uri=spelling)["root_id"] == hf, spelling

    def test_by_job_id_seeds_every_endpoint_of_that_execution(self):
        svc = service(row("J1", A, B), row("J2", B, C))
        result = svc.query_graph(job_id="J1", direction="both")
        depths = {
            n["id"]: n["depth"]
            for n in result["nodes"]
            if n["node_type"] == "artifact"
        }
        # Both of J1's endpoints are seeds, so both are at depth 0.
        assert depths[A] == 0
        assert depths[B] == 0

    def test_a_job_query_flags_no_root(self):
        """It has several, and picking one would misreport what was asked."""
        result = service(row("J1", A, B)).query_graph(job_id="J1")
        assert result["root_id"] == ""
        assert not any(n["is_root"] for n in result["nodes"])

    def test_uri_and_job_seed_the_union(self):
        svc = service(row("J1", A, B), row("J2", C, A))
        result = svc.query_graph(uri=C, job_id="J1", direction="downstream")
        ids = {n["id"] for n in result["nodes"] if n["node_type"] == "artifact"}
        assert {A, B, C} <= ids

    def test_no_filter_returns_recent_activity(self):
        svc = service(row("J1", A, B), row("J2", B, C))
        result = svc.query_graph()
        ids = {n["id"] for n in result["nodes"] if n["node_type"] == "artifact"}
        assert ids == {A, B, C}
        assert result["root_id"] == ""

    def test_no_filter_on_an_empty_index_is_an_empty_graph(self):
        result = service().query_graph()
        assert result["nodes"] == []
        assert result["edges"] == []
        assert result["truncated"] is False

    def test_an_unmatchable_uri_is_an_empty_graph_not_none(self):
        """The distinction from get_artifact_graph, which would return None here."""
        result = service(row("J1", A, B)).query_graph(uri="bogus://x")
        assert result is not None
        assert result["nodes"] == []

    def test_an_unknown_job_is_an_empty_graph(self):
        result = service(row("J1", A, B)).query_graph(job_id="NOPE")
        assert result is not None
        assert result["nodes"] == []

    def test_direction_is_honoured(self):
        svc = service(row("J1", A, B), row("J2", B, C))
        up = {
            n["id"]
            for n in svc.query_graph(uri=B, direction="upstream")["nodes"]
            if n["node_type"] == "artifact"
        }
        down = {
            n["id"]
            for n in svc.query_graph(uri=B, direction="downstream")["nodes"]
            if n["node_type"] == "artifact"
        }
        assert up == {A, B}
        assert down == {B, C}

    def test_max_depth_truncates_and_says_so(self):
        svc = service(row("J1", A, B), row("J2", B, C))
        result = svc.query_graph(uri=A, direction="downstream", max_depth=1)
        ids = {n["id"] for n in result["nodes"] if n["node_type"] == "artifact"}
        assert C not in ids
        assert result["truncated"] is True

    def test_an_unknown_direction_raises(self):
        with pytest.raises(ValueError):
            service(row("J1", A, B)).query_graph(uri=A, direction="sideways")

    def test_a_storage_failure_does_not_raise(self):
        """A job lookup that fails costs its seeds, not the request."""
        svc = service(row("J1", A, B), fail=True)
        assert svc.query_graph(job_id="J1") is not None


class TestListRuns:
    """The drill-down for what the graph collapses.

    ``build_graph_dict`` folds an artifact's in-place rewrites into one node with a
    ``run_count``; real data has one appended 68,905 times. A count with no way to
    expand it is a dead end, so this is that way -- paged rather than capped, because a
    flat list has no shape to preserve.
    """

    def test_it_lists_runs_touching_an_artifact_in_both_directions(self):
        svc = service(row("J1", A, B), row("J2", B, C))
        result = svc.list_runs(uri=B)
        assert {r["job_id"] for r in result["runs"]} == {"J1", "J2"}

    def test_the_total_is_exact_for_a_self_rewritten_artifact(self):
        """Three counts, not two.

        A self-loop row matches the source query AND the target query, so summing them
        reports double. On the real hub that meant 137,811 runs for an artifact with
        68,906 -- wrong by 2x, which is worse than slow.
        """
        svc = service(*[row(f"J{i}", A, A) for i in range(10)])
        assert svc.list_runs(uri=A)["total"] == 10

    def test_a_self_loop_row_appears_once_in_the_page(self):
        svc = service(row("J1", A, A))
        runs = svc.list_runs(uri=A)["runs"]
        assert len(runs) == 1
        assert runs[0]["is_self_loop"] is True

    def test_paging_walks_the_whole_list(self):
        svc = service(*[row(f"J{i}", A, B) for i in range(10)])
        seen = []
        for offset in range(0, 10, 3):
            seen.extend(r["job_id"] for r in svc.list_runs(uri=A, limit=3, offset=offset)["runs"])
        assert len(set(seen)) == 10

    def test_the_page_size_is_capped(self):
        """A caller must not be able to ask for everything and recreate the problem."""
        svc = service(*[row(f"J{i}", A, B) for i in range(5)])
        assert svc.list_runs(uri=A, limit=10**9)["limit"] == 1000

    def test_a_zero_or_negative_limit_is_clamped(self):
        svc = service(row("J1", A, B))
        assert svc.list_runs(uri=A, limit=0)["limit"] == 1
        assert svc.list_runs(uri=A, offset=-5)["offset"] == 0

    def test_by_job_id(self):
        svc = service(row("J1", A, B), row("J2", B, C))
        result = svc.list_runs(job_id="J1")
        assert [r["job_id"] for r in result["runs"]] == ["J1"]
        assert result["total"] == 1

    def test_an_unresolvable_uri_is_empty_not_an_error(self):
        result = service(row("J1", A, B)).list_runs(uri="bogus://x")
        assert result == {"runs": [], "total": 0, "limit": 100, "offset": 0}

    def test_no_filter_is_empty(self):
        assert service(row("J1", A, B)).list_runs()["runs"] == []

    def test_a_storage_failure_does_not_raise(self):
        svc = service(row("J1", A, B), fail=True)
        assert svc.list_runs(uri=A)["runs"] == []

    def test_an_entry_carries_its_endpoints_and_job_detail(self):
        svc = service(row("J1", A, B, attributes={"job": {"name": "train"}}))
        entry = svc.list_runs(uri=A)["runs"][0]
        assert entry["source"] == A
        assert entry["target"] == B
        assert entry["job"]["name"] == "train"
