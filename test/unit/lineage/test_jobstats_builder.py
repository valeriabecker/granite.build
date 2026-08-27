from libgbtest.lineage.mock_lineage_service import MockLineageService
from libgbtest.utils import AbstractSingletonStorageUsingTest

from gbserver.lineage.jobstats_builder import (
    _MAX_VISITED_RUNS,
    build_events_for_target,
    traverse_lineage_graph,
)
from gbserver.lineage.noop_jobstats import NoopLineageStore
from gbserver.lineage.wandb_jobstats import WandBLineageStore
from gbserver.storage.artifact_registration import (
    ArtifactRegistration,
    ArtifactRegistrationStatus,
)
from gbserver.storage.stored_build import StoredBuild
from gbserver.storage.stored_target_run import StoredTargetRun
from gbserver.types.artifact import ArtifactType
from gbserver.types.status import Status
from gbserver.utils.utils import get_utc_time


def _normalize_run_ids(value):
    """Replace every ``runId`` with a placeholder, recursively.

    Run ids are fresh random uuids by design (a deleted wandb run must be
    re-creatable, so the id cannot be derived from the target/output), which makes
    them the one field that legitimately differs between two builds of the same
    event. Blanking them keeps these tests asserting what they are about -- that
    both backends delegate to the same builder and so produce the same event
    *shape* -- instead of failing on the intended nondeterminism.
    """
    if isinstance(value, dict):
        return {
            k: ("<runId>" if k == "runId" else _normalize_run_ids(v))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_normalize_run_ids(v) for v in value]
    return value


def _make_build(storage, build_id: str, space_name: str = "test-space") -> StoredBuild:
    build = StoredBuild(
        uuid=build_id,
        name=f"build-{build_id}",
        space_name=space_name,
        source_uri="local://test",
        username="test-user",
        status=Status.SUCCESS,
    )
    storage.build_storage.add(build)
    return build


def _make_artifact(
    storage,
    artifact_id: str,
    created_by_build_id: str = "",
    created_by_target_id: str = "",
) -> ArtifactRegistration:
    artifact = ArtifactRegistration(
        uuid=artifact_id,
        name=f"artifact-{artifact_id}",
        type=ArtifactType.DATASET,
        uri=f"s3://test/{artifact_id}",
        space_name="test-space",
        username="test-user",
        status=ArtifactRegistrationStatus.SUCCESS,
        created_by_build_id=created_by_build_id,
        created_by_target_id=created_by_target_id,
    )
    storage.artifact_registry.add(artifact)
    return artifact


def _make_target(
    storage,
    target_id: str,
    build_id: str,
    input_artifacts: dict | None = None,
    output_artifacts: dict | None = None,
) -> StoredTargetRun:
    now = get_utc_time()
    target = StoredTargetRun(
        uuid=target_id,
        build_id=build_id,
        environment_uri="local://test-env",
        name=f"target-{target_id}",
        status=Status.SUCCESS,
        input_artifacts=input_artifacts or {},
        output_artifacts=output_artifacts or {},
        started_at=now,
        finished_at=now,
        target_hash=f"hash-{target_id}",
    )
    storage.target_storage.add(target)
    return target


class TestBuildEventsForTargetSharedAcrossBackends(AbstractSingletonStorageUsingTest):
    """build_events_for_target produces identical output via wandb or noop."""

    def test_identical_output_wandb_and_noop(self):
        build = _make_build(self.storage, "b1")
        _make_artifact(self.storage, "in1")
        _make_artifact(
            self.storage, "out1", created_by_build_id="b1", created_by_target_id="t1"
        )
        target = _make_target(
            self.storage,
            "t1",
            "b1",
            input_artifacts={"in": "in1"},
            output_artifacts={"out": ["out1"]},
        )

        direct_events, direct_dict = build_events_for_target(
            self.storage, build, target
        )

        wandb_store = WandBLineageStore.__new__(WandBLineageStore)
        wandb_store._service = MockLineageService()
        wandb_events, wandb_dict = wandb_store.create_jobstats_for_target(
            self.storage, target, build
        )

        noop_store = NoopLineageStore()
        noop_events, noop_dict = noop_store.create_jobstats_for_target(
            self.storage, target, build
        )

        assert (
            _normalize_run_ids(direct_events)
            == _normalize_run_ids(wandb_events)
            == _normalize_run_ids(noop_events)
        )
        assert (
            _normalize_run_ids(direct_dict)
            == _normalize_run_ids(wandb_dict)
            == _normalize_run_ids(noop_dict)
        )
        assert len(direct_events) == 1


class TestNoopReturnsRealEvents(AbstractSingletonStorageUsingTest):
    def test_returns_non_empty_events(self):
        build = _make_build(self.storage, "b1")
        _make_artifact(self.storage, "in1")
        _make_artifact(
            self.storage, "out1", created_by_build_id="b1", created_by_target_id="t1"
        )
        target = _make_target(
            self.storage,
            "t1",
            "b1",
            input_artifacts={"in": "in1"},
            output_artifacts={"out": ["out1"]},
        )

        store = NoopLineageStore()
        events, events_dict = store.create_jobstats_for_target(
            self.storage, target, build
        )

        assert events != []
        assert events_dict != {}
        assert len(events[0]["inputs"]) == 1
        assert len(events[0]["outputs"]) == 1


class TestTraverseLineageGraph(AbstractSingletonStorageUsingTest):
    def _make_chain(self):
        """A -> B -> C, joined via shared artifact UUIDs."""
        build_a = _make_build(self.storage, "build-a")
        build_b = _make_build(self.storage, "build-b")
        build_c = _make_build(self.storage, "build-c")

        _make_artifact(
            self.storage,
            "a-out",
            created_by_build_id="build-a",
            created_by_target_id="target-a",
        )
        _make_artifact(
            self.storage,
            "b-out",
            created_by_build_id="build-b",
            created_by_target_id="target-b",
        )

        target_a = _make_target(
            self.storage, "target-a", "build-a", output_artifacts={"out": ["a-out"]}
        )
        target_b = _make_target(
            self.storage,
            "target-b",
            "build-b",
            input_artifacts={"in": "a-out"},
            output_artifacts={"out": ["b-out"]},
        )
        target_c = _make_target(
            self.storage, "target-c", "build-c", input_artifacts={"in": "b-out"}
        )
        return build_a, build_b, build_c, target_a, target_b, target_c

    def test_single_build_no_cross_build_neighbors(self):
        build = _make_build(self.storage, "solo-build")
        _make_target(self.storage, "solo-target", "solo-build")

        graph = traverse_lineage_graph(
            self.storage, "solo-build", direction="both", max_depth=10
        )

        assert graph["root_build_id"] == "solo-build"
        assert len(graph["targets"]) == 1
        assert graph["truncated"] is False
        assert graph["expandable"] == []

    def test_upstream_from_b_includes_a(self):
        self._make_chain()
        graph = traverse_lineage_graph(
            self.storage, "build-b", direction="upstream", max_depth=10
        )
        assert any("target-a" in str(t) for t in graph["targets"])

    def test_downstream_from_a_includes_b(self):
        self._make_chain()
        graph = traverse_lineage_graph(
            self.storage, "build-a", direction="downstream", max_depth=10
        )
        assert any("target-b" in str(t) for t in graph["targets"])

    def test_both_from_b_includes_a_and_c(self):
        self._make_chain()
        graph = traverse_lineage_graph(
            self.storage, "build-b", direction="both", max_depth=10
        )
        serialized = str(graph["targets"])
        assert "target-a" in serialized
        assert "target-c" in serialized

    def test_max_depth_1_truncates(self):
        self._make_chain()
        graph = traverse_lineage_graph(
            self.storage, "build-a", direction="downstream", max_depth=1
        )
        assert graph["truncated"] is True

    def test_missing_build_returns_empty_graph(self):
        graph = traverse_lineage_graph(
            self.storage, "does-not-exist", direction="both", max_depth=10
        )
        assert graph["targets"] == []
        assert graph["truncated"] is False
        assert graph["expandable"] == []

    def test_expandable_reports_boundary_not_beyond(self):
        self._make_chain()
        # max_depth=1 from A downstream: B (1 hop) is included in targets;
        # C (2 hops) is beyond the cap and must show up as expandable instead.
        graph = traverse_lineage_graph(
            self.storage, "build-a", direction="downstream", max_depth=1
        )
        expandable_target_ids = {e["target_id"] for e in graph["expandable"]}
        assert "target-c" in expandable_target_ids
        assert "target-b" not in expandable_target_ids
        serialized = str(graph["targets"])
        assert "target-b" in serialized
        assert "target-c" not in serialized
        for e in graph["expandable"]:
            assert e["direction"] == "downstream"

    def test_expandable_empty_when_depth_reaches_end(self):
        self._make_chain()
        graph = traverse_lineage_graph(
            self.storage, "build-a", direction="downstream", max_depth=10
        )
        assert graph["expandable"] == []
        assert graph["truncated"] is False

    def test_leaf_run_not_in_expandable_at_depth_cap(self):
        self._make_chain()
        # target-c has no downstream consumers; even at max_depth=2 (exactly
        # reaching it), it must not appear as an expandable node.
        graph = traverse_lineage_graph(
            self.storage, "build-a", direction="downstream", max_depth=2
        )
        expandable_target_ids = {e["target_id"] for e in graph["expandable"]}
        assert "target-c" not in expandable_target_ids
        assert graph["truncated"] is False

    def test_full_map_small_graph(self):
        self._make_chain()
        graph = traverse_lineage_graph(
            self.storage, "build-a", direction="downstream", max_depth=-1
        )
        assert graph["truncated"] is False
        assert graph["expandable"] == []
        serialized = str(graph["targets"])
        assert "target-b" in serialized
        assert "target-c" in serialized

    def test_full_map_both_directions(self):
        self._make_chain()
        graph = traverse_lineage_graph(
            self.storage, "build-b", direction="both", max_depth=-1
        )
        assert graph["truncated"] is False
        serialized = str(graph["targets"])
        assert "target-a" in serialized
        assert "target-c" in serialized

    def test_safety_cap_truncates_full_map(self, monkeypatch):
        # Build a longer chain of 5 hops and monkeypatch the cap down to 3
        # visited runs to prove the cap behaves like a depth cutoff.
        import gbserver.lineage.jobstats_builder as jobstats_builder

        monkeypatch.setattr(jobstats_builder, "_MAX_VISITED_RUNS", 3)

        prev_artifact = None
        for i in range(5):
            build_id = f"chain-build-{i}"
            target_id = f"chain-target-{i}"
            _make_build(self.storage, build_id)
            out_artifact_id = f"chain-out-{i}"
            _make_artifact(
                self.storage,
                out_artifact_id,
                created_by_build_id=build_id,
                created_by_target_id=target_id,
            )
            inputs = {"in": prev_artifact} if prev_artifact else {}
            _make_target(
                self.storage,
                target_id,
                build_id,
                input_artifacts=inputs,
                output_artifacts={"out": [out_artifact_id]},
            )
            prev_artifact = out_artifact_id

        graph = jobstats_builder.traverse_lineage_graph(
            self.storage, "chain-build-0", direction="downstream", max_depth=-1
        )
        assert graph["truncated"] is True
        assert len(graph["targets"]) <= 3
        assert len(graph["expandable"]) >= 1


class TestGetLineageGraphDelegation(AbstractSingletonStorageUsingTest):
    def test_noop_and_wandb_delegate_identically(self):
        build = _make_build(self.storage, "b1")
        _make_target(self.storage, "t1", "b1")

        noop_store = NoopLineageStore()
        wandb_store = WandBLineageStore.__new__(WandBLineageStore)
        wandb_store._service = MockLineageService()

        noop_result = noop_store.get_lineage_graph(self.storage, "b1", "both", 10)
        wandb_result = wandb_store.get_lineage_graph(self.storage, "b1", "both", 10)

        assert _normalize_run_ids(noop_result) == _normalize_run_ids(wandb_result)
