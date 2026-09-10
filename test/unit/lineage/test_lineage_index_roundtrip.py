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

"""Write-then-read tests for the lineage index, from a real build.

The unit tests on either side feed the sink hand-written job entries and the
service hand-written rows, which is what makes them precise -- and also what let a
real bug through: the shared builders nest ``job_id`` under ``job_details``, so the
sink decomposed nothing and the index stayed empty while recording reported
success. Both sides passed. Only driving the real builders catches that, so these
tests start from stored builds and target runs and assert on the served graph.
"""

import os
import uuid as uuid_module
from datetime import datetime, timezone

import pytest

from gbcommon.uri.lh import LhURI
from gbserver.lineage.db_jobstats import DBLineageStore
from gbserver.lineage.db_service import DBLineageService
from gbserver.storage.artifact_registration import ArtifactRegistration
from gbserver.storage.sqlite.storage_factory import SqliteStorageFactory
from gbserver.storage.stored_build import StoredBuild
from gbserver.storage.stored_target_run import StoredTargetRun
from gbserver.types.artifact import ArtifactType
from gbserver.types.status import Status

pytestmark = pytest.mark.skipif(
    os.environ.get("SKIP_SQL_ADMIN_TESTS", "False").lower() == "true",
    reason="Don't want to run this in CICD.",
)

SPACE = "my-space"
USER = "alice"


class _AdminStorage:
    """The subset of SingletonAdminStorage the builders and the sink read."""

    def __init__(self, factory: SqliteStorageFactory, suffix: str) -> None:
        self.build_storage = factory.create_build_storage(table_name=f"rt_b_{suffix}")
        self.target_storage = factory.create_target_storage(table_name=f"rt_t_{suffix}")
        self.step_storage = factory.create_step_storage(table_name=f"rt_s_{suffix}")
        self.artifact_registry = factory.create_artifact_registry(
            table_name=f"rt_a_{suffix}"
        )
        self.lineage_row_storage = factory.create_lineage_row_storage(
            table_name=f"rt_l_{suffix}"
        )


@pytest.fixture(name="storage")
def storage_fixture():
    return _AdminStorage(SqliteStorageFactory(), uuid_module.uuid4().hex[:8])


def add_build(storage) -> StoredBuild:
    build = StoredBuild(
        uuid=str(uuid_module.uuid4()),
        name="my-build",
        space_name=SPACE,
        username=USER,
        source_uri="git://example/repo",
        created_time=datetime.now(timezone.utc),
    )
    storage.build_storage.add(build)
    return build


def add_table(storage, table: str) -> ArtifactRegistration:
    artifact = ArtifactRegistration(
        uuid=str(uuid_module.uuid4()),
        type=ArtifactType.TABLE,
        uri=LhURI.get_table_uri(table_name=table, namespace="ns"),
        space_name=SPACE,
        username=USER,
        name=table,
    )
    storage.artifact_registry.add(artifact)
    return artifact


def add_model(storage, build, target_uuid, label: str) -> ArtifactRegistration:
    artifact = ArtifactRegistration(
        uuid=str(uuid_module.uuid4()),
        type=ArtifactType.MODEL,
        uri=LhURI.get_model_uri(
            table_name="mdl_tbl",
            model_label=label,
            model_revision="v1",
            namespace="ns",
        ),
        space_name=SPACE,
        username=USER,
        name=label,
        created_by_build_id=build.uuid,
        created_by_target_id=target_uuid,
    )
    storage.artifact_registry.add(artifact)
    return artifact


def add_target(storage, build, inputs: dict, outputs: dict) -> StoredTargetRun:
    now = datetime.now(timezone.utc)
    target = StoredTargetRun(
        uuid=str(uuid_module.uuid4()),
        build_id=build.uuid,
        name="train",
        environment_uri="env://local",
        status=Status.SUCCESS,
        started_at=now,
        finished_at=now,
        input_artifacts=inputs,
        output_artifacts=outputs,
    )
    storage.target_storage.add(target)
    return target


def artifact_ids(graph: dict) -> set:
    return {n["id"] for n in graph["nodes"] if n["node_type"] == "artifact"}


class TestOneHop:
    """raw table -> [target run] -> trained model."""

    @pytest.fixture(name="wired", autouse=True)
    def wire(self, storage):
        self.build = add_build(storage)
        self.raw = add_table(storage, "raw_tbl")
        target_uuid = str(uuid_module.uuid4())
        self.model = add_model(storage, self.build, target_uuid, "trained")
        self.target = add_target(
            storage,
            self.build,
            inputs={"raw": self.raw.uuid},
            outputs={"model": [self.model.uuid]},
        )
        self.sink = DBLineageStore(storage=storage.lineage_row_storage)
        self.service = DBLineageService(storage=storage.lineage_row_storage)

    def test_recording_writes_a_row(self, storage):
        self.sink.add_jobstats_for_build(storage, self.build.uuid)
        # The regression guard: an empty index here means the job entries were
        # silently rejected, which is what a nested job_id caused.
        assert len(storage.lineage_row_storage.get_rows_by_build(self.build.uuid)) == 1

    def test_the_row_connects_the_two_artifacts(self, storage):
        self.sink.add_jobstats_for_build(storage, self.build.uuid)
        row = storage.lineage_row_storage.get_rows_by_build(self.build.uuid)[0]
        assert row.source == "table:staging/ns::raw_tbl"
        assert row.target == "model:staging/ns::trained|mdl_tbl"

    def test_the_row_carries_the_real_uris(self, storage):
        self.sink.add_jobstats_for_build(storage, self.build.uuid)
        row = storage.lineage_row_storage.get_rows_by_build(self.build.uuid)[0]
        assert row.source_uri == self.raw.uri
        assert row.target_uri == self.model.uri

    def test_downstream_reaches_the_model(self, storage):
        self.sink.add_jobstats_for_build(storage, self.build.uuid)
        graph = self.service.get_artifact_graph(
            artifact_url=self.raw.uri, direction="downstream"
        )
        assert graph is not None
        assert "model:staging/ns::trained|mdl_tbl" in artifact_ids(graph)

    def test_upstream_reaches_the_raw_table(self, storage):
        self.sink.add_jobstats_for_build(storage, self.build.uuid)
        graph = self.service.get_artifact_graph(
            artifact_url=self.model.uri, direction="upstream"
        )
        assert graph is not None
        assert "table:staging/ns::raw_tbl" in artifact_ids(graph)

    def test_the_served_graph_reports_the_registered_uri(self, storage):
        self.sink.add_jobstats_for_build(storage, self.build.uuid)
        graph = self.service.get_artifact_graph(
            artifact_url=self.raw.uri, direction="downstream"
        )
        node = next(n for n in graph["nodes"] if n["id"] == "table:staging/ns::raw_tbl")
        assert node["metadata"]["uri"] == self.raw.uri

    def test_recording_twice_does_not_duplicate(self, storage):
        self.sink.add_jobstats_for_build(storage, self.build.uuid)
        self.sink.add_jobstats_for_build(storage, self.build.uuid)
        assert len(storage.lineage_row_storage.get_rows_by_build(self.build.uuid)) == 1

    def test_a_recorded_target_is_filtered_out(self, storage):
        self.sink.add_jobstats_for_build(storage, self.build.uuid)
        assert self.sink.filter_unrecorded({self.target.uuid}) == set()

    def test_an_unrecorded_target_is_reported(self, storage):
        assert self.sink.filter_unrecorded({self.target.uuid}) == {self.target.uuid}

    def test_an_unknown_artifact_is_not_found(self, storage):
        self.sink.add_jobstats_for_build(storage, self.build.uuid)
        assert (
            self.service.get_artifact_graph(artifact_url="lh://prod/ns/tables/absent")
            is None
        )


class TestFanOutRows:
    """Two inputs and two outputs: the N*M decomposition, end to end."""

    def test_two_by_two_yields_four_rows(self, storage):
        build = add_build(storage)
        in_a = add_table(storage, "in_a")
        in_b = add_table(storage, "in_b")
        target_uuid = str(uuid_module.uuid4())
        out_a = add_model(storage, build, target_uuid, "out_a")
        out_b = add_model(storage, build, target_uuid, "out_b")
        target = add_target(
            storage,
            build,
            inputs={"a": in_a.uuid, "b": in_b.uuid},
            outputs={"models": [out_a.uuid, out_b.uuid]},
        )

        sink = DBLineageStore(storage=storage.lineage_row_storage)
        sink.add_jobstats_for_build(storage, build.uuid)

        rows = storage.lineage_row_storage.get_rows_by_build(build.uuid)
        assert len(rows) == 4
        # One execution, so every row shares the job identity -- which is what
        # keeps the flattening recoverable.
        assert len({r.job_id for r in rows}) == 1
        assert {r.target_run_uuid for r in rows} == {target.uuid}

    def test_the_graph_regroups_them_into_one_run(self, storage):
        build = add_build(storage)
        in_a = add_table(storage, "in_a")
        in_b = add_table(storage, "in_b")
        target_uuid = str(uuid_module.uuid4())
        out_a = add_model(storage, build, target_uuid, "out_a")
        add_target(
            storage,
            build,
            inputs={"a": in_a.uuid, "b": in_b.uuid},
            outputs={"models": [out_a.uuid]},
        )

        sink = DBLineageStore(storage=storage.lineage_row_storage)
        sink.add_jobstats_for_build(storage, build.uuid)

        service = DBLineageService(storage=storage.lineage_row_storage)
        graph = service.get_artifact_graph(artifact_url=out_a.uri, direction="upstream")
        runs = [n for n in graph["nodes"] if n["node_type"] == "run"]
        assert len(runs) == 1
        assert artifact_ids(graph) >= {
            "table:staging/ns::in_a",
            "table:staging/ns::in_b",
        }


class TestChainAcrossBuilds:
    """Lineage that crosses builds is one chain, which is the point of the index."""

    def test_a_two_build_chain_is_walkable(self, storage):
        # build 1: raw -> mid ;  build 2: mid -> final
        build1 = add_build(storage)
        raw = add_table(storage, "raw_tbl")
        t1_uuid = str(uuid_module.uuid4())
        mid = add_model(storage, build1, t1_uuid, "mid")
        add_target(
            storage, build1, inputs={"raw": raw.uuid}, outputs={"mid": [mid.uuid]}
        )

        build2 = add_build(storage)
        t2_uuid = str(uuid_module.uuid4())
        final = add_model(storage, build2, t2_uuid, "final")
        add_target(
            storage, build2, inputs={"mid": mid.uuid}, outputs={"final": [final.uuid]}
        )

        sink = DBLineageStore(storage=storage.lineage_row_storage)
        sink.add_jobstats_for_build(storage, build1.uuid)
        sink.add_jobstats_for_build(storage, build2.uuid)

        service = DBLineageService(storage=storage.lineage_row_storage)
        graph = service.get_artifact_graph(
            artifact_url=raw.uri, direction="downstream", max_depth=10
        )
        assert artifact_ids(graph) >= {
            "table:staging/ns::raw_tbl",
            "model:staging/ns::mid|mdl_tbl",
            "model:staging/ns::final|mdl_tbl",
        }

    def test_a_build_scoped_graph_stays_inside_its_build(self, storage):
        build1 = add_build(storage)
        raw = add_table(storage, "raw_tbl")
        t1_uuid = str(uuid_module.uuid4())
        mid = add_model(storage, build1, t1_uuid, "mid")
        add_target(
            storage, build1, inputs={"raw": raw.uuid}, outputs={"mid": [mid.uuid]}
        )

        build2 = add_build(storage)
        t2_uuid = str(uuid_module.uuid4())
        final = add_model(storage, build2, t2_uuid, "final")
        add_target(
            storage, build2, inputs={"mid": mid.uuid}, outputs={"final": [final.uuid]}
        )

        sink = DBLineageStore(storage=storage.lineage_row_storage)
        sink.add_jobstats_for_build(storage, build1.uuid)
        sink.add_jobstats_for_build(storage, build2.uuid)

        service = DBLineageService(storage=storage.lineage_row_storage)
        scoped = service.get_build_graph(
            build1.uuid, direction="downstream", within_build_only=True
        )
        assert "model:staging/ns::final|mdl_tbl" not in artifact_ids(scoped)

        crossing = service.get_build_graph(build1.uuid, direction="downstream")
        assert "model:staging/ns::final|mdl_tbl" in artifact_ids(crossing)


class TestErrors:
    def test_an_unknown_build_raises(self, storage):
        sink = DBLineageStore(storage=storage.lineage_row_storage)
        with pytest.raises(ValueError):
            sink.add_jobstats_for_build(storage, "no-such-build")

    def test_a_build_with_no_targets_raises(self, storage):
        # Matches the W&B sink, so the reconciler sees one behaviour either way.
        build = add_build(storage)
        sink = DBLineageStore(storage=storage.lineage_row_storage)
        with pytest.raises(ValueError):
            sink.add_jobstats_for_build(storage, build.uuid)
