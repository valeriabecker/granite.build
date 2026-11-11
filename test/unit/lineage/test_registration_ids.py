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

"""Registration uuids live under ``artifact_id``, not ``build_id`` (#583).

Records and W&B runs written before #583 put a registered artifact's uuid in
``build_id``. Every test here runs both shapes -- ``legacy`` and ``new`` --
through the real sink, the real indexer and the served graph and job detail,
so the guards for old data are checked end to end rather than per helper.
"""

import copy
import os
import uuid as uuid_module
from unittest.mock import MagicMock

import pytest
from libgbtest.lineage.lineage import AbstractLineageTablesTest

from gbcommon.uri.lh import LhURI
from gbserver.lineage.attributes import origin_id
from gbserver.lineage.db_jobstats import (
    DBLineageStore,
    migrate_registration_build_ids,
)
from gbserver.lineage.db_service import DBLineageService
from gbserver.lineage.row_indexing import LineageRowIndexer
from gbserver.lineage.uri_normalize import normalize_uri
from gbserver.lineage.wandb_jobstats import WandBLineageStore
from gbserver.storage.artifact_registration import ArtifactRegistration
from gbserver.storage.lineage_row_storage import row_tags
from gbserver.storage.stored_lineage_row import JobStore
from gbserver.types.artifact import ArtifactType

pytestmark = pytest.mark.skipif(
    os.environ.get("SKIP_SQL_ADMIN_TESTS", "False").lower() == "true",
    reason="Don't want to run this in CICD.",
)

SPACE = "my-space"
USER = "alice"


def _table(storage, name: str) -> ArtifactRegistration:
    artifact = ArtifactRegistration(
        uuid=str(uuid_module.uuid4()),
        type=ArtifactType.TABLE,
        uri=LhURI.get_table_uri(table_name=name, namespace="ns"),
        space_name=SPACE,
        username=USER,
        name=name,
    )
    storage.artifact_registry.add(artifact)
    return artifact


def _legacy_event(event: dict, uuid: str) -> dict:
    """The event as the pre-#583 builder emitted it: ``build_id`` tagged too."""
    legacy = copy.deepcopy(event)
    legacy["run"]["facets"]["tags"]["build_id"] = uuid
    return legacy


class _RegistrationTest(AbstractLineageTablesTest):
    @pytest.fixture(name="storage")
    def storage_fixture(self):
        return self.storage

    @pytest.fixture(autouse=True)
    def wire(self, storage):
        self.source = _table(storage, "raw_tbl")
        self.artifact = _table(storage, "registered_tbl")
        self.sink = DBLineageStore(job_storage=storage.lineage_job_storage)
        self.indexer = LineageRowIndexer(storage=storage.lineage_row_storage)
        self.service = DBLineageService(storage=storage.lineage_row_storage)

    # -- writing each shape ------------------------------------------------

    def record_new(self):
        self.sink.add_jobstats_for_original_artifact(self.artifact, [self.source])

    def record_legacy(self):
        """Write the record exactly as the sink did before #583."""
        event = self.sink.create_jobstats_for_original_artifact(
            self.artifact, [self.source]
        )
        uuid = self.artifact.uuid
        self.sink._write_job(
            _legacy_event(event, uuid), build_id=uuid, target_run_uuid=""
        )

    def record(self, shape: str):
        {"new": self.record_new, "legacy": self.record_legacy}[shape]()

    def index(self, storage):
        for page in storage.lineage_job_storage.get_paged():
            self.indexer.index_job_records(list(page))

    # -- reading -----------------------------------------------------------

    def job(self, storage):
        return storage.lineage_job_storage.get_job(self.artifact.uuid)

    def rows(self, storage):
        return storage.lineage_row_storage.get_rows_by_job(self.artifact.uuid)

    def detail(self, storage):
        return self.service.get_job_detail(
            self.artifact.uuid, authorize_build=lambda _b: None, admin_storage=storage
        )

    def assert_index_is_correct(self, storage):
        """The rows, the graph and the job detail a registration should have."""
        rows = self.rows(storage)
        assert rows, "the registration was not indexed"
        for row in rows:
            assert row.attributes["retrieve"] == {"job_id": self.artifact.uuid}
            assert "build_id" not in row_tags(row)
            assert row_tags(row).get("artifact_id") == self.artifact.uuid
        graph = self.service.query_graph(uri=self.source.uri, direction="downstream")
        ids = {n["id"] for n in graph["nodes"] if n["node_type"] == "artifact"}
        assert normalize_uri(self.artifact.uri) in ids
        detail = self.detail(storage)
        assert detail["detail_available"] is True, detail.get("detail_error")


SHAPES = ["new", "legacy"]


class TestRecording(_RegistrationTest):
    def test_a_new_record_keeps_the_uuid_out_of_build_id(self, storage):
        self.record_new()
        job = self.job(storage)
        assert origin_id(job.attributes, "build_id") == ""
        assert origin_id(job.attributes, "artifact_id") == self.artifact.uuid
        assert f"artifact_id={self.artifact.uuid}" in job.tags
        assert not any(t.startswith("build_id=") for t in job.tags)

    @pytest.mark.parametrize("shape", SHAPES)
    def test_count_finds_both_shapes(self, storage, shape):
        self.record(shape)
        assert self.sink.count_release_ids(self.artifact.uuid) == 1

    def test_count_of_a_build_ignores_registrations(self, storage):
        self.record_legacy()
        self.record_new()  # merges into the same job
        assert self.sink.count_release_ids(str(uuid_module.uuid4())) == 0


class TestIndexing(_RegistrationTest):
    @pytest.mark.parametrize("shape", SHAPES)
    def test_the_indexer_handles_both_shapes(self, storage, shape):
        self.record(shape)
        self.index(storage)
        self.assert_index_is_correct(storage)

    def test_a_legacy_wandb_run_is_indexed_like_a_new_one(self, storage):
        # The W&B indexer passes the run's build_id tag straight through, and
        # W&B runs are never migrated: the guard in write_rows is all there is.
        event = self.sink.create_jobstats_for_original_artifact(
            self.artifact, [self.source]
        )
        uuid = self.artifact.uuid
        self.indexer.index_job_entry(
            _legacy_event(event, uuid),
            build_id=uuid,
            target_run_uuid="",
            job_store=JobStore.WANDB,
        )
        rows = self.rows(storage)
        assert rows
        for row in rows:
            assert row.attributes["retrieve"] == {"job_id": uuid}
            assert "build_id" not in row_tags(row)


class TestMigration(_RegistrationTest):
    def index_pre_guard(self, storage):
        """Index the legacy record as the indexer did before #583's guard."""
        from gbserver.lineage import row_indexing

        original = row_indexing._without_legacy_registration_build_id
        row_indexing._without_legacy_registration_build_id = lambda j, b: (j, b)
        try:
            self.index(storage)
        finally:
            row_indexing._without_legacy_registration_build_id = original

    def migrate(self, storage, **kwargs):
        return migrate_registration_build_ids(
            storage.lineage_job_storage,
            row_storage=storage.lineage_row_storage,
            **kwargs,
        )

    def test_old_rows_are_wrong_before_migrating(self, storage):
        # The baseline the migration exists for: job detail looks for a build.
        self.record_legacy()
        self.index_pre_guard(storage)
        assert self.rows(storage)[0].attributes["retrieve"]["build_id"]
        assert self.detail(storage)["detail_available"] is False

    def test_migration_fixes_the_record_and_its_rows(self, storage):
        self.record_legacy()
        self.index_pre_guard(storage)
        jobs_and_rows = 1 + len(self.rows(storage))
        assert self.migrate(storage, dry_run=True) == jobs_and_rows
        assert self.migrate(storage) == jobs_and_rows

        job = self.job(storage)
        assert origin_id(job.attributes, "build_id") == ""
        assert origin_id(job.attributes, "artifact_id") == self.artifact.uuid
        assert not any(t.startswith("build_id=") for t in job.tags)
        assert self.sink.count_release_ids(self.artifact.uuid) == 1
        self.assert_index_is_correct(storage)

    def test_re_indexing_after_migrating_keeps_the_rows_correct(self, storage):
        self.record_legacy()
        self.index_pre_guard(storage)
        self.migrate(storage)
        self.index(storage)
        self.assert_index_is_correct(storage)

    def test_migration_is_idempotent(self, storage):
        self.record_legacy()
        self.index(storage)
        self.migrate(storage)
        assert self.migrate(storage) == 0

    def test_dry_run_writes_nothing(self, storage):
        self.record_legacy()
        self.migrate(storage, dry_run=True)
        assert origin_id(self.job(storage).attributes, "build_id") == self.artifact.uuid

    def test_new_records_are_left_alone(self, storage):
        self.record_new()
        self.index(storage)
        assert self.migrate(storage) == 0
        self.assert_index_is_correct(storage)


class TestJobsWithNeitherId(_RegistrationTest):
    """Jobs with no build_id and no artifact_id: imports, external runs."""

    def record_bare(self):
        event = self.sink.create_jobstats_for_original_artifact(
            self.artifact, [self.source]
        )
        bare = copy.deepcopy(event)
        bare["run"]["facets"]["tags"].pop("artifact_id", None)
        self.sink._write_job(bare, build_id="", target_run_uuid="")

    def test_they_are_never_counted(self, storage):
        self.record_bare()
        assert self.sink.count_release_ids("") == 0
        assert self.sink.count_release_ids(self.artifact.uuid) == 0

    def test_they_are_indexed_by_job_id(self, storage):
        self.record_bare()
        self.index(storage)
        rows = self.rows(storage)
        assert rows
        for row in rows:
            assert row.attributes["retrieve"] == {"job_id": self.artifact.uuid}
            assert "build_id" not in row_tags(row)
            assert "artifact_id" not in row_tags(row)

    def test_a_wandb_run_with_neither_is_indexed_untouched(self, storage):
        event = self.sink.create_jobstats_for_original_artifact(
            self.artifact, [self.source]
        )
        bare = copy.deepcopy(event)
        bare["run"]["facets"]["tags"].pop("artifact_id", None)
        self.indexer.index_job_entry(
            bare, build_id="", target_run_uuid="", job_store=JobStore.WANDB
        )
        assert self.rows(storage)

    def test_migration_leaves_them_alone(self, storage):
        self.record_bare()
        self.index(storage)
        before = (
            self.job(storage).attributes,
            [r.attributes for r in self.rows(storage)],
        )
        assert (
            migrate_registration_build_ids(
                storage.lineage_job_storage, row_storage=storage.lineage_row_storage
            )
            == 0
        )
        after = (
            self.job(storage).attributes,
            [r.attributes for r in self.rows(storage)],
        )
        assert before == after


class TestWandBCount:
    def test_an_empty_release_id_counts_nothing(self):
        store = WandBLineageStore.__new__(WandBLineageStore)
        store._service = MagicMock()
        assert store.count_release_ids("") == 0
        store._service.count_runs_by_tags.assert_not_called()

    def test_counts_runs_tagged_either_way(self):
        store = WandBLineageStore.__new__(WandBLineageStore)
        store._service = MagicMock()
        store._service.count_runs_by_tags.return_value = 1
        assert store.count_release_ids("U", target_id="T") == 1
        store._service.count_runs_by_tags.assert_called_once_with(
            ["build_id=U", "artifact_id=U"], required_tags=["target_id=T"]
        )

    def test_new_registration_runs_have_no_build_id_tag(self):
        store = WandBLineageStore.__new__(WandBLineageStore)
        artifact = ArtifactRegistration(
            uuid="U",
            type=ArtifactType.TABLE,
            uri=LhURI.get_table_uri(table_name="t", namespace="ns"),
            space_name=SPACE,
            username=USER,
            name="t",
        )
        event = store._build_event_for_artifact(artifact, [])
        tags = event["run"]["facets"]["tags"]
        assert "build_id" not in tags
        assert tags["artifact_id"] == "U"
