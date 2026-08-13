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

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from gbserver.lineage.jobstats import ILineageStore
from gbserver.lineage.jobstats_builder import (
    build_event_for_artifact,
    build_events_for_target,
    traverse_lineage_graph,
)
from gbserver.lineage.openlineage_service import LineageService, LineageServiceFactory
from gbserver.storage.artifact_registration import ArtifactRegistration
from gbserver.storage.singleton_storage import SingletonAdminStorage
from gbserver.storage.stored_build import StoredBuild
from gbserver.storage.stored_target_run import StoredTargetRun
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


class WandBLineageStore(ILineageStore):

    def __init__(self) -> None:
        self._service: LineageService = LineageServiceFactory.create("wandb")

    def add_jobstats_for_build(
        self, storage: SingletonAdminStorage, build_id: str
    ) -> None:
        build = storage.build_storage.get_by_uuid(build_id)
        if build is None:
            raise ValueError(f"Build with id {build_id} was not found")
        assert isinstance(build, StoredBuild)

        targets = storage.target_storage.get_by_where({"build_id": build_id})
        count = 0
        for target in targets:
            assert isinstance(target, StoredTargetRun)
            self.__add_jobstats_for_target(storage, build, target)
            count += 1
        if count == 0:
            raise ValueError(f"Zero targets found in build with id {build_id}")

    def add_jobstats_for_build_target(
        self, storage: SingletonAdminStorage, build_id: str, target_id: str
    ) -> None:
        build = storage.build_storage.get_by_uuid(build_id)
        if build is None:
            raise ValueError(f"Build with id {build_id} was not found")
        assert isinstance(build, StoredBuild)

        targets = storage.target_storage.get_by_where(
            {"build_id": build_id, "uuid": target_id}
        )
        count = 0
        for target in targets:
            assert isinstance(target, StoredTargetRun)
            self.__add_jobstats_for_target(storage, build, target)
            count += 1
        if count == 0:
            raise ValueError(f"Zero targets found in build with id {build_id}")

    def __add_jobstats_for_target(
        self,
        storage: SingletonAdminStorage,
        build: StoredBuild,
        targetrun: StoredTargetRun,
    ) -> None:
        events, _ = self.create_jobstats_for_target(storage, targetrun, build)
        for event in events:
            self._service.emit_event(event)

    def add_jobstats_for_original_artifact(
        self,
        artifact: ArtifactRegistration,
        sources: list[ArtifactRegistration],
    ) -> None:
        event = self._build_event_for_artifact(artifact, sources)
        self._service.emit_event(event)

    def create_jobstats_for_target(
        self,
        storage: SingletonAdminStorage,
        targetrun: StoredTargetRun,
        build: Optional[StoredBuild] = None,
    ) -> Tuple[List[dict], Dict[str, List[dict]]]:
        if build is None:
            build_result = storage.build_storage.get_by_uuid(targetrun.build_id)
            if build_result is None:
                raise ValueError(
                    f"target's build could not be found under target's build id {targetrun.build_id}"
                )
            assert isinstance(build_result, StoredBuild)
            build = build_result

        if targetrun.build_id != build.uuid:
            raise ValueError(
                f"target's build id ({targetrun.build_id}) does not match that of the given build ({build.uuid})"
            )

        if targetrun.skipped_for_prerun_target_id:
            original = storage.target_storage.get_by_uuid(
                targetrun.skipped_for_prerun_target_id
            )
            if original is not None and isinstance(original, StoredTargetRun):
                targetrun = original.model_copy(
                    update={
                        "uuid": targetrun.uuid,
                        "build_id": targetrun.build_id,
                    }
                )
            else:
                logger.warning(
                    "Skipped target %s references unknown original %s",
                    targetrun.uuid,
                    targetrun.skipped_for_prerun_target_id,
                )

        return build_events_for_target(storage, build, targetrun)

    def create_jobstats_for_original_artifact(
        self,
        artifact: ArtifactRegistration,
        sources: list[ArtifactRegistration],
    ) -> dict:
        return self._build_event_for_artifact(artifact, sources)

    def get_lineage_graph(
        self,
        storage: SingletonAdminStorage,
        build_id: str,
        direction: str = "both",
        max_depth: int = 10,
    ) -> dict:
        return traverse_lineage_graph(storage, build_id, direction, max_depth)

    def count_release_ids(
        self, release_id: str, target_id: Optional[str] = None
    ) -> int:
        # One wandb run is created per (target, output artifact), so counting
        # runs tagged with this build_id (and optionally target_id) directly
        # yields the number of jobstats records without scanning run history.
        required = [f"target_id={target_id}"] if target_id else None
        return self._service.count_runs_by_tags(
            [f"build_id={release_id}"], required_tags=required
        )

    def does_release_id_exist(
        self,
        release_id: str,
        expected_count: int,
        target_id: Optional[str] = None,
    ) -> bool:
        count = self.count_release_ids(release_id, target_id)
        return count == expected_count

    def _build_event_for_artifact(
        self,
        artifact: ArtifactRegistration,
        sources: list[ArtifactRegistration],
    ) -> dict:
        return build_event_for_artifact(artifact, sources)
