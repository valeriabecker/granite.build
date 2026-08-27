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

import time
from typing import Callable, Dict, List, Optional, Tuple

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


# How long a "fully recorded in wandb" verdict stays trusted without re-asking
# wandb. The reconciler re-selects already-recorded targets on every scan by
# design (the watermark overlap window deliberately keeps the newest target in
# range), so in steady state the same candidate is verified every
# monitoring_interval — a network round-trip per scan to re-learn an unchanged
# fact. Caching the positive verdict collapses that to one call per TTL.
#
# Only *positive* verdicts are cached, and only for this long: a run deleted in
# wandb, or a target whose runs were only partially emitted, must eventually be
# noticed and re-recorded. The TTL bounds that staleness instead of making it
# permanent, and the cache is per-process, so a restart always re-verifies.
_RECORDED_CACHE_TTL_SECONDS = 6 * 60 * 60


class WandBLineageStore(ILineageStore):

    def __init__(self) -> None:
        self._service: LineageService = LineageServiceFactory.create("wandb")
        # (target uuid, expected run count) -> monotonic deadline after which the
        # verdict is re-checked. The expected count is part of the key, not just
        # the value: a verdict of "recorded" means "has all the runs we expected
        # *at the time we asked*", so a target that later grows an output (and so
        # expects more runs) must be re-checked rather than inherit the old
        # verdict. ``None`` is a distinct key, matching the presence-check
        # fallback for candidates with no expected count.
        # Monotonic, not wall-clock, so a system clock adjustment cannot expire
        # every entry at once or freeze them past the TTL.
        self._recorded_until: dict[tuple[str, Optional[int]], float] = {}

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
        if not events:
            # No events means emit_event is never called, yet the caller
            # (reconciler) will still mark the target recorded — a silent no-op
            # that leaves nothing in the backend. Surface it rather than hide it.
            logger.warning(
                "No lineage events built for target %s (name=%s) in build %s; "
                "nothing emitted to the lineage backend",
                targetrun.uuid,
                targetrun.name,
                build.uuid,
            )
            return
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

        # Every SUCCESS run is a real run with its own outputs (in-place retry keeps
        # both the FAILED and the SUCCESS run in one build), so lineage is built
        # directly from the target's own outputs.
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

    def filter_unrecorded(
        self,
        target_ids: set[str],
        expected_counts: Optional[dict[str, int]] = None,
        on_query_error: Optional[Callable[[Exception], None]] = None,
    ) -> set[str]:
        # Drop candidates whose "fully recorded" verdict is still within its TTL,
        # so a steady-state scan that re-selects the same target does not re-ask
        # wandb every interval (see _RECORDED_CACHE_TTL_SECONDS).
        now = time.monotonic()
        # Sweep before the per-candidate lookups. Pruning only the keys touched
        # below cannot bound the dict: the checkpoint advances forward only, so
        # once a recorded target falls behind the scan's lower bound it is never
        # selected again and its entry would outlive its deadline for the life of
        # the process -- one dead tuple per (target, count) ever recorded. The
        # sweep is what makes an entry's lifetime the TTL rather than the
        # process's.
        self._prune_recorded_cache(now)
        counts = expected_counts or {}
        # A surviving entry is live by construction: the sweep above dropped every
        # expired deadline, so mere presence is the whole test.
        cached_recorded = {
            tid for tid in target_ids if (tid, counts.get(tid)) in self._recorded_until
        }
        to_check = target_ids - cached_recorded
        if cached_recorded:
            # Debug, not info: in steady state this fires every monitoring
            # interval with the same targets, which is the cache working as
            # intended rather than an event worth a line in the default log.
            logger.debug(
                "Skipping wandb dedup query for %d target(s) already known "
                "recorded (cached): %s",
                len(cached_recorded),
                sorted(cached_recorded),
            )
        if not to_check:
            return set()

        # Delegate to the service, which checks the candidates against wandb run
        # metadata. ``expected_counts`` lets it require a *full* set of runs per
        # target rather than mere presence (see ILineageStore.filter_unrecorded).
        # Never raises: it fails CLOSED, returning an empty set and reporting the
        # error through ``on_query_error``.
        #
        # The callback is wrapped rather than merely forwarded, because this layer
        # must know whether the query was answered before it caches anything. A
        # failed query now returns an EMPTY set, which is indistinguishable by
        # value from "every candidate is already recorded" -- and caching that
        # would turn one wandb outage into a TTL-long window in which real targets
        # are skipped as recorded. The flag is the only thing separating the two.
        query_failed = False

        def _note_failure(exc: Exception) -> None:
            nonlocal query_failed
            query_failed = True
            if on_query_error is not None:
                on_query_error(exc)

        unrecorded = self._service.filter_unrecorded(
            to_check, expected_counts, on_query_error=_note_failure
        )

        if query_failed:
            # Cache nothing: there is no verdict to cache. Returning the empty set
            # the service produced keeps this fail-closed -- the caller is expected
            # to abort the pass (it heard about the failure through on_query_error)
            # and retry, rather than read "nothing to record" as success.
            return unrecorded

        # Cache only the positive verdicts, and only for targets we actually asked
        # about. An unrecorded target is not cached: its verdict is expected to
        # change as soon as recording succeeds, and a stale negative would be
        # re-queried anyway.
        deadline = now + _RECORDED_CACHE_TTL_SECONDS
        newly_recorded = to_check - unrecorded
        for tid in newly_recorded:
            self._recorded_until[(tid, counts.get(tid))] = deadline
        if newly_recorded:
            # Info: this is the transition -- wandb was asked and answered "already
            # recorded", and the verdict is now cached for the TTL. Once per target
            # per TTL, not once per scan.
            logger.info(
                "wandb already holds lineage for %d target(s); caching the "
                "verdict for %ds: %s",
                len(newly_recorded),
                _RECORDED_CACHE_TTL_SECONDS,
                sorted(newly_recorded),
            )
        return unrecorded

    def _prune_recorded_cache(self, now: float) -> None:
        """Drop every "already recorded" verdict whose TTL has passed.

        Runs on each filter pass so the cache is bounded by the TTL rather than
        by how many targets the process has recorded over its lifetime.
        """
        expired = [
            key for key, deadline in self._recorded_until.items() if deadline <= now
        ]
        for key in expired:
            del self._recorded_until[key]

    def _build_event_for_artifact(
        self,
        artifact: ArtifactRegistration,
        sources: list[ArtifactRegistration],
    ) -> dict:
        return build_event_for_artifact(artifact, sources)
