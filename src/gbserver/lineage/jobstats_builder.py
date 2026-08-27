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

"""DB-only jobstats event construction and cross-build lineage traversal.

These functions build JobStats events entirely from the admin DB (build
storage, target-run storage, artifact registry) with no dependency on any
particular lineage backend (wandb, noop, etc). Backends call into this module
so the event shape and traversal semantics are identical everywhere.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from gbcommon.types.constants import DEFAULT_GH_DOMAIN, is_public_github
from gbserver.storage.artifact_registration import ArtifactRegistration
from gbserver.storage.singleton_storage import SingletonAdminStorage
from gbserver.storage.stored_build import StoredBuild
from gbserver.storage.stored_target_run import StoredTargetRun
from gbserver.types.constants import (
    GB_JOB_STATS_DETAIL_CATEGORY,
    GB_JOB_STATS_DETAIL_REGISTERED_ARTIFACT_JOB_NAME,
    GB_JOB_STATS_DETAIL_REGISTERED_ARTIFACT_TYPE,
    GB_JOB_STATS_DETAIL_TYPE,
)
from gbserver.types.status import Status
from gbserver.utils.logger import get_logger
from gbserver.utils.redaction import redact_sensitive, scrub_url_credentials
from gbserver.utils.utils import get_uuid

logger = get_logger(__name__)

_LINEAGE_REPO_ORG = "ibm-granite" if is_public_github() else "granite-dot-build"
LINEAGE_PRODUCER_URL = f"https://{DEFAULT_GH_DOMAIN}/{_LINEAGE_REPO_ORG}/granite.build"

_STATUS_TO_EVENT_TYPE: Dict[Status, str] = {
    Status.SUCCESS: "COMPLETE",
    Status.FAILED: "FAIL",
    Status.RUNNING: "RUNNING",
    Status.PENDING: "START",
    Status.SUBMITTED: "START",
    Status.CANCELLED: "ABORT",
    Status.CANCEL_REQUESTED: "RUNNING",
    Status.INVALID: "FAIL",
}

# Hard cap on the number of target-runs a single get_lineage_graph BFS will
# visit, independent of max_depth (including finite values, not just -1/"full
# map"). Protects against a single request returning a huge fraction of the
# instance's runs and stalling the server or the browser.
_MAX_VISITED_RUNS = 500


def _lh_uri_to_namespace_and_name(uri: str) -> Optional[Tuple[str, str]]:
    from urllib.parse import urlparse

    from gbcommon.uri.lh import LhType, LhURI

    parse = urlparse(uri)
    if parse.scheme not in LhURI.get_supported_schemes():
        return None

    lh = LhURI(parse)
    namespace = lh.get_lh_namespace()
    lh_type = lh.get_lh_type()
    if lh_type == LhType.TABLE:
        name = lh.get_lh_table_name()
    elif lh_type == LhType.FILESET:
        name = f"{lh.get_lh_fileset_label()}-{lh.get_lh_fileset_version()}"
    elif lh_type == LhType.MODEL:
        name = f"{lh.get_lh_model_label()}-{lh.get_lh_model_revision()}"
    elif lh_type == LhType.DATASET:
        name = lh.get_lh_dataset_name()
    else:
        return None
    return namespace, name


def _build_target_artifact_reference(
    target_name: str,
    target_artifact_name: str,
    is_input: bool,
    index: int,
) -> str:
    in_or_out = "inputs" if is_input else "outputs"
    reference = f"{target_name}.{in_or_out}.{target_artifact_name}"
    if index >= 0:
        reference = f"{reference}[{index}]"
    return reference


def _artifact_to_lineage_entry(
    artifact: ArtifactRegistration,
    target_artifact_name: str = "",
    target_name: str = "",
    is_input: bool = True,
    index: int = -1,
) -> dict:
    from urllib.parse import urlparse

    from gbcommon.uri.hf import HfURI

    artifact_type = artifact.type
    if artifact.uri:
        from gbcommon.uri.uri import UnknownURIScheme
        from gbcommon.uri.utils import get_artifact_type

        try:
            artifact_type = get_artifact_type(artifact.uri)
        except UnknownURIScheme:
            pass

    namespace = artifact.uri
    name = artifact.name or target_artifact_name or artifact.uuid

    target_artifact_reference = _build_target_artifact_reference(
        target_name, target_artifact_name, is_input, index
    )

    facets: dict[str, Any] = {
        "artifact_id": artifact.uuid,
        "artifact_uri": artifact.uri,
        "artifact_type": artifact_type.name,
        "target_artifact_reference": target_artifact_reference,
        "gb-artifact-id": artifact.uuid,
        "gb-artifact-uri": artifact.uri,
        "gb-build-id": artifact.created_by_build_id,
        "gb-target-id": artifact.created_by_target_id,
        "gb-build-target-artifact": target_artifact_reference,
    }
    facets.update(artifact.model_dump(mode="json"))

    uri = artifact.uri
    parse = urlparse(uri)
    if parse.scheme in HfURI.get_supported_schemes():
        hf = HfURI(parse)
        parts = hf._parts()
        repo_id = f"{parts.owner}/{parts.repo}"
        namespace = parts.owner
        name = repo_id
    else:
        lh_result = _lh_uri_to_namespace_and_name(uri)
        if lh_result is not None:
            namespace, name = lh_result

    return {
        "namespace": namespace,
        "name": name,
        "uri": uri,
        "facets": facets,
    }


def _add_jobstats_mirror_fields(event: dict) -> None:
    # The REST jobstats endpoints expect a flat JobStats-shaped dict (with
    # release_id / job_details / sources / targets at the top level). wandb
    # stores these inside run.facets.job_details + inputs/outputs, so mirror
    # them for readers. wandb itself ignores unknown top-level keys.
    job_details = event.get("run", {}).get("facets", {}).get("job_details", {})
    event["release_id"] = job_details.get("release_id", "")
    event["job_details"] = job_details
    event["sources"] = event.get("inputs", [])
    event["targets"] = event.get("outputs", [])


def build_events_for_target(
    storage: SingletonAdminStorage,
    build: StoredBuild,
    targetrun: StoredTargetRun,
) -> Tuple[List[dict], Dict[str, List[dict]]]:
    event_type = _STATUS_TO_EVENT_TYPE.get(targetrun.status, "OTHER")
    event_time = (
        targetrun.finished_at.isoformat()
        if targetrun.finished_at
        else (
            targetrun.started_at.isoformat()
            if targetrun.started_at
            else build.created_time.isoformat()
        )
    )

    inputs = []
    for target_artifact_name, uuid in targetrun.input_artifacts.items():
        artifact = storage.artifact_registry.get_by_uuid(uuid)
        if artifact and isinstance(artifact, ArtifactRegistration):
            inputs.append(
                _artifact_to_lineage_entry(
                    artifact,
                    target_artifact_name,
                    target_name=targetrun.name,
                    is_input=True,
                    index=-1,
                )
            )

    step_configs = []
    steps = storage.step_storage.get_by_where({"target_id": targetrun.uuid})
    for step in steps:
        # step.config is the rendered build.yaml input and step.metadata is
        # runtime data the step pushed (e.g. commit_hash). jobstats is readable
        # by any space member (not just the build owner/admin), so both are
        # emitted with secret-*named* keys masked via redact_sensitive, which
        # also scrubs userinfo@ credentials out of any URL-shaped value. The
        # definition_uri is scrubbed the same way so a credentialed BYOS clone
        # URL (git+ssh://token@... / https://token@...) cannot leak here.
        step_configs.append(
            {
                "uri": scrub_url_credentials(step.definition_uri),
                "config": redact_sensitive(step.config),
                "metadata": redact_sensitive(step.metadata),
            }
        )

    started_at = (
        targetrun.started_at.isoformat() if targetrun.started_at else event_time
    )
    completed_at = targetrun.finished_at.isoformat() if targetrun.finished_at else ""

    base_event: Dict[str, Any] = {
        "eventType": event_type,
        "eventTime": event_time,
        "run": {
            "runId": targetrun.uuid,
            "facets": {
                "tags": {
                    "build_id": build.uuid,
                    "target_id": targetrun.uuid,
                    "username": build.username,
                    "space_name": build.space_name,
                },
                "source_code": {
                    "url": build.source_uri,
                    "commit_hash": "",
                    "path": "",
                },
                "job_input_params": {"steps": step_configs},
                "execution_stats": {},
                "job_details": {
                    "job_id": targetrun.uuid,
                    "job_type": GB_JOB_STATS_DETAIL_TYPE,
                    "category": GB_JOB_STATS_DETAIL_CATEGORY,
                    "job_status": targetrun.status.name,
                    "job_started_at": started_at,
                    "job_completed_at": completed_at,
                    "release_id": targetrun.build_id,
                    "owner": build.username,
                    "job_output_stats": {},
                },
            },
        },
        "job": {
            "namespace": f"{build.space_name}/{build.name}",
            "name": targetrun.name,
            "facets": {},
        },
        "producer": LINEAGE_PRODUCER_URL,
        "schemaURL": "https://openlineage.io/spec/2-0-2/OpenLineage.json#/$defs/RunEvent",
    }

    if build.description:
        base_event["job"]["facets"]["documentation"] = {
            "description": build.description,
        }

    events_list: List[dict] = []
    events_dict: Dict[str, List[dict]] = {}

    # NOTE: the number of events emitted here (one per output artifact across
    # all output-artifact lists, or one "no-output" event below) must stay in
    # lockstep with lineage_reconciler.expected_run_count, which derives the
    # same count from the target in memory to detect partial records.
    for (
        target_artifact_name,
        output_artifact_list,
    ) in targetrun.output_artifacts.items():
        target_events: List[dict] = []
        include_index = len(output_artifact_list) > 1
        index = -1
        for output_uuid in output_artifact_list:
            if include_index:
                index += 1
            artifact = storage.artifact_registry.get_by_uuid(output_uuid)
            outputs = []
            if artifact and isinstance(artifact, ArtifactRegistration):
                outputs.append(
                    _artifact_to_lineage_entry(
                        artifact,
                        target_artifact_name,
                        target_name=targetrun.name,
                        is_input=False,
                        index=index,
                    )
                )
            event = {
                **base_event,
                "inputs": inputs,
                "outputs": outputs,
            }
            # Give each output-artifact event its own wandb run so
            # history rows are not collapsed when multiple events share
            # a single resumed run. Keeps counts aligned with the number
            # of output artifacts. The job_id in job_details still points
            # back to the logical target (targetrun.uuid).
            #
            # The id is a fresh random uuid, not derived from the target and
            # output uuids. Dedup is therefore carried entirely by the
            # target_id tag in run.facets.tags (see LineageService.
            # filter_unrecorded), which is why that tag must be present on
            # EVERY emitted event: a run without it is invisible to the
            # dedup query, cannot be counted toward expected_run_count, and
            # is unreclaimable -- no later scan can find it or replace it.
            #
            # Random is REQUIRED here, not incidental. Deriving the id from
            # the target/output (the scheme this replaced) means a run
            # DELETED in wandb can never be re-created: wandb refuses a
            # deleted run's id, and a derived id recomputes to that same
            # tombstoned value on every later scan, so the target becomes
            # permanently unrecordable. That happened with intentional
            # deletions; see commit 5824ae99 and the extended note in
            # WandBLineageService.filter_unrecorded, which also explains the
            # partial-record trade-off this buys and why it is accepted.
            #
            # Tag the run with the output artifact it represents. base_event
            # cannot carry this: its tags are shared by every event of the
            # target, while output_uuid identifies just this one. Run ids are
            # random and carry no output information, so without this tag the
            # only way to find the run for a given output is to fetch the
            # target's runs and inspect their outputs facet; with it the
            # lookup is a tag filter like the target_id ones above.
            #
            # Additive only: it does not affect dedup. filter_unrecorded
            # matches on "target_id=" tags and skips every other key, and
            # tags serialize generically as "k=v"
            # (WandBLineageService._process_event), so nothing else changes.
            event["run"] = {
                **base_event["run"],
                "runId": get_uuid(),
                "facets": {
                    **base_event["run"]["facets"],
                    "tags": {
                        **base_event["run"]["facets"]["tags"],
                        "output_id": output_uuid,
                    },
                },
            }
            _add_jobstats_mirror_fields(event)
            target_events.append(event)
        events_list.extend(target_events)
        events_dict[target_artifact_name] = target_events

    # A target that produced no outputs still gets exactly one event, even when
    # it has no inputs either (e.g. a pure generation/compute target). Guard
    # only on the absence of output-artifact events, not on having inputs;
    # otherwise an artifact-less target emits nothing and the reconciler
    # silently marks it "recorded" without ever contacting the backend.
    if len(targetrun.output_artifacts) == 0:
        event = {
            **base_event,
            "inputs": inputs,
            "outputs": [],
            # Explicit random runId: inheriting base_event's would reuse
            # targetrun.uuid, the deterministic id this design replaced, and
            # a re-record would silently resume that one run instead of
            # writing a new one. The target_id tag comes along in
            # base_event["run"]["facets"]["tags"], keeping this event
            # dedupable like the per-output ones.
            "run": {**base_event["run"], "runId": get_uuid()},
        }
        _add_jobstats_mirror_fields(event)
        events_list.append(event)
        events_dict["no-output"] = [event]

    return events_list, events_dict


def build_event_for_artifact(
    artifact: ArtifactRegistration,
    sources: list[ArtifactRegistration],
) -> dict:
    use_index = len(sources) > 0
    inputs = []
    index = -1
    for src in sources:
        if use_index:
            index += 1
        inputs.append(
            _artifact_to_lineage_entry(
                src,
                target_artifact_name=src.name,
                target_name=src.name,
                is_input=True,
                index=index,
            )
        )
    outputs = [
        _artifact_to_lineage_entry(
            artifact,
            target_artifact_name=artifact.name,
            target_name="pseudo-target",
            is_input=False,
            index=-1,
        )
    ]

    event_time = artifact.created_at.isoformat()

    job_input_params: Dict[str, Any] = {}
    if artifact.origin_uris:
        job_input_params["origin_uris"] = artifact.origin_uris
    if artifact.description:
        job_input_params["description"] = artifact.description

    event = {
        "eventType": "COMPLETE",
        "eventTime": event_time,
        "run": {
            "runId": artifact.uuid,
            "facets": {
                "tags": {
                    "artifact_id": artifact.uuid,
                    # For registered-artifact jobstats the "release_id" is
                    # the artifact uuid itself — tag build_id with that so
                    # count_release_ids({artifact.uuid}) finds this run.
                    "build_id": artifact.uuid,
                    "target_id": artifact.created_by_target_id,
                    "username": artifact.username,
                    "space_name": artifact.space_name,
                },
                "source_code": {"url": "", "commit_hash": "", "path": ""},
                "job_input_params": job_input_params,
                "execution_stats": {},
                "job_details": {
                    "job_id": artifact.uuid,
                    "job_type": GB_JOB_STATS_DETAIL_REGISTERED_ARTIFACT_TYPE,
                    "category": GB_JOB_STATS_DETAIL_CATEGORY,
                    "job_status": artifact.status.name,
                    "job_started_at": event_time,
                    "job_completed_at": event_time,
                    "release_id": artifact.uuid,
                    "owner": artifact.username,
                    "job_output_stats": {},
                },
            },
        },
        "job": {
            "namespace": artifact.space_name,
            "name": GB_JOB_STATS_DETAIL_REGISTERED_ARTIFACT_JOB_NAME,
            "facets": {},
        },
        "inputs": inputs,
        "outputs": outputs,
        "producer": LINEAGE_PRODUCER_URL,
        "schemaURL": "https://openlineage.io/spec/2-0-2/OpenLineage.json#/$defs/RunEvent",
    }
    _add_jobstats_mirror_fields(event)
    return event


def traverse_lineage_graph(
    storage: SingletonAdminStorage,
    build_id: str,
    direction: str = "both",
    max_depth: int = 10,
) -> dict:
    """BFS across target-runs, joined via shared artifact UUIDs, starting
    from all target-runs of build_id.

    Returns {"root_build_id", "targets": [<jobstats dict per visited run>],
    "truncated": bool, "expandable": [<edge-of-graph neighbors>]}.
    """
    seed_runs = storage.target_storage.get_by_where({"build_id": build_id})
    seed_runs = [r for r in seed_runs if isinstance(r, StoredTargetRun)]

    if not seed_runs:
        return {
            "root_build_id": build_id,
            "targets": [],
            "truncated": False,
            "expandable": [],
        }

    all_runs = storage.target_storage.get_by_uuid(None)
    runs_by_id: Dict[str, StoredTargetRun] = {
        r.uuid: r for r in all_runs if isinstance(r, StoredTargetRun)
    }

    consumers: Dict[str, List[StoredTargetRun]] = {}
    for run in runs_by_id.values():
        for artifact_uuid in run.input_artifacts.values():
            consumers.setdefault(artifact_uuid, []).append(run)

    artifact_cache: Dict[str, Optional[ArtifactRegistration]] = {}

    def get_artifact(uuid: str) -> Optional[ArtifactRegistration]:
        if uuid not in artifact_cache:
            artifact = storage.artifact_registry.get_by_uuid(uuid)
            artifact_cache[uuid] = (
                artifact if isinstance(artifact, ArtifactRegistration) else None
            )
        return artifact_cache[uuid]

    def downstream_neighbors(run: StoredTargetRun) -> List[StoredTargetRun]:
        neighbors = []
        for output_uuids in run.output_artifacts.values():
            for output_uuid in output_uuids:
                neighbors.extend(consumers.get(output_uuid, []))
        return neighbors

    def upstream_neighbors(run: StoredTargetRun) -> List[StoredTargetRun]:
        neighbors = []
        for input_uuid in run.input_artifacts.values():
            artifact = get_artifact(input_uuid)
            if artifact is None or not artifact.created_by_target_id:
                continue
            producer = runs_by_id.get(artifact.created_by_target_id)
            if producer is not None:
                neighbors.append(producer)
        return neighbors

    directions_to_walk: List[str] = (
        ["upstream", "downstream"] if direction == "both" else [direction]
    )

    visited: Dict[str, StoredTargetRun] = {}
    expandable_seen: set = set()
    expandable: List[dict] = []
    truncated = False
    unbounded = max_depth == -1

    for walk_direction in directions_to_walk:
        neighbor_fn = (
            downstream_neighbors
            if walk_direction == "downstream"
            else upstream_neighbors
        )
        queue: List[Tuple[StoredTargetRun, int]] = [(run, 0) for run in seed_runs]
        for run in seed_runs:
            visited.setdefault(run.uuid, run)

        while queue:
            current, depth = queue.pop(0)

            if len(visited) >= _MAX_VISITED_RUNS:
                truncated = True
                for neighbor in neighbor_fn(current):
                    if neighbor.uuid not in visited:
                        key = (neighbor.uuid, walk_direction)
                        if key not in expandable_seen:
                            expandable_seen.add(key)
                            expandable.append(
                                {
                                    "build_id": neighbor.build_id,
                                    "target_id": neighbor.uuid,
                                    "direction": walk_direction,
                                }
                            )
                continue

            for neighbor in neighbor_fn(current):
                if neighbor.uuid in visited:
                    continue
                if not unbounded and depth + 1 > max_depth:
                    truncated = True
                    key = (neighbor.uuid, walk_direction)
                    if key not in expandable_seen:
                        expandable_seen.add(key)
                        expandable.append(
                            {
                                "build_id": neighbor.build_id,
                                "target_id": neighbor.uuid,
                                "direction": walk_direction,
                            }
                        )
                    continue
                visited[neighbor.uuid] = neighbor
                if len(visited) >= _MAX_VISITED_RUNS:
                    truncated = True
                queue.append((neighbor, depth + 1))

    build_cache: Dict[str, Optional[StoredBuild]] = {}

    def get_build(bid: str) -> Optional[StoredBuild]:
        if bid not in build_cache:
            build = storage.build_storage.get_by_uuid(bid)
            build_cache[bid] = build if isinstance(build, StoredBuild) else None
        return build_cache[bid]

    targets: List[dict] = []
    for run in visited.values():
        build = get_build(run.build_id)
        if build is None:
            continue
        _, jobstats_dict = create_jobstats_for_visited_target(storage, build, run)
        targets.append(jobstats_dict)

    return {
        "root_build_id": build_id,
        "targets": targets,
        "truncated": truncated,
        "expandable": expandable,
    }


def create_jobstats_for_visited_target(
    storage: SingletonAdminStorage,
    build: StoredBuild,
    targetrun: StoredTargetRun,
) -> Tuple[List[dict], Dict[str, List[dict]]]:
    """Build jobstats for a target-run visited during traversal.

    Thin wrapper around build_events_for_target, matching
    create_jobstats_for_target's behavior on both backends.
    """
    return build_events_for_target(storage, build, targetrun)
