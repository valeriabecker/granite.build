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

"""The shape of a lineage row's ``attributes`` blob.

The row has three columns -- ``job_id``, ``source``, ``target`` -- and everything
else lives in one JSON blob. That blob needs a written contract more than the
columns do, precisely because the database will not enforce it: it is ``Text``, so
nothing in here is queryable, indexable, or validated on write. Without a contract
each producer invents its own keys and every reader has to tolerate all of them.

Four top-level groups, by who owns the value:

``source`` / ``target``
    What each endpoint *is* -- ``kind`` and ``name``. The URI is already the row's
    identity, so it is not repeated here.

``job``
    The execution: name, status, owner, timestamps, namespace. Describes the run
    node the read path rebuilds by grouping rows on ``job_id``.

``origin``
    Where the row came from: which ``system`` produced it, and that system's own
    ``ids``.

``ids`` is deliberately an open sub-map and the only open part of the contract. A
build id and a target run uuid go there for granite.build; another source puts a
pipeline id, a DAG run id, or nothing. They are **not** columns -- see
:mod:`gbserver.storage.stored_lineage_row` -- so nothing can filter on them, which
is the point: they are for explaining a row, not for finding one.

**What is deliberately NOT here.** The job's large payloads --
``job_input_params`` (the full step configs), ``execution_stats``,
``job_output_stats``, ``source_code_details`` -- are not carried. A job with N
inputs and M outputs decomposes into N*M rows, so each would store N*M copies of
the same step config, and the index would be dominated by data it never queries.
That detail is already served, per target, by ``GET /lineage/target/{id}``, which
reads it from the build state where it is stored once.

The visible cost: an ``ArtifactRunEntry`` served from this index has those four
fields empty. They default to ``{}`` in the wire model, so the panel degrades
rather than breaking, and a caller that needs them asks the jobstats endpoint.
"""

from typing import Any, Dict, Optional

# Top-level groups. Named as constants because both the writer and the readers key
# on them, and a typo in either place is a silently empty node rather than an error.
SOURCE = "source"
TARGET = "target"
JOB = "job"
ORIGIN = "origin"

# Endpoint detail keys, inside SOURCE / TARGET.
KIND = "kind"
NAME = "name"

# Job keys. These mirror what the API handler needs to build an ArtifactRunEntry;
# the names drop the redundant ``job_`` prefix they had when the blob was flat.
JOB_NAME = "name"
JOB_TYPE = "type"
JOB_STATUS = "status"
JOB_OWNER = "owner"
JOB_NAMESPACE = "namespace"
JOB_STARTED_AT = "started_at"
JOB_COMPLETED_AT = "completed_at"
JOB_CATEGORY = "category"

# Origin keys.
ORIGIN_SYSTEM = "system"
ORIGIN_IDS = "ids"

# How a flat job dict (the shape every producer emits today, via
# ``decompose.JOB_METADATA_KEYS``) maps onto the ``job`` group. Kept as data rather
# than as a chain of ``.get()`` calls so the translation is inspectable and the
# dropped keys are visibly dropped.
_JOB_KEY_FROM_FLAT = {
    "job_name": JOB_NAME,
    "job_type": JOB_TYPE,
    "job_status": JOB_STATUS,
    "owner": JOB_OWNER,
    "job_namespace": JOB_NAMESPACE,
    "job_started_at": JOB_STARTED_AT,
    "job_completed_at": JOB_COMPLETED_AT,
    "category": JOB_CATEGORY,
}

# Flat keys that are deliberately NOT carried, and why. Listed so a reader asking
# "where did this go?" finds an answer here instead of assuming an oversight.
_DROPPED_FLAT_KEYS = {
    # Large, and identical across every row of one job -- see the module docstring.
    "job_input_params",
    "execution_stats",
    "job_output_stats",
    "source_code_details",
    # Duplicates: release_id IS the build id (wandb_jobstats sets it from
    # targetrun.build_id), and job_id is already a column. Two fields holding one
    # value is the pattern that diverges silently.
    "release_id",
    "job_id",
}


def build_attributes(
    job_metadata: Optional[Dict[str, Any]] = None,
    source_artifact: Optional[Dict[str, Any]] = None,
    target_artifact: Optional[Dict[str, Any]] = None,
    source_system: str = "",
    ids: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Assemble a row's ``attributes`` blob.

    Args:
        job_metadata: the flat job dict a producer emitted; translated onto the
            ``job`` group, dropping the keys in :data:`_DROPPED_FLAT_KEYS`.
        source_artifact: the input artifact dict, read only for kind and name.
        target_artifact: the output artifact dict, likewise.
        source_system: which system produced this row.
        ids: the originating system's own identifiers.

    Returns:
        The blob. Empty groups are omitted rather than written as ``{}``, so a
        reader can tell "not recorded" from "recorded as empty" -- and a row from a
        source that knows nothing about jobs does not carry an empty ``job`` map.
    """
    attributes: Dict[str, Any] = {}

    source_detail = _endpoint_detail(source_artifact)
    if source_detail:
        attributes[SOURCE] = source_detail
    target_detail = _endpoint_detail(target_artifact)
    if target_detail:
        attributes[TARGET] = target_detail

    job = _job_detail(job_metadata or {})
    if job:
        attributes[JOB] = job

    origin: Dict[str, Any] = {ORIGIN_SYSTEM: source_system}
    carried_ids = {key: value for key, value in (ids or {}).items() if value}
    if carried_ids:
        origin[ORIGIN_IDS] = carried_ids
    attributes[ORIGIN] = origin

    return attributes


def _endpoint_detail(artifact: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """What an endpoint *is*, from the artifact dict the producer supplied.

    Only kind and name. The URI is the row's identity and is not repeated, and
    anything else an artifact dict holds belongs to the system that produced it
    rather than to the graph.
    """
    if not artifact:
        return {}
    detail: Dict[str, str] = {}
    kind = artifact.get("artifact_type") or artifact.get("type") or ""
    if kind:
        detail[KIND] = str(kind)
    name = artifact.get("name") or ""
    if name:
        detail[NAME] = str(name)
    return detail


def _job_detail(job_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Translate a flat job dict onto the ``job`` group."""
    job: Dict[str, Any] = {}
    for flat_key, key in _JOB_KEY_FROM_FLAT.items():
        value = job_metadata.get(flat_key)
        if value:
            job[key] = value
    return job


# -- Readers. Every one tolerates a missing group, because a row may come from a
# source that had nothing to say in it, and because rows written by an older
# importer outlive any given version of this contract.


def endpoint_kind(attributes: Optional[Dict[str, Any]], side: str) -> str:
    """The artifact type recorded for one endpoint, or ``""``.

    Args:
        attributes: the row's blob.
        side: :data:`SOURCE` or :data:`TARGET`.
    """
    return str(((attributes or {}).get(side) or {}).get(KIND, "") or "")


def endpoint_name(attributes: Optional[Dict[str, Any]], side: str) -> str:
    """The display name recorded for one endpoint, or ``""``."""
    return str(((attributes or {}).get(side) or {}).get(NAME, "") or "")


def job_detail(attributes: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The ``job`` group, or an empty map."""
    return (attributes or {}).get(JOB) or {}


def origin_system(attributes: Optional[Dict[str, Any]]) -> str:
    """Which system produced the row, or ``""``."""
    return str(((attributes or {}).get(ORIGIN) or {}).get(ORIGIN_SYSTEM, "") or "")


def origin_id(attributes: Optional[Dict[str, Any]], key: str) -> str:
    """One of the originating system's identifiers, or ``""``.

    Not queryable -- this reads the blob. A caller filtering many rows on an id is
    doing a scan, and should ask whether the question belongs to this index at all.
    """
    origin = (attributes or {}).get(ORIGIN) or {}
    return str((origin.get(ORIGIN_IDS) or {}).get(key, "") or "")
