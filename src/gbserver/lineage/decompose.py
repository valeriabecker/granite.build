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

"""Decompose a job into flat lineage rows.

One job execution becomes a set of ``(source, job_id, target)`` triples. The
``job_id`` on every row is what keeps the decomposition lossless: a job with N
inputs and M outputs emits N*M rows, and grouping those rows back by ``job_id``
recovers the original input and output sets. Without it the flattening would
dissolve the grouping, and "each input produced each output" would be an
unrecoverable claim rather than a projection.

This is deliberately generic -- the input is a plain dict, not a granite.build
type -- because the same function serves the build sink and the future importers
of Lakehouse/dmf-ng/W&B lineage. Nothing here imports storage or granite.build
models.

Ported from ``prototype-lineage-py`` (``lineage/decompose.py``, in turn
``NewStatsInput.getLineageRecords``), with its N*M guard deliberately dropped --
see :func:`to_lineage_rows`.
"""

from typing import Any, Optional

# Job metadata copied verbatim onto every row emitted from one job. The traversal
# never reads these; they travel so a row can be explained without a second
# lookup.
JOB_METADATA_KEYS = (
    "release_id",
    "category",
    "job_name",
    "job_id",
    "job_type",
    "owner",
    "job_started_at",
    "job_completed_at",
    "job_status",
    "job_input_params",
    "execution_stats",
    "job_output_stats",
    "source_code_details",
)


class LineageDecomposeError(ValueError):
    """A job entry cannot be decomposed into lineage rows."""


class LineageRowDraft:
    """One decomposed row, before it becomes a stored item.

    A plain container rather than the storage model, so decomposition stays
    independent of the storage layer and testable without a database.

    ``source`` and ``target`` are ``None`` for the terminal cases -- a creation
    job has no input, a deletion job has no output. That ``None`` is real
    information, not a missing value, and the traversal stops there rather than
    chaining through it.

    Attributes:
        job_id: identity of the job execution; the same value on every row of one
            job, and what makes the N*M decomposition regroupable.
        source: canonical identifier of the input artifact, or ``None`` (creation).
        target: canonical identifier of the output artifact, or ``None`` (deletion).
        source_uri: the input artifact's real URI, verbatim from the artifact dict,
            or ``""`` when it has none. Carried because a canonical identifier does
            not encode the scheme: ``hf://`` and ``s3://`` are unrecoverable from
            it, so the URI has to travel rather than be re-derived downstream.
        target_uri: the output artifact's real URI; see ``source_uri``.
        source_filter: partition filter scoping the input, or ``None``.
        target_filter: partition filter scoping the output, or ``None``.
        source_artifact: the input artifact dict this row came from, if any.
        target_artifact: the output artifact dict this row came from, if any.
        metadata: the job metadata carried onto this row.
    """

    __slots__ = (
        "job_id",
        "source",
        "target",
        "source_uri",
        "target_uri",
        "source_filter",
        "target_filter",
        "source_artifact",
        "target_artifact",
        "metadata",
    )

    def __init__(
        self,
        job_id: str,
        source: Optional[str] = None,
        target: Optional[str] = None,
        source_uri: str = "",
        target_uri: str = "",
        source_filter: Optional[str] = None,
        target_filter: Optional[str] = None,
        source_artifact: Optional[dict] = None,
        target_artifact: Optional[dict] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        self.job_id = job_id
        self.source = source
        self.target = target
        self.source_uri = source_uri
        self.target_uri = target_uri
        self.source_filter = source_filter
        self.target_filter = target_filter
        self.source_artifact = source_artifact
        self.target_artifact = target_artifact
        self.metadata = metadata if metadata is not None else {}

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LineageRowDraft):
            return NotImplemented
        return self.key() == other.key() and self.metadata == other.metadata

    def key(self) -> tuple:
        """Return the identity tuple the storage unique constraint mirrors."""
        return (self.job_id, self.source, self.target)

    def __hash__(self) -> int:
        return hash(self.key())

    def __repr__(self) -> str:
        return (
            f"LineageRowDraft(job_id={self.job_id!r}, source={self.source!r}, "
            f"target={self.target!r})"
        )


def _artifact_uri(artifact: Optional[dict]) -> str:
    """Read an artifact dict's real URI.

    The URI is carried, not re-derived, because a canonical identifier encodes
    only type, namespace, name, table and revision -- never the scheme. ``lh://``
    happens to be reconstructible from those pieces; ``hf://`` and ``s3://`` are
    not, so losing the URI here loses it for good.

    Three keys are tried because the producers disagree on shape and all three are
    real. ``_artifact_to_lineage_entry`` sets a top-level ``uri`` *and* mirrors the
    same value into the facets as ``artifact_uri`` and ``gb-artifact-uri``; an
    importer supplying only a facet bag should not silently lose its URIs.

    Args:
        artifact: the artifact dict, or ``None`` for a terminal endpoint.

    Returns:
        The URI, or ``""`` when there is no artifact or it carries none. Empty is a
        real answer -- the read path falls back to a synthesized identity rather
        than treating the node as absent.
    """
    if not artifact:
        return ""
    direct = artifact.get("uri")
    if direct:
        return str(direct)
    facets = artifact.get("facets") or {}
    for key in ("artifact_uri", "gb-artifact-uri"):
        value = facets.get(key)
        if value:
            return str(value)
    return ""


def _job_metadata(job: dict) -> dict:
    """Collect the metadata keys present on ``job``."""
    return {key: job[key] for key in JOB_METADATA_KEYS if key in job}


def to_lineage_rows(
    job: dict,
    identify: Any,
) -> list[LineageRowDraft]:
    """Decompose a job entry into flat lineage rows.

    Args:
        job: the job entry. Recognized keys: ``job_id`` (required), ``sources``
            and ``targets`` (lists of artifact dicts, either may be empty), plus
            the optional metadata in :data:`JOB_METADATA_KEYS`. Each artifact dict
            is passed to ``identify`` untouched, so its shape is that function's
            concern, except for an optional ``filter`` read here.
        identify: callable mapping one artifact dict to its canonical identifier
            string. Injected rather than imported so decomposition stays testable
            without the identity scheme, and so an importer can supply its own
            translation.

    Returns:
        The rows, in a deterministic order: by source then target as given. A job
        with N sources and M targets yields N*M rows, all sharing ``job_id``.

    Raises:
        LineageDecomposeError: if ``job_id`` is missing or empty, or if the job has
            neither sources nor targets (there is no lineage to record).

    The prototype's guard is **not** ported. It rejects ``len(sources) > 1 and
    len(targets) > 1`` outright, but granite.build produces exactly that
    routinely -- a target run with 3 inputs and 2 outputs -- and the W&B sink
    already handles it without error today. Rejecting it would drop real lineage.
    N*M does not invent provenance here because ``job_id`` is on every row, so the
    original grouping stays recoverable.
    """
    job_id = job.get("job_id") or ""
    if not job_id:
        raise LineageDecomposeError(
            "job entry has no job_id; rows would be unattributable and "
            f"ungroupable: {job!r}"
        )

    sources: list[dict] = list(job.get("sources") or [])
    targets: list[dict] = list(job.get("targets") or [])
    if not sources and not targets:
        raise LineageDecomposeError(
            f"job {job_id!r} has neither sources nor targets; nothing to record"
        )

    metadata = _job_metadata(job)

    def draft(
        source: Optional[dict],
        target: Optional[dict],
    ) -> LineageRowDraft:
        return LineageRowDraft(
            job_id=job_id,
            source=identify(source) if source is not None else None,
            target=identify(target) if target is not None else None,
            source_uri=_artifact_uri(source),
            target_uri=_artifact_uri(target),
            source_filter=source.get("filter") if source else None,
            target_filter=target.get("filter") if target else None,
            source_artifact=source,
            target_artifact=target,
            metadata=dict(metadata),
        )

    # Deletion terminal: inputs consumed, nothing produced. target stays None.
    if not targets:
        return [draft(source, None) for source in sources]

    # Creation terminal: an output with no recorded input. source stays None.
    if not sources:
        return [draft(None, target) for target in targets]

    # The general case, which subsumes the prototype's 2b (N sources, 1 target)
    # and 3 (1 source, M targets) without special-casing either.
    return [draft(source, target) for source in sources for target in targets]


def group_by_job(rows: list[LineageRowDraft]) -> dict[str, dict[str, set]]:
    """Recover each job's input and output sets from decomposed rows.

    The inverse of the N*M flattening, and the reason dropping the prototype's
    guard is safe: the rows of one job still say which inputs and which outputs
    that execution had, so the flat pairing is a projection rather than a loss.

    Args:
        rows: decomposed rows, from any number of jobs.

    Returns:
        ``{job_id: {"sources": {...}, "targets": {...}}}`` with ``None``
        identifiers excluded, so a creation job reports no sources and a deletion
        job no targets.
    """
    grouped: dict[str, dict[str, set]] = {}
    for row in rows:
        entry = grouped.setdefault(row.job_id, {"sources": set(), "targets": set()})
        if row.source is not None:
            entry["sources"].add(row.source)
        if row.target is not None:
            entry["targets"].add(row.target)
    return grouped
