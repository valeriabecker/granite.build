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

"""Derive a canonical lineage identity from a registered artifact.

This is the bridge ``to_lineage_rows`` needs: it takes the artifact dicts the
jobstats builders already produce and returns the canonical identifier string the
lineage index keys on.

Two hazards shape the implementation, both verified in ``gbcommon/uri/lh.py``:

**Dispatch on the type first, never on an assert.** Five of ``LhURI``'s piece
getters use ``assert`` as flow control -- ``get_lh_model_label`` opens with
``assert self.get_lh_type() == LhType.MODEL`` -- and ``python -O`` strips those.
Under ``-O`` a fileset URI passed to ``get_lh_model_label`` therefore returns
whatever sits at the model-label index rather than raising, which would mint a
plausible-looking identifier for the wrong artifact. So the type is resolved with
``get_lh_type()`` up front and only the matching getters are called; the asserts
are never relied on.

**Derive from the raw URI, never the normalized one.** ``LhURI.__init__`` appends
a default revision (``granite-dot-build``) when a model or fileset URI carries
none, and its guard is a substring test (``if not revision in uristr``) -- so
whether a given URI gets the injection is not even deterministic. Constructing an
``LhURI`` is unavoidable for the getters, but the revision it reports is passed
through ``canonical_id``'s ``_effective_revision``, which treats the injected
default as "no revision". That is what makes the same fileset converge whether or
not it went through ``LhURI``.
"""

import logging
from typing import Optional
from urllib.parse import urlparse

from gbserver.lineage.identity import (
    ArtifactIdentity,
    LineageIdentityError,
    canonical_id,
)
from gbserver.types.artifact import ArtifactType

logger = logging.getLogger(__name__)

# Namespace for an artifact whose URI carries none of its own. A canonical
# identifier REQUIRES a namespace -- it is what stops two same-named artifacts in
# different organizations from collapsing into one node -- so a fallback is needed
# rather than an empty string. The space name is the right scope: it is the
# ownership boundary granite.build already enforces.
_UNKNOWN_NAMESPACE = "unknown"


def identity_from_artifact_dict(artifact: dict) -> Optional[str]:
    """Return the canonical identifier for one artifact dict, or ``None``.

    The ``identify`` callable :func:`~gbserver.lineage.decompose.to_lineage_rows`
    expects.

    Args:
        artifact: an artifact dict as the jobstats builders produce it. Read keys:
            ``uri`` (preferred), ``name``, ``space_name``, and ``facets`` for the
            artifact type and space fallbacks.

    Returns:
        The canonical identifier, or ``None`` when one cannot be built. ``None``
        drops the endpoint rather than inventing an identity, which the caller
        records -- an artifact merged into the wrong node is invisible, whereas a
        dropped one is countable.
    """
    if not artifact:
        return None

    facets = artifact.get("facets") or {}
    uri = artifact.get("uri") or facets.get("artifact_uri") or ""
    space_name = artifact.get("space_name") or facets.get("space_name") or ""
    name = artifact.get("name") or ""

    identity = identity_from_uri(uri, fallback_name=name, space_name=space_name)
    if identity is None:
        return None
    try:
        return canonical_id(identity)
    except LineageIdentityError:
        logger.warning(
            "Artifact has no canonical lineage identity (uri=%r, name=%r); "
            "dropping the endpoint rather than merging it into a wrong node",
            uri,
            name,
        )
        return None


def identity_from_uri(
    uri: str,
    fallback_name: str = "",
    space_name: str = "",
) -> Optional[ArtifactIdentity]:
    """Decompose an artifact URI into the pieces a canonical identifier needs.

    Args:
        uri: the artifact's URI. A Lakehouse URI yields exact pieces; anything
            else falls to the generic rule.
        fallback_name: the artifact's registered name, used when the URI carries
            no name of its own.
        space_name: the owning space, used as the namespace when the URI has none.

    Returns:
        The pieces, or ``None`` when there is not enough to identify anything --
        neither a usable URI nor a name.
    """
    lh_identity = _lakehouse_identity(uri)
    if lh_identity is not None:
        return lh_identity

    # Generic rule: <type>:<namespace>::<name>. This is what keeps every artifact
    # identifiable, including s3:// and hf:// ones and the deprecated types. An
    # unidentifiable artifact would land as an empty source/target, which the
    # traversal reads as a create/delete terminal -- so it would not merely be
    # mislabelled, it would vanish from the graph as a false end-of-path.
    name = fallback_name or _name_from_uri(uri)
    if not name:
        return None

    return ArtifactIdentity(
        artifact_type=_generic_artifact_type(uri),
        namespace=space_name or _UNKNOWN_NAMESPACE,
        name=name,
    )


def _lakehouse_identity(uri: str) -> Optional[ArtifactIdentity]:
    """Decompose an ``lh://`` URI, or return ``None`` for any other scheme.

    Every piece is read through a getter chosen by ``get_lh_type()``, never by
    relying on a getter's own ``assert`` (stripped under ``python -O``).
    """
    if not uri:
        return None

    from gbcommon.uri.lh import LhType, LhURI

    parsed = urlparse(uri)
    if parsed.scheme not in LhURI.get_supported_schemes():
        return None

    try:
        lh = LhURI(parsed)
        lh_type = lh.get_lh_type()
    except Exception:
        # A malformed lh:// URI. Falling through to the generic rule keeps the
        # artifact in the graph under its registered name instead of dropping it.
        logger.debug("Not a well-formed lh:// URI: %r", uri)
        return None

    if lh_type is None:
        return None

    namespace = f"{lh.get_lh_environment()}/{lh.get_lh_namespace()}"

    try:
        if lh_type == LhType.TABLE:
            return ArtifactIdentity(
                artifact_type=ArtifactType.TABLE,
                namespace=namespace,
                table=lh.get_lh_table_name(),
            )
        if lh_type == LhType.MODEL:
            return ArtifactIdentity(
                artifact_type=ArtifactType.MODEL,
                namespace=namespace,
                name=lh.get_lh_model_label(),
                table=lh.get_lh_table_name(),
                # Passed through as reported; canonical_id drops the injected
                # default, so a model with no real revision converges either way.
                revision=lh.get_lh_model_revision(),
            )
        if lh_type == LhType.DATASET:
            return ArtifactIdentity(
                artifact_type=ArtifactType.DATASET,
                namespace=namespace,
                name=lh.get_lh_dataset_name(),
                table=lh.get_lh_table_name(),
            )
        if lh_type == LhType.FILESET:
            return ArtifactIdentity(
                artifact_type=ArtifactType.FILESET,
                namespace=namespace,
                name=lh.get_lh_fileset_label(),
                table=lh.get_lh_table_name(),
                revision=lh.get_lh_fileset_version(),
            )
    except Exception:
        # A getter failed on a URI its own type dispatch claimed to support (a
        # short path, say). Falls to the generic rule rather than guessing pieces.
        logger.debug("lh:// URI missing a piece its type requires: %r", uri)
        return None

    return None


# Types whose canonical shape REQUIRES a Lakehouse table. canonical_id raises for
# any of these without one, and only an lh:// URI has one -- so a non-Lakehouse
# artifact must not be labelled with them, however accurate the label looks.
#
# This is not hypothetical: get_artifact_type("hf:///org/repo") returns MODEL, and
# a HuggingFace repo has no table. Passing that type through made canonical_id
# raise for EVERY HuggingFace artifact, dropping all of them from the graph. The
# type is therefore demoted to UNDEFINED, which the generic rule can identify.
#
# The demotion loses the "it is a model" label, not the identity: the artifact
# stays one distinct node with its real URI stored alongside (source_uri /
# target_uri), and the UI reads its type from the promoted kind column.
_TABLE_BOUND_TYPES = frozenset(
    {
        ArtifactType.TABLE,
        ArtifactType.MODEL,
        ArtifactType.DATASET,
        ArtifactType.FILESET,
    }
)


def _generic_artifact_type(uri: str) -> ArtifactType:
    """Artifact type usable with the generic ``<type>:<ns>::<name>`` rule.

    ``UNDEFINED`` is a legitimate outcome and stays identifiable: the generic rule
    spells it out as a type token rather than emitting a leading separator.
    """
    if not uri:
        return ArtifactType.UNDEFINED
    try:
        from gbcommon.uri.utils import get_artifact_type

        artifact_type = get_artifact_type(uri)
    except Exception:
        return ArtifactType.UNDEFINED

    if artifact_type in _TABLE_BOUND_TYPES:
        return ArtifactType.UNDEFINED
    return artifact_type


def _name_from_uri(uri: str) -> str:
    """Last-resort name for an artifact with no registered one.

    The full URI path, with separators the identity scheme reserves swapped out.
    Reserved characters are *rejected* by ``canonical_id`` rather than escaped, so
    a name carrying one would drop the artifact from the graph entirely; replacing
    them keeps it identifiable. The substitution is lossy in principle, but only
    for a name that already had no better source than its own URI.
    """
    if not uri:
        return ""
    parsed = urlparse(uri)
    raw = (parsed.netloc + parsed.path).strip("/")
    if not raw:
        raw = uri
    for reserved in (":", "@", "|"):
        raw = raw.replace(reserved, "_")
    return raw
