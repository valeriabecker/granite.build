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

"""The URI the UI identifies a lineage node by.

Why a node needs one at all: the frontend deduplicates lineage refs by
``ref.uri ?? "ref-" + ref.name``, so ``uri`` *is* a node's identity there. Two
distinct nodes reporting the same URI -- or both reporting none -- collapse into a
single node in the panel, a silent wrong answer rather than a visible gap.

The URI is **stored, not inferred**. A canonical lineage identifier encodes only
type, namespace, name, table and revision; it does not encode the scheme, so
``hf://`` and ``s3://`` cannot be recovered from one. Rows therefore carry
``source_uri``/``target_uri`` verbatim from the artifact that produced them (see
``decompose._artifact_uri``), and this module's first job is to prefer that value.

:func:`synthesized_uri_for_identifier` is the fallback for rows that have no
stored URI -- lineage imported from a source that recorded none, and rows written
before the column existed. It rebuilds what the pieces do determine:

- **Lakehouse (``lh://``)** is exact. Its grammar is fully determined by the
  stored pieces (see ``LhURI``), and a canonical identifier's namespace is
  ``host/namespace`` for these -- precisely the URI's authority plus first path
  segment.
- Everything else gets a ``gb://`` URI built from the same pieces. It is not a
  resolvable location and does not claim to be one; it exists so N distinct nodes
  stay N distinct nodes. Guessing ``hf://`` here would assert an origin the index
  never recorded, and a wrong URI that looks authoritative is worse than one that
  is visibly synthetic.

The invariant every branch holds: **two different canonical identifiers never
produce the same URI.** A collision re-introduces the node merging this module
exists to prevent, so the fallback keeps every piece the identifier carries.
"""

from typing import Optional
from urllib.parse import quote

from gbserver.lineage.identity import (
    ArtifactIdentity,
    LineageIdentityError,
    parse_canonical_id,
)
from gbserver.types.artifact import ArtifactType

# Path segment per Lakehouse type, mirroring LhURI._get_urisegment_for_type. Not
# imported from there: that helper is private and keyed by LhType, a different
# enum from ArtifactType. The segments are a stable part of the lh:// grammar
# documented on LhURI itself.
_LH_TYPE_SEGMENT = {
    ArtifactType.TABLE: "tables",
    ArtifactType.MODEL: "models",
    ArtifactType.FILESET: "filesets",
    ArtifactType.DATASET: "datasets",
}

_LH_SCHEME = "lh"
_FALLBACK_SCHEME = "gb"

_KIND_KEY = "kind"
_NS_KEY = "namespace"
_NAME_KEY = "name"
_TABLE_KEY = "table"
_REV_KEY = "revision"


def node_uri(identifier: str, stored_uri: str = "") -> Optional[str]:
    """Return the URI to identify one lineage node by.

    Args:
        identifier: the node's canonical identifier, from a row's ``source`` or
            ``target``.
        stored_uri: the URI recorded alongside it (``source_uri``/``target_uri``).
            Preferred whenever present, because it is the artifact's real URI
            rather than a reconstruction.

    Returns:
        The URI, or ``None`` when ``identifier`` is empty (a terminal marker) or
        malformed. ``None`` means "this is not a node" -- it must never reach the
        UI as a node's ``uri``, since that is what merges nodes.
    """
    if not identifier:
        return None
    if stored_uri:
        return stored_uri
    return synthesized_uri_for_identifier(identifier)


def synthesized_uri_for_identifier(identifier: str) -> Optional[str]:
    """Rebuild a URI from a canonical identifier alone.

    For rows with no stored URI. Prefer :func:`node_uri`, which falls back to this
    only when nothing was recorded.

    Args:
        identifier: a canonical identifier.

    Returns:
        The reconstructed URI, or ``None`` when ``identifier`` is empty or is not a
        well-formed canonical identifier.
    """
    if not identifier:
        return None
    try:
        identity = parse_canonical_id(identifier)
    except LineageIdentityError:
        # A malformed identifier is not a node. Returning None rather than
        # synthesizing something keeps a corrupt row from presenting as real.
        return None
    return synthesized_uri_for_identity(identity)


def synthesized_uri_for_identity(identity: ArtifactIdentity) -> str:
    """Rebuild a URI from an artifact's decomposed pieces.

    Args:
        identity: the pieces, as recovered by
            :func:`~gbserver.lineage.identity.parse_canonical_id`.

    Returns:
        The reconstructed URI. Always non-empty: every :class:`ArtifactIdentity`
        carries a namespace plus a name or a table, which the fallback form needs.
    """
    if _is_lakehouse_shaped(identity):
        return _lakehouse_uri(identity)
    return _fallback_uri(identity)


def _is_lakehouse_shaped(identity: ArtifactIdentity) -> bool:
    """Whether these pieces can be rendered as an ``lh://`` URI.

    Two conditions. The type must have an ``lh://`` shape -- a bucket, an
    undefined type or a deprecated one has none -- and the namespace must be
    ``host/namespace``, since the URI needs both an authority and a namespace
    segment.

    Any non-empty host qualifies, not just ``prod``/``staging``: ``LhURI``
    explicitly documents custom hostnames, so restricting the set would push real
    Lakehouse artifacts into the fallback form.
    """
    if identity.artifact_type not in _LH_TYPE_SEGMENT:
        return False
    host, _, namespace = identity.namespace.partition("/")
    if not host or not namespace or "/" in namespace:
        return False
    return True


def _lakehouse_uri(identity: ArtifactIdentity) -> str:
    """Build the ``lh://`` URI, matching the grammar on :class:`LhURI`::

        table:   lh://<host>/<ns>/tables/<table>
        model:   lh://<host>/<ns>/models/<table>/<label>/<revision>
        fileset: lh://<host>/<ns>/filesets/<table>/<label>/<version>
        dataset: lh://<host>/<ns>/datasets/<table>/<dataset>

    A model or fileset with no revision is emitted **without** one rather than
    with ``LhURI``'s injected ``granite-dot-build`` default. ``canonical_id``
    treats that default as "no revision" (``_effective_revision``), so emitting it
    would assert a revision the index never recorded.
    """
    host, _, namespace = identity.namespace.partition("/")
    parts = [_q(host), _q(namespace), _LH_TYPE_SEGMENT[identity.artifact_type]]

    if identity.artifact_type == ArtifactType.TABLE:
        parts.append(_q(identity.table))
    elif identity.artifact_type == ArtifactType.DATASET:
        parts.extend([_q(identity.table), _q(identity.name)])
    else:
        # MODEL and FILESET share the <table>/<label>[/<revision>] tail.
        parts.extend([_q(identity.table), _q(identity.name)])
        if identity.revision:
            parts.append(_q(identity.revision))

    return f"{_LH_SCHEME}://" + "/".join(parts)


def _fallback_uri(identity: ArtifactIdentity) -> str:
    """Build a ``gb://`` URI for a node whose real URI is unknown.

    Buckets, undefined-type artifacts, the deprecated types, and anything whose
    namespace is not Lakehouse-shaped. Deliberately not resolvable: it is a stable
    identity for deduplication and nothing more.

    Every piece the identity carries goes in, which is what guarantees no
    collisions -- two identifiers differing in any piece yield different URIs. The
    pieces go in the query string rather than the path so an absent one is
    unambiguous: an omitted ``table`` cannot be misread as a name.
    """
    kind = identity.artifact_type.value or "undefined"
    parts = [f"{_KIND_KEY}={_q(kind)}", f"{_NS_KEY}={_q(identity.namespace)}"]
    if identity.name:
        parts.append(f"{_NAME_KEY}={_q(identity.name)}")
    if identity.table:
        parts.append(f"{_TABLE_KEY}={_q(identity.table)}")
    if identity.revision:
        parts.append(f"{_REV_KEY}={_q(identity.revision)}")
    return f"{_FALLBACK_SCHEME}://artifact?" + "&".join(parts)


def _q(piece: str) -> str:
    """Percent-encode one URI piece.

    ``/`` is escaped along with everything else: a name containing a slash would
    otherwise introduce a path segment and let two different artifacts render as
    one URI, the single failure this module must not have. Canonical identifiers
    reject the identity separators but not ``/``, so this is reachable.
    """
    return quote(piece, safe="")
