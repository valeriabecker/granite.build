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

"""Canonical artifact identifiers for the lineage index.

A lineage row identifies its endpoints by a *string*, not by a uuid: artifacts
coming from Lakehouse or dmf-ng have no uuid of their own, so an identity scheme
keyed on ``ArtifactRegistration.uuid`` would make their lineage unreachable.
The identifier built here is derived from what every source does have -- the
type, the namespace/organization, and the naming pieces -- so the same artifact
described by two sources converges on one graph node with no alias table.

The scheme is the type dispatch of the ``lakehouse-data`` Java service (ported in
``prototype-lineage-py``, ``lineage/identifiers.py``) with four corrections; each
one closes a way that scheme silently merged or dropped artifacts. See
``docs/`` -- and the tests -- for the cases that pin them down.

Form::

    <type>:<namespace>::<rest, by type>

    table:host/ns::tablename
    model:host/ns::label|tablename
    dataset:host/ns::datasetname|tablename
    fileset:host/ns::label@version|tablename
    <anything else>:namespace::name

The type dispatch is what lets a source with no ``table`` or ``revision`` be
identified at all: each type asks only for the pieces it has, so there is no
"absent piece" to encode. An ``s3://`` artifact is a generic type and its
identifier is just ``<type>:<namespace>::<name>``.

Separators are **not escaped**. A piece containing one is rejected instead, so an
identifier is never ambiguous: escaping can be added later without invalidating
any identifier already stored, since no accepted piece contains a separator to
escape. Rejecting rather than escaping keeps the prototype's ambiguity from
coming back -- there, ``name.version`` cannot be inverted, and two different
artifacts can produce one identifier -- while surfacing any real name that needs
escaping as a loud error rather than a silent merge.
"""

from gbserver.types.artifact import ArtifactType

# The identifier is stored in the lineage table's ``source``/``target`` columns.
# Every promoted string column in the SQL storage layer is String(256) and
# TRUNCATES SILENTLY (sql_storage.py, __create_sqlalchemy_class_from_dict), and
# these are the columns the traversal joins on: a truncated identifier would
# merge two distinct artifacts into one graph node and invent provenance that
# nobody can see. So the length is enforced here, loudly, instead.
MAX_IDENTIFIER_LENGTH = 256

# Structural separators. A piece containing any of them is REJECTED (see
# _reject_separators) rather than escaped, which keeps the identifier
# unambiguous and invertible without an escape grammar.
#
# Why these three and not the prototype's:
#
# TYPE_SEP (":") puts the type in the *key*. The prototype builds
# f"{name}|{table}" for BOTH model and dataset -- byte for byte identical -- so a
# model and a dataset sharing name+table collapse into one node. It also
# distinguishes file from other only by the literal prefixes "File - " / "Other
# - ", decorative text inside the value. With the type leading, the collision is
# structurally impossible and needs no merge logic.
#
# NS_SEP ("::") carries the namespace/organization. The prototype's identifier
# for a table is the bare tablename, so two same-named tables in different
# organizations collapse -- in a cross-space, multi-source index that invents
# provenance. "::" and not "." or "/": an LH host contains dots
# (lakehouse.prod.ibm.com) and a namespace contains slashes, so either would
# force escaping inside the very piece it delimits, on nearly every identifier.
#
# VERSION_SEP ("@") replaces the prototype's ".". Its "name.version" is
# ambiguous -- "a.b.c" splits two ways -- and its own spec admits the function
# "can be generated but not inverted".
TYPE_SEP = ":"
NS_SEP = "::"
VERSION_SEP = "@"
TABLE_SEP = "|"

# Checked against every piece. NS_SEP is two colons, so rejecting TYPE_SEP
# already covers it.
RESERVED_CHARACTERS = (TYPE_SEP, VERSION_SEP, TABLE_SEP)

# LhURI.__init__ injects this when a model/fileset URI carries no revision, and
# its guard is a SUBSTRING test (``if not revision in uristr``), so the injection
# is not even deterministic. Treating the injected default as "no revision" is
# what makes the same fileset converge whether or not it went through LhURI.
INJECTED_REVISION_DEFAULT = "granite-dot-build"

# Types with a dedicated identifier shape. Everything else -- BUCKET, UNDEFINED,
# and the five deprecated types the enum keeps "so we can deserialize old
# entries" -- falls to the generic <type>:<ns>::<name> rule.
#
# The generic rule is not a nicety: the prototype returns None for an
# unrecognized type, None becomes NULL in source/target, and the traversal reads
# NULL as a creation terminal. Such an artifact is not mis-identified, it
# *vanishes from the graph* as a false end-of-path. That matters most for the
# deprecated types, which are historical artifacts that will not be re-derivable
# once the upstream sources are switched off.
_TYPED_RULES = frozenset(
    {ArtifactType.TABLE, ArtifactType.MODEL, ArtifactType.DATASET, ArtifactType.FILESET}
)

# ArtifactType.UNDEFINED is the empty string, which would render as a leading
# ":" -- indistinguishable from a missing type. It gets a spelled-out token.
UNDEFINED_TYPE_TOKEN = "undefined"


class LineageIdentityError(ValueError):
    """An artifact cannot be given a canonical lineage identifier.

    Raised instead of returning a degraded identifier. A row that cannot be
    identified is dropped visibly and recorded by the caller; an identifier that
    is silently wrong merges unrelated artifacts, which nothing downstream can
    detect.
    """


def _reject_separators(piece: str, field: str, identity: "ArtifactIdentity") -> str:
    """Return ``piece`` unchanged, or raise if it contains a reserved character.

    Separators are not escaped, so a piece containing one would make the
    identifier ambiguous: ``fileset:ns::a@b@c|t`` cannot be split back into name
    and revision, and a fileset named ``a@b`` with no revision would produce the
    same identifier as one named ``a`` at revision ``b``. That is the silent merge
    this scheme exists to prevent, so such a piece is refused instead.

    Args:
        piece: the piece about to go into an identifier.
        field: its field name, for the error message.
        identity: the artifact being identified, for the error message.

    Returns:
        ``piece``, unchanged.

    Raises:
        LineageIdentityError: if ``piece`` contains a reserved character.
    """
    for reserved in RESERVED_CHARACTERS:
        if reserved in piece:
            raise LineageIdentityError(
                f"{field} contains the reserved character {reserved!r}, which would "
                f"make the identifier ambiguous: {piece!r} (in {identity!r})"
            )
    return piece


class ArtifactIdentity:
    """The pieces a canonical identifier is built from.

    Not every field is used by every type -- that is the point of the type
    dispatch. ``table`` is meaningless for a generic artifact and ``revision``
    only for a fileset, so both default to empty rather than being required.

    Attributes:
        artifact_type: the artifact's type; decides which pieces participate.
        namespace: namespace or organization -- ``host/namespace`` for Lakehouse,
            the space name for granite.build, ``entity/project`` for W&B. Present
            on every type, so two same-named artifacts in different
            organizations cannot collapse.
        name: the artifact's own name (a model label, a dataset name, a fileset
            label, or the registration name for generic types).
        table: the Lakehouse table the artifact lives in; empty for types that
            have none.
        revision: version/revision; only ``fileset`` puts it in the identifier.
    """

    __slots__ = ("artifact_type", "namespace", "name", "table", "revision")

    def __init__(
        self,
        artifact_type: ArtifactType,
        namespace: str,
        name: str = "",
        table: str = "",
        revision: str = "",
    ) -> None:
        self.artifact_type = artifact_type
        self.namespace = namespace
        self.name = name
        self.table = table
        self.revision = revision

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ArtifactIdentity):
            return NotImplemented
        return (
            self.artifact_type == other.artifact_type
            and self.namespace == other.namespace
            and self.name == other.name
            and self.table == other.table
            and self.revision == other.revision
        )

    def __hash__(self) -> int:
        return hash(
            (self.artifact_type, self.namespace, self.name, self.table, self.revision)
        )

    def __repr__(self) -> str:
        return (
            f"ArtifactIdentity(artifact_type={self.artifact_type!r}, "
            f"namespace={self.namespace!r}, name={self.name!r}, "
            f"table={self.table!r}, revision={self.revision!r})"
        )


def _type_token(artifact_type: ArtifactType) -> str:
    """Return the identifier's leading type token.

    ``ArtifactType.UNDEFINED`` is the empty string, which would render as a
    leading ``:`` and be indistinguishable from a missing type, so it is spelled
    out instead.
    """
    return (
        UNDEFINED_TYPE_TOKEN
        if artifact_type == ArtifactType.UNDEFINED
        else str(artifact_type.value)
    )


def _effective_revision(revision: str) -> str:
    """Drop the revision LhURI injects, keep a real one.

    ``LhURI.__init__`` appends ``granite-dot-build`` when a model/fileset URI has
    no revision, and its guard is a substring test, so whether a given URI gets
    the default is not reliable. Treating the default as "no revision" is what
    makes the same fileset converge whether or not it passed through ``LhURI``.
    """
    return "" if revision == INJECTED_REVISION_DEFAULT else revision


def canonical_id(identity: ArtifactIdentity) -> str:
    """Build the canonical lineage identifier for an artifact.

    Args:
        identity: the artifact's pieces. Only the pieces the type's rule uses are
            read, so a generic artifact needs no ``table`` or ``revision``.

    Returns:
        The canonical identifier -- the exact string stored in a lineage row's
        ``source``/``target`` column and joined on by the traversal.

    Raises:
        LineageIdentityError: if a piece the type's rule requires is empty, or if
            the result exceeds :data:`MAX_IDENTIFIER_LENGTH`. Both fail loudly
            rather than yielding an identifier that would merge distinct
            artifacts.
    """
    artifact_type = identity.artifact_type
    if not identity.namespace:
        raise LineageIdentityError(
            f"namespace is required for every artifact type (got {identity!r})"
        )

    namespace = _reject_separators(identity.namespace, "namespace", identity)
    head = f"{_type_token(artifact_type)}{TYPE_SEP}{namespace}{NS_SEP}"

    if artifact_type == ArtifactType.TABLE:
        if not identity.table:
            raise LineageIdentityError(f"table is required for a table ({identity!r})")
        body = _reject_separators(identity.table, "table", identity)
    elif artifact_type in (ArtifactType.MODEL, ArtifactType.DATASET):
        if not identity.name or not identity.table:
            raise LineageIdentityError(
                f"name and table are required for {artifact_type.value} ({identity!r})"
            )
        name = _reject_separators(identity.name, "name", identity)
        table = _reject_separators(identity.table, "table", identity)
        body = f"{name}{TABLE_SEP}{table}"
    elif artifact_type == ArtifactType.FILESET:
        if not identity.name or not identity.table:
            raise LineageIdentityError(
                f"name and table are required for a fileset ({identity!r})"
            )
        body = _reject_separators(identity.name, "name", identity)
        revision = _effective_revision(identity.revision)
        if revision:
            body += VERSION_SEP + _reject_separators(revision, "revision", identity)
        body += TABLE_SEP + _reject_separators(identity.table, "table", identity)
    else:
        # Generic rule: BUCKET, UNDEFINED, and the five deprecated types. Keeps
        # every ArtifactType identifiable, so NULL in source/target keeps its one
        # legitimate meaning -- the create/delete terminal.
        if not identity.name:
            raise LineageIdentityError(
                f"name is required for {_type_token(artifact_type)} ({identity!r})"
            )
        body = _reject_separators(identity.name, "name", identity)

    identifier = head + body
    if len(identifier) > MAX_IDENTIFIER_LENGTH:
        raise LineageIdentityError(
            f"canonical identifier is {len(identifier)} characters, over the "
            f"{MAX_IDENTIFIER_LENGTH} the source/target column holds; it would be "
            f"truncated silently and merge distinct artifacts: {identifier!r}"
        )
    return identifier


def parse_canonical_id(identifier: str) -> ArtifactIdentity:
    """Recover an artifact's pieces from its canonical identifier.

    The inverse of :func:`canonical_id` for every input it accepts, including
    pieces containing the separators themselves.

    Args:
        identifier: a canonical identifier.

    Returns:
        The pieces it was built from. ``table`` and ``revision`` come back empty
        for types whose rule does not use them.

    Raises:
        LineageIdentityError: if the string is not a well-formed identifier.
    """
    if not identifier:
        raise LineageIdentityError("empty identifier")

    ns_parts = identifier.split(NS_SEP)
    if len(ns_parts) != 2:
        raise LineageIdentityError(
            f"expected exactly one {NS_SEP!r} separator, found "
            f"{len(ns_parts) - 1}: {identifier!r}"
        )
    head, body = ns_parts

    head_parts = head.split(TYPE_SEP)
    if len(head_parts) != 2:
        raise LineageIdentityError(
            f"expected exactly one {TYPE_SEP!r} before {NS_SEP!r}: {identifier!r}"
        )
    # No empty-namespace check is needed here: an identifier missing its namespace
    # cannot reach this point. Splitting on "::" and then on ":" already rejects it
    # ("model::" leaves a head of "model", with no TYPE_SEP), so a guard here would
    # be unreachable.
    type_token, namespace = head_parts
    artifact_type = _parse_type_token(type_token, identifier)

    if artifact_type == ArtifactType.TABLE:
        return ArtifactIdentity(
            artifact_type=artifact_type,
            namespace=namespace,
            table=body,
        )

    if artifact_type in (ArtifactType.MODEL, ArtifactType.DATASET):
        name_raw, table_raw = _split_table(body, identifier)
        return ArtifactIdentity(
            artifact_type=artifact_type,
            namespace=namespace,
            name=name_raw,
            table=table_raw,
        )

    if artifact_type == ArtifactType.FILESET:
        name_raw, table_raw = _split_table(body, identifier)
        version_parts = name_raw.split(VERSION_SEP)
        if len(version_parts) > 2:
            raise LineageIdentityError(
                f"fileset has more than one {VERSION_SEP!r}: {identifier!r}"
            )
        revision = version_parts[1] if len(version_parts) == 2 else ""
        return ArtifactIdentity(
            artifact_type=artifact_type,
            namespace=namespace,
            name=version_parts[0],
            table=table_raw,
            revision=revision,
        )

    return ArtifactIdentity(
        artifact_type=artifact_type,
        namespace=namespace,
        name=body,
    )


def _parse_type_token(type_token: str, identifier: str) -> ArtifactType:
    """Map an identifier's leading token back to its ``ArtifactType``."""
    if type_token == UNDEFINED_TYPE_TOKEN:
        return ArtifactType.UNDEFINED
    try:
        return ArtifactType(type_token)
    except ValueError as exc:
        raise LineageIdentityError(
            f"unknown artifact type {type_token!r} in {identifier!r}"
        ) from exc


def _split_table(body: str, identifier: str) -> tuple[str, str]:
    """Split an identifier body into its name-ish part and its table."""
    parts = body.split(TABLE_SEP)
    if len(parts) != 2:
        raise LineageIdentityError(
            f"expected exactly one {TABLE_SEP!r}, found "
            f"{len(parts) - 1}: {identifier!r}"
        )
    if not parts[1]:
        raise LineageIdentityError(f"identifier has an empty table: {identifier!r}")
    return parts[0], parts[1]
