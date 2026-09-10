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

"""Tests for canonical lineage identifiers.

Several tests here pin down corrections to the scheme ported from
``prototype-lineage-py``; each such test names the failure it prevents, because
every one of them is a silent failure -- a wrong identifier merges unrelated
artifacts or drops one from the graph, and nothing downstream can tell.
"""

import itertools

import pytest

from gbserver.lineage.identity import (
    INJECTED_REVISION_DEFAULT,
    MAX_IDENTIFIER_LENGTH,
    ArtifactIdentity,
    LineageIdentityError,
    canonical_id,
    parse_canonical_id,
)
from gbserver.types.artifact import ArtifactType

# Types whose rule needs name+table, so one fixture serves both.
NAME_TABLE_TYPES = (ArtifactType.MODEL, ArtifactType.DATASET)

# Every type that falls to the generic <type>:<ns>::<name> rule.
GENERIC_TYPES = (
    ArtifactType.BUCKET,
    ArtifactType.UNDEFINED,
    ArtifactType.SEED_DATA,
    ArtifactType.GENERATED_DATA,
    ArtifactType.TUNING_DATA,
    ArtifactType.BASE_MODEL,
    ArtifactType.TUNED_MODEL,
)


def identity_for(artifact_type: ArtifactType, **overrides) -> ArtifactIdentity:
    """Build a valid identity for ``artifact_type``, filling required pieces."""
    fields: dict = {"namespace": "host/ns", "name": "thing", "table": "tbl"}
    fields.update(overrides)
    return ArtifactIdentity(artifact_type=artifact_type, **fields)


class TestRoundTrip:
    """parse_canonical_id inverts canonical_id for every accepted input."""

    @pytest.mark.parametrize(
        "identity",
        [
            ArtifactIdentity(ArtifactType.TABLE, "host/ns", table="tbl"),
            ArtifactIdentity(ArtifactType.MODEL, "host/ns", name="m", table="tbl"),
            ArtifactIdentity(ArtifactType.DATASET, "host/ns", name="d", table="tbl"),
            ArtifactIdentity(ArtifactType.FILESET, "host/ns", name="f", table="tbl"),
            ArtifactIdentity(
                ArtifactType.FILESET, "host/ns", name="f", table="tbl", revision="v1"
            ),
            ArtifactIdentity(ArtifactType.BUCKET, "my-space", name="n"),
            ArtifactIdentity(ArtifactType.UNDEFINED, "my-space", name="n"),
            ArtifactIdentity(ArtifactType.TUNING_DATA, "my-space", name="n"),
        ],
        ids=lambda i: str(i.artifact_type.name),
    )
    def test_round_trip(self, identity):
        assert parse_canonical_id(canonical_id(identity)) == identity


class TestSeparatorsAreRejected:
    """A piece containing a separator is refused, not escaped.

    Escaping is deliberately absent for now, so an ambiguous identifier must never
    be built: without it, ``fileset:ns::a@b@c|t`` cannot be split back into name
    and revision, and a fileset named ``a@b`` with no revision would collide with
    one named ``a`` at revision ``b``. Refusing the piece surfaces the case as an
    error instead of a silent merge, and escaping can be added later without
    invalidating any identifier already stored.
    """

    RESERVED = (":", "|", "@")

    @pytest.mark.parametrize("reserved", RESERVED)
    def test_namespace_with_separator_is_rejected(self, reserved):
        with pytest.raises(LineageIdentityError, match="reserved character"):
            canonical_id(identity_for(ArtifactType.MODEL, namespace=f"a{reserved}b"))

    @pytest.mark.parametrize("reserved", RESERVED)
    def test_name_with_separator_is_rejected(self, reserved):
        with pytest.raises(LineageIdentityError, match="reserved character"):
            canonical_id(identity_for(ArtifactType.MODEL, name=f"a{reserved}b"))

    @pytest.mark.parametrize("reserved", RESERVED)
    def test_table_with_separator_is_rejected(self, reserved):
        with pytest.raises(LineageIdentityError, match="reserved character"):
            canonical_id(identity_for(ArtifactType.MODEL, table=f"a{reserved}b"))

    @pytest.mark.parametrize("reserved", RESERVED)
    def test_fileset_revision_with_separator_is_rejected(self, reserved):
        with pytest.raises(LineageIdentityError, match="reserved character"):
            canonical_id(identity_for(ArtifactType.FILESET, revision=f"a{reserved}b"))

    def test_double_colon_in_a_piece_is_rejected(self):
        """The namespace separator is two colons, so rejecting ":" covers it."""
        with pytest.raises(LineageIdentityError, match="reserved character"):
            canonical_id(identity_for(ArtifactType.MODEL, namespace="a::b"))

    def test_error_names_the_offending_field(self):
        with pytest.raises(LineageIdentityError, match="^table contains"):
            canonical_id(identity_for(ArtifactType.MODEL, table="a|b"))

    def test_a_dot_is_not_reserved(self):
        """Only ":", "|" and "@" are structural -- a dotted host is fine, which is
        why the namespace separator is "::" and not "." (an LH host is dotted).
        """
        identity = ArtifactIdentity(
            ArtifactType.TABLE, "lakehouse.prod.ibm.com/ns", table="tbl"
        )
        assert parse_canonical_id(canonical_id(identity)) == identity

    def test_a_slash_is_not_reserved(self):
        """A namespace is "host/namespace", so "/" must pass through."""
        identity = identity_for(ArtifactType.MODEL, namespace="host/deep/ns")
        assert parse_canonical_id(canonical_id(identity)) == identity


class TestCollisions:
    """The merges the prototype's identifier performs silently."""

    def test_model_and_dataset_do_not_collide(self):
        """prototype-lineage-py identifiers.py:102-105 builds f"{name}|{table}" for
        BOTH, byte for byte, so a model and a dataset sharing name+table become one
        graph node -- a model appears to be trained on data it never saw.
        """
        model = identity_for(ArtifactType.MODEL, name="shared", table="tbl")
        dataset = identity_for(ArtifactType.DATASET, name="shared", table="tbl")
        assert canonical_id(model) != canonical_id(dataset)

    def test_same_table_in_different_namespaces_does_not_collide(self):
        """The prototype's table identifier is the bare tablename, with no
        namespace, so two same-named tables in different organizations collapse
        into one node and invent provenance.
        """
        one = ArtifactIdentity(ArtifactType.TABLE, "org-a/ns", table="events")
        two = ArtifactIdentity(ArtifactType.TABLE, "org-b/ns", table="events")
        assert canonical_id(one) != canonical_id(two)

    def test_generic_types_do_not_collide_across_namespaces(self):
        """The namespace is on every type, not just the Lakehouse ones."""
        one = ArtifactIdentity(ArtifactType.BUCKET, "space-a", name="n")
        two = ArtifactIdentity(ArtifactType.BUCKET, "space-b", name="n")
        assert canonical_id(one) != canonical_id(two)

    def test_deprecated_types_keep_their_own_identity(self):
        """A deprecated type is not folded into its modern equivalent."""
        ids = {
            canonical_id(identity_for(t, name="n"))
            for t in (
                ArtifactType.TUNING_DATA,
                ArtifactType.SEED_DATA,
                ArtifactType.DATASET,
            )
        }
        assert len(ids) == 3

    def test_dotted_fileset_names_stay_distinguishable(self):
        """The prototype separates name and version with ".", so "a.b.c" splits two
        ways and its own spec admits the function cannot be inverted. With "@" as
        the version separator, a dot is just a character and both round-trip.
        """
        one = identity_for(ArtifactType.FILESET, name="a", revision="b.c")
        two = identity_for(ArtifactType.FILESET, name="a.b", revision="c")
        assert canonical_id(one) != canonical_id(two)
        assert parse_canonical_id(canonical_id(one)) == one
        assert parse_canonical_id(canonical_id(two)) == two

    def test_version_separator_inside_a_name_cannot_forge_a_revision(self):
        """Without escaping, a name containing "@" would collide with a real
        revision, so it is refused outright.
        """
        with pytest.raises(LineageIdentityError, match="reserved character"):
            canonical_id(identity_for(ArtifactType.FILESET, name="a@b"))


class TestNamespaceSeparatorChoice:
    """Why "::" and not "." or "/" -- both appear inside real namespaces."""

    def test_dotted_host_needs_no_escaping(self):
        identity = ArtifactIdentity(
            ArtifactType.TABLE, "lakehouse.prod.ibm.com/ns", table="tbl"
        )
        identifier = canonical_id(identity)
        assert "lakehouse.prod.ibm.com/ns" in identifier, identifier
        assert "\\" not in identifier, identifier
        assert parse_canonical_id(identifier) == identity


class TestEveryArtifactTypeIsIdentifiable:
    """No ArtifactType may fall through to a None identifier.

    The prototype returns None for an unrecognized type; None becomes NULL in
    source/target, and the traversal reads NULL as a creation terminal. Such an
    artifact does not get a wrong identifier -- it disappears from the graph as a
    false end-of-path. That is unacceptable for the deprecated types, which are
    historical artifacts that will not be re-derivable.
    """

    @pytest.mark.parametrize("artifact_type", list(ArtifactType), ids=lambda t: t.name)
    def test_type_produces_an_identifier(self, artifact_type):
        identifier = canonical_id(identity_for(artifact_type))
        assert identifier
        assert parse_canonical_id(identifier).artifact_type == artifact_type

    def test_undefined_type_is_not_an_empty_token(self):
        """ArtifactType.UNDEFINED is the empty string, which would render as a
        leading ":" and be indistinguishable from a missing type.
        """
        identifier = canonical_id(identity_for(ArtifactType.UNDEFINED, name="n"))
        assert not identifier.startswith(":"), identifier


class TestLengthValidation:
    """The source/target column is String(256) and truncates silently."""

    def test_over_limit_raises(self):
        identity = identity_for(ArtifactType.MODEL, name="x" * MAX_IDENTIFIER_LENGTH)
        with pytest.raises(LineageIdentityError, match="truncated"):
            canonical_id(identity)

    def test_at_limit_is_accepted(self):
        identity = identity_for(ArtifactType.BUCKET, namespace="ns", name="n")
        head = len(canonical_id(identity)) - len("n")
        identity = identity_for(
            ArtifactType.BUCKET,
            namespace="ns",
            name="n" * (MAX_IDENTIFIER_LENGTH - head),
        )
        assert len(canonical_id(identity)) == MAX_IDENTIFIER_LENGTH

    def test_limit_counts_the_whole_identifier(self):
        """The type token and namespace count too, not just the name."""
        identity = identity_for(
            ArtifactType.BUCKET,
            namespace="n" * 200,
            name="n" * (MAX_IDENTIFIER_LENGTH - 200),
        )
        with pytest.raises(LineageIdentityError, match="truncated"):
            canonical_id(identity)


class TestInjectedRevision:
    """LhURI.__init__ appends a default revision, guarded by a substring test."""

    def test_injected_default_is_treated_as_absent(self):
        """So the same fileset converges whether or not it passed through LhURI.

        The guard is ``if not revision in uristr`` -- a substring test -- so
        whether a given URI receives the default is not even deterministic.
        """
        injected = identity_for(
            ArtifactType.FILESET, revision=INJECTED_REVISION_DEFAULT
        )
        absent = identity_for(ArtifactType.FILESET, revision="")
        assert canonical_id(injected) == canonical_id(absent)

    def test_a_real_revision_is_kept(self):
        with_revision = identity_for(ArtifactType.FILESET, revision="v1")
        without = identity_for(ArtifactType.FILESET, revision="")
        assert canonical_id(with_revision) != canonical_id(without)
        assert parse_canonical_id(canonical_id(with_revision)).revision == "v1"


class TestRequiredPieces:
    """A missing required piece fails loudly, never degrades."""

    # One typed rule and one generic: the namespace check runs before the type
    # dispatch, so exercising the whole enum here would re-test one branch eleven
    # times. The enum is walked where it matters -- see
    # TestEveryArtifactTypeIsIdentifiable.
    @pytest.mark.parametrize(
        "artifact_type",
        [ArtifactType.MODEL, ArtifactType.BUCKET],
        ids=lambda t: t.name,
    )
    def test_namespace_is_required(self, artifact_type):
        with pytest.raises(LineageIdentityError, match="namespace"):
            canonical_id(identity_for(artifact_type, namespace=""))

    def test_table_required_for_table_type(self):
        with pytest.raises(LineageIdentityError, match="table"):
            canonical_id(identity_for(ArtifactType.TABLE, table=""))

    @pytest.mark.parametrize("artifact_type", NAME_TABLE_TYPES, ids=lambda t: t.name)
    def test_name_and_table_required(self, artifact_type):
        with pytest.raises(LineageIdentityError):
            canonical_id(identity_for(artifact_type, name=""))
        with pytest.raises(LineageIdentityError):
            canonical_id(identity_for(artifact_type, table=""))

    @pytest.mark.parametrize(
        "artifact_type",
        [ArtifactType.BUCKET, ArtifactType.UNDEFINED],
        ids=lambda t: t.name,
    )
    def test_name_required_for_generic_types(self, artifact_type):
        with pytest.raises(LineageIdentityError, match="name"):
            canonical_id(identity_for(artifact_type, name=""))

    def test_generic_type_needs_no_table_or_revision(self):
        """The point of the type dispatch: an s3:// artifact has neither, and
        ArtifactRegistration has no such fields at all.
        """
        identity = ArtifactIdentity(ArtifactType.BUCKET, "my-space", name="n")
        assert parse_canonical_id(canonical_id(identity)) == identity


class TestParseRejectsMalformed:
    @pytest.mark.parametrize(
        "identifier",
        [
            "",
            "no-separators",
            "model:host/ns",  # no "::"
            "model:a::b::c",  # two namespace separators
            "host/ns::name",  # no type
            "nosuchtype:host/ns::n",
            "model:host/ns::justname",  # model needs name|table
            "model:host/ns::a|b|c",  # two table separators
            "model:host/ns::a|",  # empty table
            "fileset:host/ns::a@b@c|t",  # two version separators
            "model:::n|t",  # empty namespace
            "model:::",  # nothing but separators
        ],
    )
    def test_malformed_raises(self, identifier):
        with pytest.raises(LineageIdentityError):
            parse_canonical_id(identifier)


class TestIdentityValueSemantics:
    """ArtifactIdentity is compared and hashed by value, so it can key a dict."""

    def test_equal_identities_are_equal(self):
        assert identity_for(ArtifactType.MODEL) == identity_for(ArtifactType.MODEL)

    def test_comparison_with_another_type_is_not_an_error(self):
        assert identity_for(ArtifactType.MODEL) != "model:host/ns::thing|tbl"

    def test_usable_as_a_dict_key(self):
        one, two = identity_for(ArtifactType.MODEL), identity_for(ArtifactType.MODEL)
        assert len({one, two}) == 1

    def test_repr_names_every_piece(self):
        text = repr(identity_for(ArtifactType.FILESET, revision="v1"))
        assert "artifact_type" in text and "v1" in text


class TestFilesetRequiredPieces:
    """The fileset rule needs name and table, checked on its own branch."""

    def test_name_is_required(self):
        with pytest.raises(LineageIdentityError, match="fileset"):
            canonical_id(identity_for(ArtifactType.FILESET, name=""))

    def test_table_is_required(self):
        with pytest.raises(LineageIdentityError, match="fileset"):
            canonical_id(identity_for(ArtifactType.FILESET, table=""))
