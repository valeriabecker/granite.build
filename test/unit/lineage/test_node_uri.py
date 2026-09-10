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

"""Tests for the URI a lineage node is identified by in the UI.

The property that actually matters is the last class here: distinct identifiers
must never share a URI. The frontend deduplicates refs by ``uri``, so a collision
does not degrade the display -- it merges two artifacts into one node.
"""

import itertools

import pytest

from gbserver.lineage.identity import ArtifactIdentity, canonical_id
from gbserver.lineage.node_uri import (
    node_uri,
    synthesized_uri_for_identifier,
)
from gbserver.types.artifact import ArtifactType


class TestStoredUriWins:
    """A recorded URI is returned verbatim; it is the artifact's real one."""

    def test_stored_uri_is_preferred_over_reconstruction(self):
        # An hf:// URI is NOT derivable from an identifier -- the scheme is not
        # encoded -- so this is the only way it can ever be right.
        assert node_uri("model:org::repo|tbl", "hf:///org/repo") == "hf:///org/repo"

    def test_stored_s3_uri_survives(self):
        assert node_uri("bucket:space::b1", "s3://bkt/path") == "s3://bkt/path"

    def test_stored_uri_wins_even_for_lakehouse(self):
        # The reconstruction would be right here, but the recorded value is still
        # the source of truth: it is what the artifact was registered with.
        assert (
            node_uri("table:prod/ns::tbl", "lh://custom.host/ns/tables/tbl")
            == "lh://custom.host/ns/tables/tbl"
        )

    def test_empty_stored_uri_falls_back(self):
        assert node_uri("table:prod/ns::tbl", "") == "lh://prod/ns/tables/tbl"


class TestNotANode:
    """``None`` means "not a node" and must never reach the UI as a uri."""

    def test_terminal_marker_is_not_a_node(self):
        assert node_uri("") is None

    def test_terminal_marker_with_a_uri_is_still_not_a_node(self):
        assert node_uri("", "s3://bkt/x") is None

    def test_malformed_identifier_is_not_a_node(self):
        assert synthesized_uri_for_identifier("no-separators-here") is None

    def test_unknown_type_token_is_not_a_node(self):
        assert synthesized_uri_for_identifier("nonsense:ns::name") is None


class TestLakehouseReconstruction:
    """``lh://`` is the one scheme fully determined by the stored pieces."""

    @pytest.mark.parametrize(
        "identifier,expected",
        [
            ("table:prod/ns::tbl", "lh://prod/ns/tables/tbl"),
            ("model:prod/ns::label|tbl", "lh://prod/ns/models/tbl/label"),
            ("dataset:prod/ns::ds|tbl", "lh://prod/ns/datasets/tbl/ds"),
            ("fileset:staging/ns::fs@v2|tbl", "lh://staging/ns/filesets/tbl/fs/v2"),
            # A custom hostname is explicitly allowed by LhURI.
            ("table:lh.ibm.com/ns::tbl", "lh://lh.ibm.com/ns/tables/tbl"),
        ],
    )
    def test_shapes_match_the_lh_uri_grammar(self, identifier, expected):
        assert synthesized_uri_for_identifier(identifier) == expected

    def test_missing_revision_is_omitted_not_defaulted(self):
        # LhURI injects "granite-dot-build" when a model URI has no revision, and
        # canonical_id treats that injected value as "no revision". Emitting it
        # here would assert a revision the index never recorded.
        uri = synthesized_uri_for_identifier("model:prod/ns::label|tbl")
        assert uri == "lh://prod/ns/models/tbl/label"
        assert "granite-dot-build" not in uri

    def test_injected_default_revision_does_not_reappear(self):
        identifier = canonical_id(
            ArtifactIdentity(
                artifact_type=ArtifactType.FILESET,
                namespace="prod/ns",
                name="fs",
                table="tbl",
                revision="granite-dot-build",
            )
        )
        assert synthesized_uri_for_identifier(identifier) == (
            "lh://prod/ns/filesets/tbl/fs"
        )

    def test_namespace_without_a_host_is_not_lakehouse_shaped(self):
        # No host means no authority for the URI, so it takes the fallback form
        # rather than producing "lh:///ns/...".
        uri = synthesized_uri_for_identifier("table:justns::tbl")
        assert uri.startswith("gb://")


class TestFallbackForm:
    """Non-Lakehouse artifacts get a synthetic, visibly non-resolvable URI."""

    def test_bucket_gets_a_gb_uri(self):
        assert synthesized_uri_for_identifier("bucket:space::b1") == (
            "gb://artifact?kind=bucket&namespace=space&name=b1"
        )

    def test_undefined_type_is_spelled_out(self):
        assert "kind=undefined" in synthesized_uri_for_identifier("undefined:space::x")

    def test_no_scheme_is_guessed(self):
        # The index never recorded a scheme for these, so claiming hf:// or s3://
        # would assert an origin that was never known.
        uri = synthesized_uri_for_identifier("bucket:org::repo")
        assert not uri.startswith("hf:")
        assert not uri.startswith("s3:")


class TestNoCollisions:
    """The invariant: two different identifiers never share a URI.

    A collision here re-creates the exact node-merging bug this module exists to
    prevent, so it is checked exhaustively over the pieces rather than by example.
    """

    def test_distinct_identifiers_yield_distinct_uris(self):
        identities = []
        for kind in (
            ArtifactType.TABLE,
            ArtifactType.MODEL,
            ArtifactType.DATASET,
            ArtifactType.FILESET,
            ArtifactType.BUCKET,
            ArtifactType.UNDEFINED,
        ):
            for namespace in ("prod/ns", "prod/other", "staging/ns", "bare"):
                for name in ("a", "b"):
                    for table in ("t1", "t2"):
                        for revision in ("", "v1"):
                            identity = ArtifactIdentity(
                                artifact_type=kind,
                                namespace=namespace,
                                name=name,
                                table=table,
                                revision=revision,
                            )
                            try:
                                identifier = canonical_id(identity)
                            except Exception:
                                continue
                            identities.append(identifier)

        by_uri: dict = {}
        for identifier in set(identities):
            uri = synthesized_uri_for_identifier(identifier)
            assert uri is not None
            by_uri.setdefault(uri, []).append(identifier)

        collisions = {u: ids for u, ids in by_uri.items() if len(ids) > 1}
        assert not collisions, f"URI collisions merge distinct nodes: {collisions}"

    def test_a_slash_in_a_name_cannot_forge_a_path_segment(self):
        # A name containing "/" would otherwise add a path segment and let two
        # different artifacts render as one URI.
        a = synthesized_uri_for_identifier("dataset:prod/ns::x/y|tbl")
        b = synthesized_uri_for_identifier("dataset:prod/ns::x|y/tbl")
        assert a != b

    def test_a_model_and_a_dataset_do_not_share_a_uri(self):
        # The collision §0(0) of the plan exists to prevent: same name and table,
        # different type.
        model = synthesized_uri_for_identifier("model:prod/ns::same|tbl")
        dataset = synthesized_uri_for_identifier("dataset:prod/ns::same|tbl")
        assert model != dataset
