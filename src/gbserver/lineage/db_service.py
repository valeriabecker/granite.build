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

"""Serve artifact lineage out of the local index instead of W&B.

``POST /lineage/artifact`` delegates to a :class:`LineageService` and returns 404
when that service returns ``None`` -- which is what ``NoopLineageService`` does for
everything, so standalone shows "Lineage is not available" for every artifact
today. This implementation answers the same question from the lineage table, so a
deployment with no W&B still has lineage.

Only the graph is implemented here. The event-emission and tag-search methods of
the interface belong to the W&B-shaped write path (this service reads an index
another writer populates), so they are explicit no-ops rather than pretending to
succeed -- see each method for why its degenerate value is the safe one.

Root resolution is the part worth reading. The request identifies an artifact the
way a UI can -- a name, or a URL -- while the index is keyed by canonical
identifier. Rather than reconstruct an identifier from a name (which cannot be done:
the identifier needs a namespace and a type the request does not carry), the lookup
goes the other way and matches against what rows already store. That makes lineage
imported from a source with no uuid reachable, which was the point of keying the
index by identifier in the first place.
"""

import logging
from typing import Callable, Dict, List, Optional, Tuple

from gbserver.lineage.graph_builder import build_graph_dict
from gbserver.lineage.openlineage_service import LineageService
from gbserver.lineage.walk import DEFAULT_MAX_NODES_PER_LEVEL, Direction, walk_lineage
from gbserver.storage.lineage_row_storage import ILineageRowStorage

logger = logging.getLogger(__name__)

# The wire keeps W&B's direction words; the walk names directions for what they
# walk toward. The mapping is explicit rather than positional because the words are
# easy to invert by accident: the prototype's "downstream" walks toward origins,
# but the live W&B backend treats downstream as used_by() -- toward descendants --
# and the frontend sends it with that meaning. This follows the live backend.
_WIRE_DIRECTIONS = {
    "downstream": Direction.DESCENDANTS,
    "upstream": Direction.ANCESTORS,
    "both": Direction.BOTH,
}


class DBLineageService(LineageService):
    """Read artifact lineage from the local lineage index.

    Args:
        storage: the lineage row storage to read. Defaults to the process-wide
            admin storage, resolved lazily so importing this module does not
            require a configured database.
    """

    def __init__(self, storage: Optional[ILineageRowStorage] = None) -> None:
        self._storage = storage

    @property
    def storage(self) -> ILineageRowStorage:
        """The lineage row storage, resolved on first use."""
        if self._storage is None:
            from gbserver.storage.singleton_storage import get_admin_storage

            self._storage = get_admin_storage().lineage_row_storage
        return self._storage

    def get_artifact_graph(
        self,
        artifact_name: Optional[str] = None,
        artifact_url: Optional[str] = None,
        artifact_type: Optional[str] = None,
        max_depth: int = 10,
        direction: str = "downstream",
    ) -> Optional[Dict]:
        """Return the lineage graph for one artifact, or ``None`` if unknown.

        Args:
            artifact_name: the artifact's name, matched against rows' promoted
                name columns and against canonical identifiers.
            artifact_url: the artifact's URI, matched against rows' stored URIs.
                The more precise of the two: a URI identifies one artifact, a name
                may not.
            artifact_type: when given, the resolved root must have this type, and a
                mismatch is an error rather than a miss -- the caller asserted
                something about the artifact that turned out to be false.
            max_depth: how many hops to expand.
            direction: ``downstream``, ``upstream`` or ``both``, in wire terms.

        Returns:
            ``{root_id, nodes, edges, truncated}``, or ``None`` when no row
            mentions the artifact. ``None`` becomes the 404 the frontend shows as
            "not available", so it must mean "unknown here", never "no lineage":
            an artifact that IS in the index but has no edges yet returns a graph
            with just its own node.

        Raises:
            ValueError: if ``direction`` is not a wire direction, or if
                ``artifact_type`` contradicts the resolved root. The API layer maps
                this to a 400.
        """
        walk_direction = _WIRE_DIRECTIONS.get(direction)
        if walk_direction is None:
            raise ValueError(
                f"direction must be one of {sorted(_WIRE_DIRECTIONS)}, got {direction!r}"
            )

        root = self._resolve_root(artifact_name, artifact_url)
        if root is None:
            return None
        root_identifier, root_uri, root_kind = root

        if artifact_type and root_kind and artifact_type != root_kind:
            raise ValueError(
                f"Artifact type mismatch: expected {artifact_type!r}, but "
                f"{root_identifier!r} has type {root_kind!r}"
            )

        graph = walk_lineage(
            storage=self.storage,
            seeds=[root_identifier],
            direction=walk_direction,
            max_depth=max_depth,
            max_nodes_per_level=DEFAULT_MAX_NODES_PER_LEVEL,
        )
        return build_graph_dict(graph, root_identifier, root_uri=root_uri)

    def get_build_graph(
        self,
        build_id: str,
        direction: str = "both",
        max_depth: int = 10,
        within_build_only: bool = False,
    ) -> Optional[Dict]:
        """Return the lineage graph seeded from every artifact a build touched.

        A build is not a graph node; it is a way to *seed* one. One indexed query
        gets the build's rows, their endpoints become the seed set, and the walk is
        the ordinary one from there -- no separate traversal.

        Args:
            build_id: the build to seed from.
            direction: ``downstream``, ``upstream`` or ``both``, in wire terms.
            max_depth: how many hops to expand beyond the seeds.
            within_build_only: keep the walk inside this build's own rows. Both
                scopes are useful -- the build's internal graph, or the full chain
                it sits in -- so this is a filter on one walk rather than a second
                implementation.

        Returns:
            ``{root_id, nodes, edges, truncated}`` with ``root_id`` set to
            ``build_id``, or ``None`` when the build has no rows. ``root_id`` names
            no artifact node here, so nothing is flagged ``is_root``: a build's
            graph has several roots, and picking one arbitrarily would misreport
            which artifact was asked about.

        Raises:
            ValueError: if ``direction`` is not a wire direction.
        """
        walk_direction = _WIRE_DIRECTIONS.get(direction)
        if walk_direction is None:
            raise ValueError(
                f"direction must be one of {sorted(_WIRE_DIRECTIONS)}, got {direction!r}"
            )
        if not build_id:
            return None

        rows = self.storage.get_rows_by_build(build_id)
        if not rows:
            return None

        seeds = {row.source for row in rows} | {row.target for row in rows}
        graph = walk_lineage(
            storage=self.storage,
            seeds=seeds,
            direction=walk_direction,
            max_depth=max_depth,
            build_id=build_id if within_build_only else None,
        )
        return build_graph_dict(graph, root_identifier=build_id, root_is_artifact=False)

    def _resolve_root(
        self,
        artifact_name: Optional[str],
        artifact_url: Optional[str],
    ) -> Optional[Tuple[str, str, str]]:
        """Find the canonical identifier of the artifact a request names.

        Resolution is by lookup, not by construction: an identifier needs a
        namespace and a type that the request does not carry, so building one from
        a bare name would guess. Matching against rows also means an artifact is
        findable by whatever its source recorded, including an imported one with no
        uuid.

        Order matters. A URI identifies exactly one artifact; a name may match
        several, so the URI is tried first and the name is a fallback.

        Args:
            artifact_name: the name to match, if any.
            artifact_url: the URI to match, if any.

        Returns:
            ``(identifier, uri, kind)`` for the resolved root, or ``None`` if no row
            mentions it. ``uri`` and ``kind`` come from the matched row so the
            caller need not re-derive them.
        """
        if artifact_url:
            found = self._match_rows(lambda row: _endpoints_by_uri(row, artifact_url))
            if found is not None:
                return found

        if artifact_name:
            exact = self._match_rows(
                lambda row: _endpoints_by_identifier(row, artifact_name)
            )
            if exact is not None:
                return exact
            return self._match_rows(lambda row: _endpoints_by_name(row, artifact_name))

        return None

    def _match_rows(self, matcher: Callable) -> Optional[Tuple[str, str, str]]:
        """Scan rows for an endpoint a matcher accepts.

        A scan, deliberately, and only on the root lookup: the columns that would
        make this indexed are ``source``/``target``, which hold canonical
        identifiers, and the request does not carry one to match on. Every hop
        *after* the root is a single indexed query, so this happens once per
        request rather than per level.

        Paged rather than loaded whole, and it returns at the first match, so the
        cost is bounded by where the match falls instead of by the table size.

        Args:
            matcher: called with a row; returns ``(identifier, uri, kind)`` for a
                matching endpoint, or ``None``.

        Returns:
            The first match, or ``None``. A storage failure is logged and reported
            as "not found": the caller turns that into a 404, which is the same
            thing the frontend already shows when lineage is unavailable.
        """
        try:
            for page in self.storage.get_paged():
                for row in page:
                    matched = matcher(row)
                    if matched is not None:
                        return matched
        except Exception:
            logger.exception("Lineage root lookup failed")
        return None

    # -- Write-path methods. This service reads an index that another writer
    # populates, so none of these apply; each returns the value that makes a caller
    # behave correctly rather than one that merely avoids an exception.

    def emit_event(self, event: Dict) -> None:
        """Ignore an emitted event.

        Rows are written by the lineage sink from build state, not by callers
        pushing events here. Raising would break a recorder that fans out to every
        configured provider.
        """
        return None

    def search_lineage_by_tags(
        self, tags: List[str], limit: int = 10, offset: int = 0
    ) -> Tuple[int, List[Dict]]:
        """Return no results: the index stores no run tags to search by."""
        return 0, []

    def count_events_by_tags(
        self, tags: List[str], required_tags: Optional[List[str]] = None
    ) -> int:
        """Return 0: see :meth:`search_lineage_by_tags`."""
        return 0

    def count_runs_by_tags(
        self, tags: List[str], required_tags: Optional[List[str]] = None
    ) -> int:
        """Return 0: see :meth:`search_lineage_by_tags`."""
        return 0

    def filter_unrecorded(
        self,
        target_ids: set[str],
        expected_counts: Optional[dict[str, int]] = None,
        on_query_error: Optional[Callable[[Exception], None]] = None,
    ) -> set[str]:
        """Report every candidate as unrecorded.

        The interface documents that implementations must fail toward
        re-recording, because recording is idempotent and this is only an
        efficiency filter. Returning ``target_ids`` unchanged is that safe answer.

        It is deliberately not answered from the index here: this service is the
        *read* side, and the sink that writes rows does its own presence-based
        dedup against the same table (``get_recorded_target_runs``). Answering here
        too would put the same decision in two places, keyed differently -- target
        id here versus target run uuid there -- and the two would disagree.
        """
        return target_ids


def _endpoints_by_uri(row, uri: str) -> Optional[Tuple[str, str, str]]:
    """Match a row endpoint by its stored URI."""
    if row.source_uri and row.source_uri == uri:
        return row.source, row.source_uri, row.source_kind
    if row.target_uri and row.target_uri == uri:
        return row.target, row.target_uri, row.target_kind
    return None


def _endpoints_by_identifier(row, identifier: str) -> Optional[Tuple[str, str, str]]:
    """Match a row endpoint by canonical identifier.

    Lets a caller that already has an identifier -- a link built from a previous
    graph response -- pass it as the name and get an exact hit.
    """
    if row.source == identifier:
        return row.source, row.source_uri, row.source_kind
    if row.target == identifier:
        return row.target, row.target_uri, row.target_kind
    return None


def _endpoints_by_name(row, name: str) -> Optional[Tuple[str, str, str]]:
    """Match a row endpoint by its promoted name column.

    The loosest match, and last: a name is not unique across namespaces, so this
    can resolve to one of several artifacts. It is still worth having, because a UI
    that only knows a display name has nothing else to ask with.
    """
    if row.source_name and row.source_name == name:
        return row.source, row.source_uri, row.source_kind
    if row.target_name and row.target_name == name:
        return row.target, row.target_uri, row.target_kind
    return None
