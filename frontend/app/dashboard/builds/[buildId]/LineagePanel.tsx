'use client'

import * as React from 'react'
import { Button, ComposedModal, InlineLoading, Modal, ModalBody, ModalFooter, ModalHeader, OverflowMenu, OverflowMenuItem } from '@carbon/react'
import {
  ArrowLeft,
  ArrowRight,
  CenterSquare,
  Launch,
  ZoomFit,
  ZoomIn,
  ZoomOut,
} from '@carbon/icons-react'
import { useRouter } from 'next/navigation'
import styles from './LineagePanel.module.scss'
import type { ElkExtendedEdge } from 'elkjs'
import { parse as parseYaml } from 'yaml'
import { useQuery, useQueries } from '@tanstack/react-query'
import type { Build, BuildStatusDetail } from '@/types'
import { getArtifact } from '@/api/gbserver'
import { getBuildArchiveFiles, getBuildLineage, type LineageDirection } from '@/api/gbserver'
import Graph, { type ElkNodeEx, type GraphHandle, type NodeType } from '@/components/LineageGraph/Graph'
import {
  artifactTypeToNodeType,
  jobstatsToGraph,
  mergeGraphs,
  renameNodes,
} from '@/components/LineageGraph/jobstatsGraph'

const ACTIVE_STATUSES = new Set(['running', 'submitted', 'pending'])

interface PlannedTarget {
  target_name: string
  inputs: Record<string, string>
  outputs: Record<string, string>
}

function parseDefinitionTargets(yaml: string): PlannedTarget[] {
  try {
    const def = parseYaml(yaml) as {
      targets?: Record<string, {
        inputs?: Record<string, unknown>
        outputs?: Record<string, unknown>
      } | null>
    }
    if (!def?.targets) return []
    return Object.entries(def.targets).map(([name, config]) => ({
      target_name: name,
      inputs: Object.fromEntries(
        Object.entries(config?.inputs ?? {}).map(([k, v]) => [k, String(v ?? '')])
      ),
      outputs: Object.fromEntries(
        Object.entries(config?.outputs ?? {}).map(([k, v]) => [k, String(v ?? '')])
      ),
    }))
  } catch {
    return []
  }
}

interface LineagePanelProps {
  build: Build | undefined
  buildStatus: BuildStatusDetail | undefined
  describe: Build | undefined
  loading: boolean
  statusError?: Error | null
  showFocusNode?: boolean
  initialFocusNodeId?: string
}

function buildGraphData(
  buildStatus: BuildStatusDetail | undefined,
  plannedTargets: PlannedTarget[],
  isActive: boolean,
): {
  nodes: ElkNodeEx[]
  links: ElkExtendedEdge[]
  artifactIds: string[]
} {
  if (!buildStatus && !plannedTargets.length) return { nodes: [], links: [], artifactIds: [] }

  const nodes: ElkNodeEx[] = []
  const links: ElkExtendedEdge[] = []
  const seenArtifacts = new Set<string>()
  const seenEdges = new Set<string>()
  const seenTargets = new Set<string>()

  // ── Actual lineage from runtime status ────────────────────────────────────
  for (const [targetName, targetRun] of Object.entries(buildStatus?.targets ?? {})) {
    const targetId = `target-${targetName}`
    seenTargets.add(targetName)

    nodes.push({
      id: targetId,
      title: targetName,
      type: 'Build',
      width: 192,
      height: 64,
      labels: [{ text: targetName }],
    })

    for (const [paramName, artifactId] of Object.entries(targetRun.inputs ?? {})) {
      if (!artifactId) continue
      if (!seenArtifacts.has(artifactId)) {
        seenArtifacts.add(artifactId)
        nodes.push({ id: artifactId, title: paramName, type: 'Fileset', width: 224, height: 64, labels: [{ text: paramName }] })
      }
      const edgeId = `${artifactId}-to-${targetId}`
      if (!seenEdges.has(edgeId)) {
        seenEdges.add(edgeId)
        links.push({ id: edgeId, sources: [`${artifactId}-output`], targets: [`${targetId}-input`] })
      }
    }

    for (const [paramName, artifactId] of Object.entries(targetRun.outputs ?? {})) {
      if (!artifactId) continue
      if (!seenArtifacts.has(artifactId)) {
        seenArtifacts.add(artifactId)
        nodes.push({ id: artifactId, title: paramName, type: 'Fileset', width: 224, height: 64, labels: [{ text: paramName }] })
      }
      const edgeId = `${targetId}-to-${artifactId}`
      if (!seenEdges.has(edgeId)) {
        seenEdges.add(edgeId)
        links.push({ id: edgeId, sources: [`${targetId}-output`], targets: [`${artifactId}-input`] })
      }
    }
  }

  // ── Planned lineage overlay from build definition (active builds only) ────
  if (isActive && plannedTargets.length > 0) {
    for (const plannedTarget of plannedTargets) {
      const targetName = plannedTarget.target_name
      if (seenTargets.has(targetName)) continue  // target already in actual lineage

      const targetId = `target-${targetName}`
      seenTargets.add(targetName)

      nodes.push({
        id: targetId,
        title: targetName,
        type: 'Build',
        planned: true,
        width: 192,
        height: 64,
        labels: [{ text: targetName }],
      })

      for (const [paramName, artifactId] of Object.entries(plannedTarget.inputs ?? {})) {
        if (!artifactId) continue
        // If the input artifact already exists in actual lineage, just add the edge
        if (!seenArtifacts.has(artifactId)) {
          seenArtifacts.add(artifactId)
          nodes.push({ id: artifactId, title: paramName, type: 'Fileset', planned: true, width: 224, height: 64, labels: [{ text: paramName }] })
        }
        const edgeId = `${artifactId}-to-${targetId}`
        if (!seenEdges.has(edgeId)) {
          seenEdges.add(edgeId)
          links.push({ id: edgeId, sources: [`${artifactId}-output`], targets: [`${targetId}-input`] })
        }
      }

      for (const [paramName, artifactId] of Object.entries(plannedTarget.outputs ?? {})) {
        if (!artifactId) continue
        if (!seenArtifacts.has(artifactId)) {
          seenArtifacts.add(artifactId)
          nodes.push({ id: artifactId, title: paramName, type: 'Fileset', planned: true, width: 224, height: 64, labels: [{ text: paramName }] })
        }
        const edgeId = `${targetId}-to-${artifactId}`
        if (!seenEdges.has(edgeId)) {
          seenEdges.add(edgeId)
          links.push({ id: edgeId, sources: [`${targetId}-output`], targets: [`${artifactId}-input`] })
        }
      }
    }
  }

  return { nodes, links, artifactIds: Array.from(seenArtifacts) }
}

function isUUID(s: string) {
  return /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(s)
}

// Mirrors src/gbcommon/utils/hf_utils.py:convert_hf_uri_to_url — model URLs
// never include a "models/" segment; datasets/spaces/buckets keep their
// pluralized type segment.
function getHuggingFaceUrl(uri: string): string | null {
  if (!uri) return null

  if (uri.startsWith('hf://')) {
    const remainder = uri.slice(5)
    let parts: string[]

    if (remainder.startsWith('/')) {
      // hf:///[type/]org/name
      parts = remainder.replace(/^\/+/, '').split('/')
    } else if (remainder.startsWith('huggingface.co/')) {
      // hf://huggingface.co/[type/]org/name
      parts = remainder.slice('huggingface.co/'.length).split('/')
    } else if (remainder.includes('/')) {
      // hf://<domain>/[type/]org/name — the domain segment is discarded;
      // the browsable URL is always on huggingface.co
      parts = remainder.split('/').slice(1)
    } else {
      return null
    }

    if (parts.length === 2) {
      const [org, name] = parts
      return `https://huggingface.co/${org}/${name}`
    }
    if (parts.length === 3) {
      const [type, org, name] = parts
      switch (type) {
        case 'models':   return `https://huggingface.co/${org}/${name}`
        case 'datasets': return `https://huggingface.co/datasets/${org}/${name}`
        case 'spaces':   return `https://huggingface.co/spaces/${org}/${name}`
        case 'buckets':  return `https://huggingface.co/buckets/${org}/${name}`
        default: return null
      }
    }
    return null
  }

  if (/huggingface\.co/.test(uri)) return uri.startsWith('http') ? uri : `https://${uri}`
  return null
}

const LineagePanelInner = React.forwardRef<GraphHandle, LineagePanelProps>(function LineagePanelInner(
  { build, buildStatus, loading, statusError, showFocusNode = false, initialFocusNodeId },
  ref
) {
  const graphRef = React.useRef<GraphHandle>(null)
  const isActive = ACTIVE_STATUSES.has(build?.status ?? '')

  React.useImperativeHandle(ref, () => ({
    zoomIn: () => graphRef.current?.zoomIn(),
    zoomOut: () => graphRef.current?.zoomOut(),
    resetZoom: () => graphRef.current?.resetZoom(),
    currentZoom: () => graphRef.current?.currentZoom() ?? 90,
    centerOnNode: (nodeId: string) => graphRef.current?.centerOnNode(nodeId),
  }))

  // Fetch build archive YAML to derive planned targets for active builds
  const { data: archiveFiles } = useQuery({
    queryKey: ['build-archive', build?.uuid],
    queryFn: () => getBuildArchiveFiles(build!.uuid),
    enabled: Boolean(build?.uuid) && isActive,
    staleTime: 60_000,
  })

  const plannedTargets = React.useMemo<PlannedTarget[]>(() => {
    if (!archiveFiles) return []
    const yaml =
      archiveFiles['build.yaml'] ??
      archiveFiles[Object.keys(archiveFiles).find((k) => k.endsWith('.yaml') || k.endsWith('.yml')) ?? '']
    return yaml ? parseDefinitionTargets(yaml) : []
  }, [archiveFiles])

  const baseGraph = React.useMemo(
    () => buildGraphData(buildStatus, plannedTargets, isActive),
    [buildStatus, plannedTargets, isActive]
  )

  // ── Cross-build lineage expansion ───────────────────────────────────────────
  // The Upstream/Downstream buttons deepen a traversal that follows shared
  // artifact UUIDs into other builds. Depth 0 means "not expanded", which keeps
  // the query disabled so the initial render stays local-only.
  const [upstreamDepth, setUpstreamDepth] = React.useState(0)
  const [downstreamDepth, setDownstreamDepth] = React.useState(0)

  const expandDirection: LineageDirection | null =
    upstreamDepth > 0 && downstreamDepth > 0 ? 'both'
      : upstreamDepth > 0 ? 'upstream'
        : downstreamDepth > 0 ? 'downstream'
          : null
  // The endpoint takes a single max_depth for both walk directions, so asymmetric
  // depths can't be expressed. max() over-fetches in one direction but never
  // under-fetches, and the extra nodes are genuine lineage, so showing them is
  // correct. If asymmetry ever matters, two per-direction queries merge cleanly
  // (dedup is by node/edge id).
  const expandDepth = Math.max(upstreamDepth, downstreamDepth)

  const {
    data: expansion,
    isFetching: expanding,
    error: expandError,
  } = useQuery({
    queryKey: ['build-lineage', build?.uuid, expandDirection, expandDepth],
    queryFn: () => getBuildLineage(build!.uuid, expandDirection!, expandDepth),
    enabled: Boolean(build?.uuid) && expandDirection !== null,
    staleTime: 60_000,
    retry: false,
    // Keep the current graph on screen while a deeper level loads.
    placeholderData: (prev) => prev,
  })

  const expandedGraph = React.useMemo(() => {
    // expandDirection is the source of truth for "is the graph expanded":
    // placeholderData deliberately retains the previous level's data across key
    // changes, so Reset view has to be gated on the direction, not on `expansion`.
    if (!expansion || !expandDirection || !build?.uuid) return null
    const graph = jobstatsToGraph(expansion.targets, build.uuid)
    // buildGraphData keys this build's targets by name (planned targets have no
    // uuid at all), while the API keys them by uuid. Alias the API's own-build
    // target nodes onto the local form so the two copies collapse into one.
    const rename = new Map<string, string>()
    for (const node of graph.nodes) {
      if (node.type !== 'Build' || node.foreignBuild) continue
      const name = graph.targetNamesById.get(node.id)
      if (name) rename.set(node.id, `target-${name}`)
    }
    const { nodes, links } = renameNodes(graph.nodes, graph.links, rename)
    return { nodes, links, artifactIds: graph.artifactIds }
  }, [expansion, expandDirection, build?.uuid])

  const { nodes: allNodes, links: allLinks, artifactIds } = React.useMemo(
    () => (expandedGraph ? mergeGraphs(baseGraph, expandedGraph) : baseGraph),
    [baseGraph, expandedGraph]
  )

  // Fetch artifact names for all UUID-shaped artifact IDs
  const uuidArtifactIds = artifactIds.filter(isUUID)
  const artifactQueries = useQueries({
    queries: uuidArtifactIds.map((id) => ({
      queryKey: ['artifact', id],
      queryFn: () => getArtifact(id),
      retry: false,
      staleTime: 5 * 60 * 1000,
    })),
  })

  // Enrich nodes with resolved artifact names and types
  const enrichedNodes = React.useMemo<ElkNodeEx[]>(() => {
    const artifactMap = new Map<string, { name: string; type: NodeType }>()
    uuidArtifactIds.forEach((id, i) => {
      const result = artifactQueries[i]?.data
      if (result) {
        artifactMap.set(id, {
          name: result.name,
          type: artifactTypeToNodeType(result.artifact_type),
        })
      }
    })

    return allNodes.map((node) => {
      const enriched = artifactMap.get(node.id)
      if (enriched) {
        // Expanded nodes already carry a resolved name/type from the lineage
        // API, so fall back to those if this artifact's lookup came back empty.
        return {
          ...node,
          title: enriched.name || node.title,
          type: enriched.type ?? node.type,
        }
      }
      return node
    })
  }, [allNodes, artifactQueries, uuidArtifactIds])

  const artifactUriMap = React.useMemo(() => {
    const map = new Map<string, string>()
    uuidArtifactIds.forEach((id, i) => {
      const uri = artifactQueries[i]?.data?.uri
      if (uri) map.set(id, uri)
    })
    return map
  }, [artifactQueries, uuidArtifactIds])

  const artifactNavModalHeader = (artifactNavNode: { node: ElkNodeEx; hfUrl: string | null } | null) => {
    if (artifactNavNode) {
      return <h4>Would you like to view <code>{artifactNavNode.node?.title || artifactNavNode.node?.id}</code> on HuggingFace or proceed to the artifact page?`</h4>
    } else {
      return <h4>Would you like to view this artifact on HuggingFace or proceed to the artifact page?`</h4>
    }
  }

  // Navigation state
  const [focusNodeId, setFocusNodeId] = React.useState<string | null>(initialFocusNodeId ?? null)
  const [partial, setPartial] = React.useState(false)
  const [artifactNavNode, setArtifactNavNode] = React.useState<{ node: ElkNodeEx; hfUrl: string | null } | null>(null)
  const router = useRouter()
  const [rendered, setRendered] = React.useState(false)

  // The current artifact's node is always highlighted on artifact pages
  // (showFocusNode is only true there) — this is not click-driven.
  const currentArtifactNode = React.useMemo(
    () => (showFocusNode && initialFocusNodeId
      ? enrichedNodes.find((n) => n.id === initialFocusNodeId)
      : undefined),
    [showFocusNode, initialFocusNodeId, enrichedNodes]
  )

  const handleNodeClick = (node: ElkNodeEx) => {
    if (!showFocusNode) {
      setFocusNodeId(node.id)
    }
    if (node.type !== 'Build' && isUUID(node.id)) {
      const uri = artifactUriMap.get(node.id)
      setArtifactNavNode({ node, hfUrl: uri ? getHuggingFaceUrl(uri) : null })
    }
  }

  // Purely a viewport action: it re-centers without discarding lineage already
  // fetched, so the partial banner stays as accurate as it was.
  const handleFocusNode = () => {
    if (!focusNodeId) return
    graphRef.current?.centerOnNode?.(focusNodeId)
  }

  // Each click pulls in one more hop of lineage from other builds. Clamped to 50,
  // which is the endpoint's ceiling (it rejects anything higher with a 400).
  const handleUpstream = () => setUpstreamDepth((depth) => Math.min(depth + 1, 50))
  const handleDownstream = () => setDownstreamDepth((depth) => Math.min(depth + 1, 50))

  // The banner now reports whether the *backend* has more lineage to give.
  React.useEffect(() => {
    if (!expansion || !expandDirection) return
    setPartial(expansion.truncated || expansion.expandable.length > 0)
  }, [expansion, expandDirection])

  const noLineage = !loading && !statusError && allNodes.length === 0

  return (
    <div className={styles.container}>
      {/* Toolbar */}
      <div className={styles.toolbar}>
        <div className={styles.toolbarLeft}>
          <Button
            size="sm"
            kind="ghost"
            renderIcon={ArrowLeft}
            disabled={!build?.uuid || expanding}
            onClick={handleUpstream}
            iconDescription="One level upstream"
          >
            Upstream
          </Button>
          {showFocusNode && (
            <Button
              size="sm"
              kind="ghost"
              renderIcon={CenterSquare}
              disabled={!focusNodeId}
              onClick={handleFocusNode}
            >
              Focus Node
            </Button>
          )}
          <Button
            size="sm"
            kind="ghost"
            renderIcon={ArrowRight}
            disabled={!build?.uuid || expanding}
            onClick={handleDownstream}
            iconDescription="One level downstream"
          >
            Downstream
          </Button>

          <div className={styles.toolbarDivider} />

          <Button
            size="sm"
            kind="ghost"
            hasIconOnly
            tooltipPosition="right"
            iconDescription="Zoom In (+10%)"
            renderIcon={ZoomIn}
            onClick={() => graphRef.current?.zoomIn()}
          />
          <Button
            size="sm"
            kind="ghost"
            hasIconOnly
            tooltipPosition="right"
            iconDescription="Reset Zoom"
            renderIcon={ZoomFit}
            onClick={() => graphRef.current?.resetZoom()}
          />
          <Button
            size="sm"
            kind="ghost"
            hasIconOnly
            tooltipPosition="right"
            iconDescription="Zoom Out (-10%)"
            renderIcon={ZoomOut}
            onClick={() => graphRef.current?.zoomOut()}
          />

          <div className={styles.toolbarDivider} />

          <OverflowMenu size="sm" selectorPrimaryFocus=".overflow-item">
            <OverflowMenuItem
              className="overflow-item"
              itemText="Reset view"
              onClick={() => {
                setFocusNodeId(null);
                // Dropping the depths to 0 disables the expansion query, so the
                // graph falls back to this build's own lineage.
                setUpstreamDepth(0);
                setDownstreamDepth(0);
                setPartial(false);
                graphRef.current?.resetZoom();
              }}
            />
          </OverflowMenu>

          <div className={styles.toolbarDivider} />
        </div>
      </div>

      {/* Status messages */}
      {expanding && (
        <div className={styles.partialMessage}>
          <InlineLoading description="Expanding lineage…" status="active" />
        </div>
      )}

      {/* Kept out of the main error branch on purpose: a failed expansion must
          not discard the lineage already on screen. */}
      {!expanding && expandError && (
        <div className={styles.partialMessage}>
          Could not expand lineage: {(expandError as Error).message || String(expandError)}
        </div>
      )}

      {partial && (
        <div className={styles.partialMessage}>
          The lineage graph is partially displayed. Click Upstream or Downstream
          to show more nodes.
        </div>
      )}

      {/* Graph area */}
      <div className={styles.graphArea}>
        {loading && (
          <div className={styles.centeredContent}>
            <InlineLoading description="Loading lineage…" />
          </div>
        )}

        {!loading && statusError && (
          <div className={styles.errorContent}>
            Failed to load lineage: {String(statusError)}
          </div>
        )}

        {!loading && noLineage && (
          <div className={styles.emptyContent}>
            No lineage data available for build
            {build?.name ? ` "${build.name}"` : ""}.
          </div>
        )}

        {!loading && !noLineage && (
          <>
            {!rendered && (
              <InlineLoading
                className={styles.renderingIndicator}
                description="Lineage is rendering…"
              />
            )}
            {/* Depth is enforced by the backend traversal, so everything it
                returns is meant to be shown — no client-side level filtering.
                `allLinks` must be this same merged set: Graph derives its "…"
                stub nodes by comparing it against `links`. */}
            <Graph
              ref={graphRef}
              nodes={enrichedNodes}
              links={allLinks}
              allLinks={allLinks}
              selectedNode={currentArtifactNode}
              onClick={handleNodeClick}
              onSvgRendered={() => setRendered(true)}
            />
          </>
        )}
      </div>

      {artifactNavNode?.hfUrl ? (
        <ComposedModal
          open={artifactNavNode !== null}
          onClose={() => setArtifactNavNode(null)}
          size="sm"
        >
          <ModalHeader>{artifactNavModalHeader(artifactNavNode)}</ModalHeader>
          <ModalBody />
          <ModalFooter className={styles.navModalActions}>
            <Button
              kind="secondary"
              onClick={() => {
                setArtifactNavNode(null);
              }}
            >
              Cancel
            </Button>
            <Button
              kind="secondary"
              onClick={() => {
                if (artifactNavNode)
                  router.push(`/dashboard/artifacts/_/?id=${artifactNavNode.node.id}`);
                setArtifactNavNode(null);
              }}
            >
              View artifact page
            </Button>

            <Button
              kind="secondary"
              renderIcon={Launch}
              href={artifactNavNode.hfUrl}
              target="_blank"
              rel="noopener noreferrer"
              onClick={() => setArtifactNavNode(null)}
            >
              Open on HuggingFace
            </Button>
          </ModalFooter>
        </ComposedModal>
      ) : (
        <Modal
          open={artifactNavNode !== null}
          onRequestClose={() => setArtifactNavNode(null)}
          modalHeading="Navigate to artifact"
          primaryButtonText="Proceed"
          secondaryButtonText="Cancel"
          onRequestSubmit={() => {
            if (artifactNavNode)
              router.push(`/dashboard/artifacts/_/?id=${artifactNavNode.node.id}`);
            setArtifactNavNode(null);
          }}
          onSecondarySubmit={() => setArtifactNavNode(null)}
          size="sm"
        >
          <p>
            Go to the artifact page for{" "}
            <strong>
              {artifactNavNode?.node.title || artifactNavNode?.node.id}
            </strong>
            ?
          </p>
        </Modal>
      )}
    </div>
  );
})

export default LineagePanelInner

// Re-export GraphHandle for use in parent
export type { GraphHandle }
