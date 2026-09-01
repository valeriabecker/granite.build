import type { ElkExtendedEdge } from 'elkjs'
import type { ElkNodeEx, NodeType } from './Graph'
import type { JobStatsArtifactEntry, JobStatsEvent, TargetJobStats } from '@/api/gbserver'

/**
 * Converts gbserver JobStats (from GET /lineage/build/{id}?direction=…) into the
 * node/link shape the lineage Graph renders.
 *
 * The ids produced here must match buildGraphData() in
 * app/dashboard/builds/[buildId]/LineagePanel.tsx exactly — artifact nodes keyed
 * by artifact UUID, ports suffixed -output/-input, edges `${from}-to-${this}` —
 * otherwise merged local and remote subgraphs won't connect or dedupe.
 *
 * Kept free of React/Carbon imports (types only) so it is unit-testable under
 * the plain `node --test` runner.
 */

// Match buildGraphData's node dimensions so ELK lays merged graphs out evenly.
export const TARGET_NODE_WIDTH = 192
export const TARGET_NODE_HEIGHT = 64
export const ARTIFACT_NODE_WIDTH = 224
export const ARTIFACT_NODE_HEIGHT = 64

export function artifactTypeToNodeType(artifactType?: string): NodeType {
  switch ((artifactType ?? '').toUpperCase()) {
    case 'MODEL': return 'Model'
    case 'DATASET': return 'Dataset'
    case 'FILESET': return 'Fileset'
    default: return 'Fileset'
  }
}

/** Node id for a target-run. Keyed by uuid: target names collide across builds. */
export function targetNodeId(targetUuid: string): string {
  return `target-${targetUuid}`
}

export interface LineageGraphData {
  nodes: ElkNodeEx[]
  links: ElkExtendedEdge[]
  artifactIds: string[]
}

export interface JobStatsGraphResult extends LineageGraphData {
  /** Target node id -> target name, for aliasing onto local `target-<name>` ids. */
  targetNamesById: Map<string, string>
}

function stripPort(port: string): string {
  return port.replace(/-(output|input)$/, '')
}

export function jobstatsToGraph(
  targets: TargetJobStats[],
  rootBuildId: string,
): JobStatsGraphResult {
  const nodes: ElkNodeEx[] = []
  const links: ElkExtendedEdge[] = []
  const seenNodes = new Set<string>()
  const seenEdges = new Set<string>()
  const artifactIds = new Set<string>()
  const targetNamesById = new Map<string, string>()

  const addEdge = (from: string, to: string) => {
    const id = `${from}-to-${to}`
    if (seenEdges.has(id)) return
    seenEdges.add(id)
    links.push({ id, sources: [`${from}-output`], targets: [`${to}-input`] })
  }

  const addArtifact = (entry: JobStatsArtifactEntry, foreign: boolean, buildId: string) => {
    const artifactId = entry?.facets?.artifact_id
    if (!artifactId) return null
    artifactIds.add(artifactId)
    if (!seenNodes.has(artifactId)) {
      seenNodes.add(artifactId)
      const title = entry.name || artifactId
      nodes.push({
        id: artifactId,
        title,
        type: artifactTypeToNodeType(entry.facets?.artifact_type),
        width: ARTIFACT_NODE_WIDTH,
        height: ARTIFACT_NODE_HEIGHT,
        labels: [{ text: title }],
        ...(foreign ? { foreignBuild: true, buildId } : {}),
      })
    }
    return artifactId
  }

  for (const targetStats of targets ?? []) {
    for (const events of Object.values(targetStats ?? {})) {
      for (const event of (events ?? []) as JobStatsEvent[]) {
        const tags = event?.run?.facets?.tags
        const details = event?.run?.facets?.job_details
        // Prefer job_details.job_id: run.runId is suffixed with the output
        // artifact uuid on per-output events, so it is not a stable target id.
        const targetUuid = details?.job_id || tags?.target_id
        if (!targetUuid) continue

        const buildId = tags?.build_id || details?.release_id || ''
        const foreign = Boolean(buildId) && buildId !== rootBuildId
        const targetName = event.job?.name || targetUuid
        const nodeId = targetNodeId(targetUuid)

        targetNamesById.set(nodeId, targetName)

        if (!seenNodes.has(nodeId)) {
          seenNodes.add(nodeId)
          nodes.push({
            id: nodeId,
            title: targetName,
            type: 'Build',
            width: TARGET_NODE_WIDTH,
            height: TARGET_NODE_HEIGHT,
            labels: [{ text: targetName }],
            // job.namespace is "<space>/<build name>" — shows the owning build.
            ...(foreign ? { foreignBuild: true, buildId, subtitle: event.job?.namespace } : {}),
          })
        }

        // Only inputs/outputs: event.sources and event.targets are mirrors of
        // them (jobstats_builder._add_jobstats_mirror_fields).
        for (const input of event.inputs ?? []) {
          const artifactId = addArtifact(input, foreign, buildId)
          if (artifactId) addEdge(artifactId, nodeId)
        }
        for (const output of event.outputs ?? []) {
          const artifactId = addArtifact(output, foreign, buildId)
          if (artifactId) addEdge(nodeId, artifactId)
        }
      }
    }
  }

  return { nodes, links, artifactIds: [...artifactIds], targetNamesById }
}

/**
 * Rewrites node ids through `rename`, along with every edge port referencing
 * them. Edge ids are recomputed from the renamed endpoints so they come out
 * byte-identical to buildGraphData's — that is what lets a renamed remote edge
 * dedupe against its local twin.
 */
export function renameNodes(
  nodes: ElkNodeEx[],
  links: ElkExtendedEdge[],
  rename: Map<string, string>,
): { nodes: ElkNodeEx[]; links: ElkExtendedEdge[] } {
  if (rename.size === 0) return { nodes, links }

  const map = (id: string) => rename.get(id) ?? id
  const mapPort = (port: string) => {
    if (port.endsWith('-output')) return `${map(port.slice(0, -'-output'.length))}-output`
    if (port.endsWith('-input')) return `${map(port.slice(0, -'-input'.length))}-input`
    return map(port)
  }

  return {
    nodes: nodes.map((node) => (rename.has(node.id) ? { ...node, id: map(node.id) } : node)),
    links: links.map((link) => {
      const sources = link.sources.map(mapPort)
      const targets = link.targets.map(mapPort)
      return { ...link, id: `${stripPort(sources[0])}-to-${stripPort(targets[0])}`, sources, targets }
    }),
  }
}

/**
 * Merges an API-derived subgraph into the locally-derived one, deduping by id.
 *
 * API nodes win on conflict: they carry the artifact's real name and type, while
 * the local graph only knows the target's parameter name and assumes 'Fileset'.
 * `planned` is the exception — only the local graph knows a target hasn't run
 * yet, so that flag is carried over.
 */
export function mergeGraphs(base: LineageGraphData, expanded: LineageGraphData): LineageGraphData {
  const nodeById = new Map<string, ElkNodeEx>()
  for (const node of base.nodes) nodeById.set(node.id, node)
  for (const node of expanded.nodes) {
    const prev = nodeById.get(node.id)
    nodeById.set(node.id, prev?.planned ? { ...node, planned: true } : node)
  }

  const linkById = new Map<string, ElkExtendedEdge>()
  for (const link of [...base.links, ...expanded.links]) linkById.set(link.id, link)

  return {
    nodes: [...nodeById.values()],
    links: [...linkById.values()],
    artifactIds: [...new Set([...base.artifactIds, ...expanded.artifactIds])],
  }
}
