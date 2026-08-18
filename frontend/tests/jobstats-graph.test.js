/**
 * Unit tests for the JobStats -> lineage graph converter.
 *
 * The module under test is TypeScript, but it imports only *types* from the app
 * (which `--experimental-strip-types` erases), so node can load it directly with
 * no bundler or path-alias resolution.
 *
 * Usage: node --test --experimental-strip-types tests/jobstats-graph.test.js
 */

const { describe, it } = require('node:test')
const assert = require('node:assert/strict')

const MODULE = '../components/LineageGraph/jobstatsGraph.ts'

const ROOT_BUILD = 'build-root'
const OTHER_BUILD = 'build-other'

/** Builds one OpenLineage-shaped jobstats event, mirroring the backend's output. */
function event({
  targetUuid = 'target-uuid-1',
  targetName = 'train',
  buildId = ROOT_BUILD,
  namespace = 'myspace/my-build',
  inputs = [],
  outputs = [],
  jobId = undefined,
  omitJobId = false,
  runIdSuffix = '',
} = {}) {
  const jobDetails = { job_status: 'SUCCESS', release_id: buildId }
  if (!omitJobId) jobDetails.job_id = jobId ?? targetUuid
  return {
    run: {
      runId: `${targetUuid}${runIdSuffix}`,
      facets: {
        tags: { build_id: buildId, target_id: targetUuid, space_name: 'myspace' },
        job_details: jobDetails,
      },
    },
    job: { namespace, name: targetName },
    inputs,
    outputs,
  }
}

function artifact(uuid, name, type = 'FILESET') {
  return { namespace: 'ns', name, facets: { artifact_id: uuid, artifact_type: type, artifact_uri: `gb://${uuid}` } }
}

/** Wraps events into the { <artifact name>: events[] } per-target shape. */
function target(events, key = 'out') {
  return { [key]: events }
}

const byId = (nodes) => new Map(nodes.map((n) => [n.id, n]))
const ids = (nodes) => nodes.map((n) => n.id).sort()

describe('jobstatsToGraph', () => {
  it('converts one target with an input and an output into 3 nodes and 2 edges', async () => {
    const { jobstatsToGraph } = await import(MODULE)
    const g = jobstatsToGraph(
      [target([event({ inputs: [artifact('in-uuid', 'input-dataset')], outputs: [artifact('out-uuid', 'output-dataset')] })])],
      ROOT_BUILD,
    )

    assert.deepEqual(ids(g.nodes), ['in-uuid', 'out-uuid', 'target-target-uuid-1'].sort())
    assert.deepEqual(g.artifactIds.sort(), ['in-uuid', 'out-uuid'])

    // Edge ids and ports must match buildGraphData's conventions exactly.
    const edges = new Map(g.links.map((l) => [l.id, l]))
    assert.deepEqual([...edges.keys()].sort(), [
      'in-uuid-to-target-target-uuid-1',
      'target-target-uuid-1-to-out-uuid',
    ])
    assert.deepEqual(edges.get('in-uuid-to-target-target-uuid-1').sources, ['in-uuid-output'])
    assert.deepEqual(edges.get('in-uuid-to-target-target-uuid-1').targets, ['target-target-uuid-1-input'])
    assert.deepEqual(edges.get('target-target-uuid-1-to-out-uuid').sources, ['target-target-uuid-1-output'])
    assert.deepEqual(edges.get('target-target-uuid-1-to-out-uuid').targets, ['out-uuid-input'])

    const targetNode = byId(g.nodes).get('target-target-uuid-1')
    assert.equal(targetNode.type, 'Build')
    assert.equal(targetNode.title, 'train')
    assert.deepEqual(targetNode.labels, [{ text: 'train' }])
  })

  it('collapses the per-output events of one target onto a single Build node', async () => {
    const { jobstatsToGraph } = await import(MODULE)
    // The backend emits one event per output artifact, each with runId
    // `${target_uuid}-${output_uuid}` but the same job_details.job_id.
    const g = jobstatsToGraph(
      [{
        'out-a': [event({ outputs: [artifact('out-a', 'a')], runIdSuffix: '-out-a' })],
        'out-b': [event({ outputs: [artifact('out-b', 'b')], runIdSuffix: '-out-b' })],
      }],
      ROOT_BUILD,
    )

    const buildNodes = g.nodes.filter((n) => n.type === 'Build')
    assert.equal(buildNodes.length, 1, 'per-output events must not create duplicate target nodes')
    assert.equal(buildNodes[0].id, 'target-target-uuid-1')
    assert.equal(g.links.length, 2)
  })

  it('ignores the sources/targets mirror fields so inputs are not double-counted', async () => {
    const { jobstatsToGraph } = await import(MODULE)
    const input = artifact('in-uuid', 'input-dataset')
    const output = artifact('out-uuid', 'output-dataset')
    const e = event({ inputs: [input], outputs: [output] })
    // _add_jobstats_mirror_fields duplicates inputs->sources, outputs->targets.
    e.sources = [input]
    e.targets = [output]

    const g = jobstatsToGraph([target([e])], ROOT_BUILD)
    assert.equal(g.nodes.filter((n) => n.id === 'in-uuid').length, 1)
    assert.equal(g.links.length, 2)
  })

  it('keeps same-named targets from different builds as distinct nodes', async () => {
    const { jobstatsToGraph } = await import(MODULE)
    const g = jobstatsToGraph(
      [
        target([event({ targetUuid: 'uuid-a', targetName: 'train', buildId: ROOT_BUILD })], 'a'),
        target([event({ targetUuid: 'uuid-b', targetName: 'train', buildId: OTHER_BUILD })], 'b'),
      ],
      ROOT_BUILD,
    )
    assert.deepEqual(ids(g.nodes.filter((n) => n.type === 'Build')), ['target-uuid-a', 'target-uuid-b'])
  })

  it('marks nodes from other builds and leaves root-build nodes unmarked', async () => {
    const { jobstatsToGraph } = await import(MODULE)
    const g = jobstatsToGraph(
      [
        target([event({ targetUuid: 'uuid-local', buildId: ROOT_BUILD, outputs: [artifact('local-art', 'local')] })], 'a'),
        target([event({
          targetUuid: 'uuid-remote',
          buildId: OTHER_BUILD,
          namespace: 'otherspace/other-build',
          outputs: [artifact('remote-art', 'remote')],
        })], 'b'),
      ],
      ROOT_BUILD,
    )
    const nodes = byId(g.nodes)

    const local = nodes.get('target-uuid-local')
    assert.equal(local.foreignBuild, undefined)
    assert.equal(local.subtitle, undefined)
    assert.equal(nodes.get('local-art').foreignBuild, undefined)

    const remote = nodes.get('target-uuid-remote')
    assert.equal(remote.foreignBuild, true)
    assert.equal(remote.buildId, OTHER_BUILD)
    assert.equal(remote.subtitle, 'otherspace/other-build')
    assert.equal(nodes.get('remote-art').foreignBuild, true)
    assert.equal(nodes.get('remote-art').buildId, OTHER_BUILD)
  })

  it('maps artifact types, defaulting unknown and missing values to Fileset', async () => {
    const { jobstatsToGraph, artifactTypeToNodeType } = await import(MODULE)
    assert.equal(artifactTypeToNodeType('MODEL'), 'Model')
    assert.equal(artifactTypeToNodeType('DATASET'), 'Dataset')
    assert.equal(artifactTypeToNodeType('FILESET'), 'Fileset')
    assert.equal(artifactTypeToNodeType('model'), 'Model', 'must be case-insensitive')
    assert.equal(artifactTypeToNodeType('TABLE'), 'Fileset')
    assert.equal(artifactTypeToNodeType(undefined), 'Fileset')

    const g = jobstatsToGraph(
      [target([event({ outputs: [artifact('m-uuid', 'a-model', 'MODEL')] })])],
      ROOT_BUILD,
    )
    assert.equal(byId(g.nodes).get('m-uuid').type, 'Model')
  })

  it('skips artifact entries with no artifact_id without creating phantom nodes', async () => {
    const { jobstatsToGraph } = await import(MODULE)
    const g = jobstatsToGraph(
      [target([event({
        inputs: [{ namespace: 'ns', name: 'nameless', facets: {} }, artifact('good', 'good')],
        outputs: [{ name: 'no-facets-at-all' }],
      })])],
      ROOT_BUILD,
    )
    assert.deepEqual(ids(g.nodes), ['good', 'target-target-uuid-1'].sort())
    assert.deepEqual(g.artifactIds, ['good'])
    assert.equal(g.links.length, 1)
  })

  it('falls back to tags.target_id, and skips the event when no target id exists', async () => {
    const { jobstatsToGraph } = await import(MODULE)

    const fallback = jobstatsToGraph(
      [target([event({ targetUuid: 'uuid-from-tags', omitJobId: true })])],
      ROOT_BUILD,
    )
    assert.deepEqual(ids(fallback.nodes), ['target-uuid-from-tags'])

    const orphan = event({ omitJobId: true })
    delete orphan.run.facets.tags.target_id
    const skipped = jobstatsToGraph([target([orphan])], ROOT_BUILD)
    assert.deepEqual(skipped.nodes, [])
    assert.deepEqual(skipped.links, [])
  })

  it('tolerates empty, null and malformed payloads', async () => {
    const { jobstatsToGraph } = await import(MODULE)
    for (const payload of [[], [{}], [null], [target([])], [target([null])]]) {
      const g = jobstatsToGraph(payload, ROOT_BUILD)
      assert.deepEqual(g.nodes, [])
      assert.deepEqual(g.links, [])
    }
  })
})

describe('renameNodes', () => {
  it('rewrites node ids, ports, and recomputes edge ids so they dedupe with local ids', async () => {
    const { jobstatsToGraph, renameNodes } = await import(MODULE)
    const g = jobstatsToGraph(
      [target([event({ inputs: [artifact('in-uuid', 'in')], outputs: [artifact('out-uuid', 'out')] })])],
      ROOT_BUILD,
    )

    const renamed = renameNodes(g.nodes, g.links, new Map([['target-target-uuid-1', 'target-train']]))

    assert.ok(byId(renamed.nodes).has('target-train'))
    assert.ok(!byId(renamed.nodes).has('target-target-uuid-1'))

    const edges = new Map(renamed.links.map((l) => [l.id, l]))
    // These ids are exactly what buildGraphData produces for the local graph.
    assert.deepEqual([...edges.keys()].sort(), ['in-uuid-to-target-train', 'target-train-to-out-uuid'])
    assert.deepEqual(edges.get('in-uuid-to-target-train').targets, ['target-train-input'])
    assert.deepEqual(edges.get('target-train-to-out-uuid').sources, ['target-train-output'])
    // Untouched endpoints keep their ids.
    assert.deepEqual(edges.get('in-uuid-to-target-train').sources, ['in-uuid-output'])
  })

  it('is identity for an empty rename map, preserving object references', async () => {
    const { jobstatsToGraph, renameNodes } = await import(MODULE)
    const g = jobstatsToGraph([target([event({ outputs: [artifact('out-uuid', 'out')] })])], ROOT_BUILD)
    const renamed = renameNodes(g.nodes, g.links, new Map())
    // Same references => no id churn => no needless ELK relayout.
    assert.equal(renamed.nodes, g.nodes)
    assert.equal(renamed.links, g.links)
  })
})

describe('mergeGraphs', () => {
  it('dedupes by id, prefers API nodes, and preserves the local planned flag', async () => {
    const { mergeGraphs } = await import(MODULE)

    const base = {
      // Local graph: parameter name as title, type always assumed Fileset.
      nodes: [
        { id: 'art-1', title: 'training_data', type: 'Fileset' },
        { id: 'target-train', title: 'train', type: 'Build', planned: true },
        { id: 'local-only', title: 'local', type: 'Fileset' },
      ],
      links: [{ id: 'art-1-to-target-train', sources: ['art-1-output'], targets: ['target-train-input'] }],
      artifactIds: ['art-1', 'local-only'],
    }
    const expanded = {
      // API graph: real artifact name and resolved type.
      nodes: [
        { id: 'art-1', title: 'squad-v2', type: 'Dataset' },
        { id: 'target-train', title: 'train', type: 'Build' },
        { id: 'remote-art', title: 'remote', type: 'Model', foreignBuild: true },
      ],
      links: [
        { id: 'art-1-to-target-train', sources: ['art-1-output'], targets: ['target-train-input'] },
        { id: 'target-train-to-remote-art', sources: ['target-train-output'], targets: ['remote-art-input'] },
      ],
      artifactIds: ['art-1', 'remote-art'],
    }

    const merged = mergeGraphs(base, expanded)
    const nodes = byId(merged.nodes)

    assert.equal(merged.nodes.length, 4)
    assert.equal(nodes.get('art-1').title, 'squad-v2', 'API node should win — it has the real name')
    assert.equal(nodes.get('art-1').type, 'Dataset')
    assert.equal(nodes.get('target-train').planned, true, 'planned is only known locally and must survive')
    assert.ok(nodes.has('local-only'), 'planned-overlay nodes absent from the API must be kept')
    assert.ok(nodes.has('remote-art'))

    assert.equal(merged.links.length, 2, 'identical edge ids must collapse')
    assert.deepEqual(merged.artifactIds.sort(), ['art-1', 'local-only', 'remote-art'])
  })

  it('returns the base unchanged when the expansion is empty', async () => {
    const { mergeGraphs } = await import(MODULE)
    const base = {
      nodes: [{ id: 'a', title: 'a', type: 'Fileset' }],
      links: [],
      artifactIds: ['a'],
    }
    const merged = mergeGraphs(base, { nodes: [], links: [], artifactIds: [] })
    assert.deepEqual(ids(merged.nodes), ['a'])
    assert.deepEqual(merged.artifactIds, ['a'])
  })
})
