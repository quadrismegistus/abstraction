<script lang="ts">
  import { onMount } from 'svelte';
  import { goto } from '$app/navigation';
  import { norm, chunkSize } from '$lib/stores';
  import { fetchTrajectory } from '$lib/api';
  import type { TrajectoryResponse } from '$lib/types';

  let { corpus, textId }: { corpus: string; textId: string } = $props();

  let plotDiv: HTMLDivElement;
  let Plotly: any;
  let data: TrajectoryResponse | null = $state(null);
  let loading = $state(true);
  let error = $state('');

  async function loadData() {
    loading = true;
    error = '';
    try {
      data = await fetchTrajectory(corpus, textId, {
        col: $norm,
        chunk_size: String($chunkSize),
      });
    } catch (e: any) {
      error = e.message;
    }
    loading = false;
    renderPlot();
  }

  function renderPlot() {
    if (!Plotly || !plotDiv || !data) return;

    const chunks = data.chunks;
    const x = chunks.map(c => c.index);
    const y = chunks.map(c => c.score);

    const traces: any[] = [
      {
        x, y,
        type: 'scatter',
        mode: 'lines+markers',
        line: { color: '#2196F3', width: 2 },
        marker: { size: 6, color: y.map(s =>
          s === null ? '#999' : s < 0 ? '#1565C0' : '#E65100'
        )},
        hovertemplate: 'Passage %{x}<br>Score: %{y:.3f}<extra></extra>',
      },
    ];

    // Overall score reference line
    if (data.overall_score !== null) {
      traces.push({
        x: [x[0], x[x.length - 1]],
        y: [data.overall_score, data.overall_score],
        type: 'scatter',
        mode: 'lines',
        line: { color: '#999', dash: 'dash', width: 1 },
        name: `Overall: ${data.overall_score.toFixed(3)}`,
        hoverinfo: 'skip',
      });
    }

    const title = [
      data.metadata.title,
      data.metadata.author,
      data.metadata.year ? `(${data.metadata.year})` : '',
    ].filter(Boolean).join(' — ');

    Plotly.react(plotDiv, traces, {
      title: { text: title || `${corpus}/${textId}`, font: { size: 14 } },
      xaxis: { title: `Passage (${$chunkSize}-word chunks)` },
      yaxis: { title: 'Abstractness score', zeroline: true },
      margin: { t: 40, r: 20, b: 50, l: 60 },
      hovermode: 'closest',
    }, { responsive: true });

    plotDiv.removeAllListeners?.('plotly_click');
    plotDiv.on('plotly_click', (d: any) => {
      const idx = d.points[0]?.x;
      if (idx !== undefined) {
        goto(`/passage/${corpus}/${textId}?chunk=${idx}&chunk_size=${$chunkSize}`);
      }
    });
  }

  onMount(async () => {
    Plotly = await import('plotly.js-dist-min');
    loadData();
  });

  $effect(() => {
    $norm; $chunkSize;
    if (Plotly) loadData();
  });
</script>

<div class="trajectory-container">
  {#if loading}
    <div class="loading">Computing trajectory...</div>
  {:else if error}
    <div class="error">Error: {error}</div>
  {/if}

  <div class="controls">
    <label>
      Chunk size:
      <select onchange={(e) => $chunkSize = +(e.target as HTMLSelectElement).value}>
        {#each [100, 250, 500, 1000, 2000] as size}
          <option value={size} selected={size === $chunkSize}>{size} words</option>
        {/each}
      </select>
    </label>
  </div>

  <div bind:this={plotDiv} class="plot"></div>
</div>

<style>
  .trajectory-container { flex: 1; display: flex; flex-direction: column; }
  .plot { flex: 1; min-height: 400px; }
  .controls { padding: 0.5rem 1rem; display: flex; gap: 1rem; align-items: center; }
  .controls label { font-size: 0.85rem; display: flex; align-items: center; gap: 0.5rem; }
  .controls select { padding: 2px 4px; }
  .loading, .error { padding: 1rem; text-align: center; }
  .loading { color: #666; font-style: italic; }
  .error { color: #d32f2f; }
</style>
