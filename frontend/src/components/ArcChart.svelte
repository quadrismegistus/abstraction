<script lang="ts">
  import { onMount } from 'svelte';
  import { goto } from '$app/navigation';
  import { norm, selectedGenres, selectedCorpora, yearRange } from '$lib/stores';
  import { fetchArcByCorpus, fetchArcAggregated } from '$lib/api';
  import type { CorpusArc, ArcBin } from '$lib/types';

  let plotDiv: HTMLDivElement;
  let Plotly: any;
  let corpora: CorpusArc[] = $state([]);
  let overallBins: ArcBin[] = $state([]);
  let loading = $state(true);
  let totalTexts = $state(0);

  const genreColors: Record<string, string> = {
    'Fiction': '#2196F3',
    'Poetry': '#4CAF50',
    'Drama': '#FF9800',
    'Periodical': '#9C27B0',
    'Essay/Treatise': '#795548',
    'Letters': '#00BCD4',
    'Legal': '#607D8B',
    'Political': '#F44336',
    'Sermon': '#FF5722',
    'Biography': '#3F51B5',
    'Nonfiction': '#8BC34A',
    'Criticism': '#E91E63',
  };

  function getParams() {
    const p: Record<string, string | string[]> = {};
    p.col = $norm;
    if ($selectedGenres.length) p.genre = $selectedGenres;
    if ($selectedCorpora.length) p.corpus = $selectedCorpora;
    if ($yearRange[0] > 1000) p.year_min = String($yearRange[0]);
    if ($yearRange[1] < 2025) p.year_max = String($yearRange[1]);
    return p;
  }

  async function loadData() {
    loading = true;
    try {
      const [corpusData, aggData] = await Promise.all([
        fetchArcByCorpus(getParams()),
        fetchArcAggregated(getParams()),
      ]);
      corpora = corpusData;
      overallBins = aggData.bins;
      totalTexts = aggData.total;
      renderPlot();
    } catch (e) {
      console.error('Failed to load arc data:', e);
    }
    loading = false;
  }

  function renderPlot() {
    if (!Plotly || !plotDiv) return;

    const traces: any[] = [];

    // Overall ribbon (q25-q75) from aggregated data
    if (overallBins.length > 0) {
      const decades = overallBins.map(b => b.decade);
      traces.push({
        x: decades,
        y: overallBins.map(b => b.q75),
        type: 'scatter', mode: 'lines',
        line: { color: 'transparent' },
        showlegend: false, hoverinfo: 'skip',
      });
      traces.push({
        x: decades,
        y: overallBins.map(b => b.q25),
        type: 'scatter', mode: 'lines',
        fill: 'tonexty',
        fillcolor: 'rgba(0, 0, 0, 0.06)',
        line: { color: 'transparent' },
        name: 'Overall IQR',
        hoverinfo: 'skip',
      });
      traces.push({
        x: decades,
        y: overallBins.map(b => b.median),
        type: 'scatter', mode: 'lines',
        line: { color: 'rgba(0, 0, 0, 0.3)', width: 1.5, dash: 'dot' },
        name: 'Overall median',
        hovertemplate: 'Decade: %{x}<br>Median: %{y:.3f}<extra></extra>',
      });
    }

    // One line per corpus, colored by genre
    for (const c of corpora) {
      if (c.bins.length < 2) continue; // skip corpora with too few decades
      const color = genreColors[c.genre || ''] || '#999';
      traces.push({
        x: c.bins.map(b => b.decade),
        y: c.bins.map(b => b.mean),
        customdata: c.bins.map(() => c.corpus),
        type: 'scatter',
        mode: 'lines',
        line: { color, width: 2 },
        name: `${c.corpus} (${c.genre || '?'}, ${c.n_texts.toLocaleString()})`,
        hovertemplate:
          `<b>${c.corpus}</b><br>` +
          `${c.genre} &middot; ${c.n_texts.toLocaleString()} texts<br>` +
          'Decade: %{x}<br>' +
          'Mean: %{y:.3f}<extra></extra>',
      });
    }

    Plotly.react(plotDiv, traces, {
      title: {
        text: `Abstraction Arc — ${corpora.length} corpora, ${totalTexts.toLocaleString()} texts`,
        font: { size: 14 },
      },
      xaxis: { title: 'Year', range: [$yearRange[0], $yearRange[1]] },
      yaxis: { title: 'Abstractness (mean score)', zeroline: true },
      margin: { t: 40, r: 20, b: 50, l: 60 },
      legend: { orientation: 'v' as const, x: 1.02, y: 1, font: { size: 10 } },
      hovermode: 'closest' as const,
    }, { responsive: true, scrollZoom: true });

    // Click a corpus line → navigate to corpus detail view
    plotDiv.removeAllListeners?.('plotly_click');
    plotDiv.on('plotly_click', (data: any) => {
      const point = data.points[0];
      if (point?.customdata) {
        goto(`/corpus/${point.customdata}`);
      }
    });
  }

  onMount(async () => {
    Plotly = await import('plotly.js-dist-min');
    loadData();
  });

  $effect(() => {
    $norm; $selectedGenres; $selectedCorpora; $yearRange;
    if (Plotly) loadData();
  });
</script>

<div class="chart-container">
  {#if loading}
    <div class="loading">Loading arc data...</div>
  {/if}
  <div bind:this={plotDiv} class="plot"></div>
</div>

<style>
  .chart-container { flex: 1; display: flex; flex-direction: column; min-height: 0; }
  .plot { flex: 1; min-height: 400px; }
  .loading {
    padding: 1rem; text-align: center; color: #666;
    font-style: italic;
  }
</style>
