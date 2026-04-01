<script lang="ts">
  import { onMount } from 'svelte';
  import { goto } from '$app/navigation';
  import { norm, selectedGenres, selectedCorpora, yearRange } from '$lib/stores';
  import { fetchArcAggregated, fetchArcTexts } from '$lib/api';
  import type { ArcBin, ArcText } from '$lib/types';

  let plotDiv: HTMLDivElement;
  let Plotly: any;
  let bins: ArcBin[] = $state([]);
  let texts: ArcText[] = $state([]);
  let loading = $state(true);
  let totalTexts = $state(0);

  // Build query params from current filter state
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
    texts = [];
    try {
      // Load aggregated data first (fast)
      const agg = await fetchArcAggregated(getParams());
      bins = agg.bins;
      totalTexts = agg.total;
      renderPlot();

      // Then progressively load raw texts
      let page = 0;
      const pageSize = 5000;
      while (true) {
        const result = await fetchArcTexts({ ...getParams(), page: String(page), page_size: String(pageSize) });
        if (result.texts.length === 0) break;
        texts = [...texts, ...result.texts];
        totalTexts = result.total;
        renderPlot();
        if (texts.length >= result.total) break;
        page++;
      }
    } catch (e) {
      console.error('Failed to load arc data:', e);
    }
    loading = false;
  }

  function renderPlot() {
    if (!Plotly || !plotDiv) return;

    const traces: any[] = [];

    // Ribbon from aggregated data (q25-q75)
    if (bins.length > 0) {
      const decades = bins.map(b => b.decade);
      const q25 = bins.map(b => b.q25);
      const q75 = bins.map(b => b.q75);
      const medians = bins.map(b => b.median);

      // Upper bound
      traces.push({
        x: decades,
        y: q75,
        type: 'scatter',
        mode: 'lines',
        line: { color: 'transparent' },
        showlegend: false,
        hoverinfo: 'skip',
      });

      // Lower bound (fill to upper)
      traces.push({
        x: decades,
        y: q25,
        type: 'scatter',
        mode: 'lines',
        fill: 'tonexty',
        fillcolor: 'rgba(100, 100, 200, 0.15)',
        line: { color: 'transparent' },
        showlegend: false,
        hoverinfo: 'skip',
      });

      // Median line
      traces.push({
        x: decades,
        y: medians,
        type: 'scatter',
        mode: 'lines',
        line: { color: 'rgba(60, 60, 150, 0.8)', width: 2 },
        name: 'Median trend',
        hovertemplate: 'Decade: %{x}<br>Median: %{y:.3f}<extra></extra>',
      });
    }

    // Raw scatter points
    if (texts.length > 0) {
      // Color by genre
      const genreColors: Record<string, string> = {
        'Fiction': '#2196F3',
        'Poetry': '#4CAF50',
        'Drama': '#FF9800',
        'Periodical': '#9C27B0',
        'Essay/Treatise': '#795548',
        'Letters': '#00BCD4',
        'Legal': '#607D8B',
        'Political': '#F44336',
      };

      // Group texts by genre for legend
      const byGenre: Record<string, ArcText[]> = {};
      for (const t of texts) {
        const g = t.genre || 'Other';
        (byGenre[g] ??= []).push(t);
      }

      for (const [genre, gTexts] of Object.entries(byGenre)) {
        traces.push({
          x: gTexts.map(t => t.year),
          y: gTexts.map(t => t.score),
          customdata: gTexts.map(t => [t.id, t.corpus, t.title, t.author]),
          type: 'scattergl',
          mode: 'markers',
          marker: {
            color: genreColors[genre] || '#999',
            size: 3,
            opacity: 0.3,
          },
          name: genre,
          hovertemplate:
            '<b>%{customdata[2]}</b><br>' +
            '%{customdata[3]}<br>' +
            'Year: %{x}<br>' +
            'Score: %{y:.3f}<br>' +
            '<i>%{customdata[1]}</i>' +
            '<extra>%{fullData.name}</extra>',
        });
      }
    }

    const layout = {
      title: {
        text: `Abstraction Arc (${totalTexts.toLocaleString()} texts)`,
        font: { size: 14 },
      },
      xaxis: { title: 'Year', range: [$yearRange[0], $yearRange[1]] },
      yaxis: { title: $norm.split('.').slice(1).join(' — '), zeroline: true },
      margin: { t: 40, r: 20, b: 50, l: 60 },
      legend: { orientation: 'h' as const, y: -0.15 },
      hovermode: 'closest' as const,
    };

    const config = { responsive: true, scrollZoom: true };

    Plotly.react(plotDiv, traces, layout, config);

    // Click handler: navigate to text trajectory
    plotDiv.removeAllListeners?.('plotly_click');
    plotDiv.on('plotly_click', (data: any) => {
      const point = data.points[0];
      if (point?.customdata) {
        const [id, corpus] = point.customdata;
        goto(`/text/${corpus}/${id}`);
      }
    });
  }

  onMount(async () => {
    Plotly = await import('plotly.js-dist-min');
    loadData();
  });

  // Reload when filters change
  $effect(() => {
    // Track reactive deps
    $norm; $selectedGenres; $selectedCorpora; $yearRange;
    if (Plotly) loadData();
  });
</script>

<div class="chart-container">
  {#if loading && texts.length === 0}
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
