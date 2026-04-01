<script lang="ts">
  import { page } from '$app/state';
  import { onMount } from 'svelte';
  import { goto } from '$app/navigation';
  import { norm, selectedGenres, yearRange } from '$lib/stores';
  import { fetchArcTexts } from '$lib/api';
  import type { ArcText } from '$lib/types';

  let corpusName = $derived(page.params.name);

  let plotDiv: HTMLDivElement;
  let Plotly: any;
  let texts: ArcText[] = $state([]);
  let loading = $state(true);
  let totalTexts = $state(0);

  function getParams() {
    const p: Record<string, string | string[]> = {};
    p.col = $norm;
    p.corpus = [corpusName];
    if ($selectedGenres.length) p.genre = $selectedGenres;
    if ($yearRange[0] > 1000) p.year_min = String($yearRange[0]);
    if ($yearRange[1] < 2025) p.year_max = String($yearRange[1]);
    return p;
  }

  async function loadData() {
    loading = true;
    texts = [];
    try {
      let page_num = 0;
      const pageSize = 10000;
      while (true) {
        const result = await fetchArcTexts({
          ...getParams(),
          page: String(page_num),
          page_size: String(pageSize),
        });
        if (result.texts.length === 0) break;
        texts = [...texts, ...result.texts];
        totalTexts = result.total;
        renderPlot();
        if (texts.length >= result.total) break;
        page_num++;
      }
    } catch (e) {
      console.error('Failed to load corpus texts:', e);
    }
    loading = false;
  }

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

  function renderPlot() {
    if (!Plotly || !plotDiv) return;

    const traces: any[] = [];

    // Group by genre
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
          size: 4,
          opacity: 0.5,
        },
        name: genre,
        hovertemplate:
          '<b>%{customdata[2]}</b><br>' +
          '%{customdata[3]}<br>' +
          'Year: %{x}<br>' +
          'Score: %{y:.3f}' +
          '<extra>%{fullData.name}</extra>',
      });
    }

    Plotly.react(plotDiv, traces, {
      title: { text: `${corpusName} (${totalTexts.toLocaleString()} texts)`, font: { size: 14 } },
      xaxis: { title: 'Year' },
      yaxis: { title: 'Abstractness', zeroline: true },
      margin: { t: 40, r: 20, b: 50, l: 60 },
      legend: { orientation: 'h' as const, y: -0.15 },
      hovermode: 'closest' as const,
    }, { responsive: true, scrollZoom: true });

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

  $effect(() => {
    $norm; $selectedGenres; $yearRange;
    if (Plotly) loadData();
  });
</script>

<div class="corpus-page">
  <div class="header">
    <a href="/arc">&larr; Back to arc</a>
    <h2>{corpusName}</h2>
  </div>
  {#if loading && texts.length === 0}
    <div class="loading">Loading texts...</div>
  {/if}
  <div bind:this={plotDiv} class="plot"></div>
</div>

<style>
  .corpus-page { flex: 1; display: flex; flex-direction: column; }
  .header {
    padding: 0.5rem 1rem; border-bottom: 1px solid #eee;
    display: flex; align-items: center; gap: 1rem;
  }
  .header a { color: #1565C0; text-decoration: none; font-size: 0.85rem; }
  .header h2 { margin: 0; font-size: 1.1rem; }
  .plot { flex: 1; min-height: 400px; }
  .loading { padding: 1rem; text-align: center; color: #666; font-style: italic; }
</style>
