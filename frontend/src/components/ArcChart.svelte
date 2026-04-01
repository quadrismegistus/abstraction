<script lang="ts">
  import { onMount } from 'svelte';
  import { goto } from '$app/navigation';
  import { norm, selectedGenres, yearRange, periodMatched, loessSpan, globalLoading } from '$lib/stores';
  import { fetchArcByGenre } from '$lib/api';
  import type { GenreArc } from '$lib/types';

  let plotDiv: HTMLDivElement;
  let Plotly: any;
  let genreArcs: GenreArc[] = $state([]);

  function pStars(p: number | null): string {
    if (p === null) return '';
    if (p < 0.001) return '***';
    if (p < 0.01) return '**';
    if (p < 0.05) return '*';
    return 'n.s.';
  }


  // Match the book figure's visual style
  const genreStyles: Record<string, { color: string; dash: string; width: number }> = {
    'Fiction':    { color: '#222',    dash: 'solid',    width: 3 },
    'Poetry':     { color: '#666',    dash: 'dashdot',  width: 2.5 },
    'Periodical': { color: '#aaa',    dash: 'dash',     width: 2.5 },
    'Drama':      { color: '#cc6600', dash: 'dot',      width: 2 },
    'Essay/Treatise': { color: '#336699', dash: 'longdash', width: 2 },
  };

  // Corpus shapes for scatter points
  const corpusShapes: string[] = [
    'circle', 'square', 'diamond', 'triangle-up', 'triangle-down',
    'pentagon', 'hexagon', 'star', 'cross', 'x',
    'triangle-left', 'triangle-right', 'diamond-wide', 'hourglass',
  ];

  function getParams() {
    const p: Record<string, string | string[]> = {};
    p.col = $norm;
    const genres = $selectedGenres.length
      ? $selectedGenres
      : ['Fiction', 'Poetry', 'Periodical'];
    p.genre = genres;
    p.year_min = String($yearRange[0]);
    p.year_max = String($yearRange[1]);
    if ($periodMatched) p.period_matched = 'true';
    p.loess_span = String($loessSpan);
    return p;
  }

  async function loadData() {
    $globalLoading = true;
    try {
      genreArcs = await fetchArcByGenre(getParams());
      renderPlot();
    } catch (e) {
      console.error('Failed to load arc data:', e);
    }
    $globalLoading = false;
  }

  function renderPlot() {
    if (!Plotly || !plotDiv || genreArcs.length === 0) return;

    const traces: any[] = [];

    // Build a global corpus→shape map
    const allCorpora = [...new Set(genreArcs.flatMap(g => g.points.map(p => p.corpus)))];
    const corpusShapeMap: Record<string, string> = {};
    allCorpora.forEach((c, i) => {
      if (c) corpusShapeMap[c] = corpusShapes[i % corpusShapes.length];
    });

    // Max n_texts for sizing
    const maxN = Math.max(...genreArcs.flatMap(g => g.points.map(p => p.n_texts)), 1);

    for (const arc of genreArcs) {
      const style = genreStyles[arc.genre] || { color: '#999', dash: 'solid', width: 2 };

      // SE ribbon (filled area between se_lo and se_hi)
      if (arc.loess.length > 0) {
        const lx = arc.loess.map(p => p.year);
        traces.push({
          x: [...lx, ...lx.slice().reverse()],
          y: [...arc.loess.map(p => p.se_hi), ...arc.loess.map(p => p.se_lo).reverse()],
          type: 'scatter',
          mode: 'lines',
          fill: 'toself',
          fillcolor: `rgba(150,150,150,0.15)`,
          line: { color: 'transparent' },
          showlegend: false,
          hoverinfo: 'skip',
        });
      }

      // Scatter: corpus-adjusted points, sized by n_texts, shaped by corpus
      const pts = arc.points;
      // Group by corpus for shapes
      const byCorpus: Record<string, typeof pts> = {};
      for (const p of pts) {
        const c = p.corpus || 'unknown';
        (byCorpus[c] ??= []).push(p);
      }

      for (const [corpus, cPts] of Object.entries(byCorpus)) {
        traces.push({
          x: cPts.map(p => p.year),
          y: cPts.map(p => p.adjusted),
          customdata: cPts.map(p => [corpus, p.n_texts]),
          type: 'scatter',
          mode: 'markers',
          marker: {
            color: style.color,
            symbol: corpusShapeMap[corpus] || 'circle',
            size: cPts.map(p => 3 + Math.sqrt(p.n_texts / maxN) * 12),
            opacity: 0.35,
            line: { width: 0.5, color: style.color },
          },
          name: `${arc.genre}: ${corpus}`,
          legendgroup: arc.genre,
          showlegend: false,
          hovertemplate:
            `<b>${arc.genre}</b> — ${corpus}<br>` +
            'Decade: %{x}<br>' +
            'Score: %{y:.3f}<br>' +
            'Texts: %{customdata[1]:,}' +
            '<extra></extra>',
        });
      }

      // LOESS line
      if (arc.loess.length > 0) {
        traces.push({
          x: arc.loess.map(p => p.year),
          y: arc.loess.map(p => p.fitted),
          type: 'scatter',
          mode: 'lines',
          line: { color: style.color, dash: style.dash, width: style.width },
          name: `${arc.genre} (${arc.n_texts_total.toLocaleString()} texts, ${arc.n_corpora} corpora)`,
          legendgroup: arc.genre,
          hovertemplate:
            `<b>${arc.genre}</b><br>` +
            'Year: %{x}<br>' +
            'Score: %{y:.3f}<extra></extra>',
        });
      }
    }

    const totalTexts = genreArcs.reduce((s, g) => s + g.n_texts_total, 0);

    Plotly.react(plotDiv, traces, {
      title: {
        text: `Abstraction Arc (${totalTexts.toLocaleString()} texts)`,
        font: { size: 14 },
      },
      xaxis: {
        title: 'Decade of publication',
        range: [$yearRange[0], $yearRange[1]],
        dtick: 50,
        gridcolor: 'rgba(0,0,0,0.08)',
      },
      yaxis: {
        title: '<< More concrete | More abstract >>',
        zeroline: true,
        zerolinecolor: 'rgba(0,0,0,0.2)',
        gridcolor: 'rgba(0,0,0,0.08)',
      },
      margin: { t: 40, r: 20, b: 50, l: 60 },
      legend: {
        orientation: 'v' as const,
        x: 1.02, y: 1,
        font: { size: 11 },
      },
      hovermode: 'closest' as const,
      plot_bgcolor: 'white',
    }, { responsive: true, scrollZoom: true });

    // Click a point → navigate to that corpus
    plotDiv.removeAllListeners?.('plotly_click');
    plotDiv.on('plotly_click', (data: any) => {
      const point = data.points[0];
      if (point?.customdata?.[0]) {
        goto(`/corpus/${point.customdata[0]}`);
      }
    });
  }

  let debounceTimer: ReturnType<typeof setTimeout>;

  onMount(async () => {
    Plotly = await import('plotly.js-dist-min');
    loadData();
  });

  $effect(() => {
    $norm; $selectedGenres; $yearRange; $periodMatched; $loessSpan;
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      if (Plotly) loadData();
    }, 400);
  });
</script>

<div class="chart-container">
  <div bind:this={plotDiv} class="plot"></div>

  {#if genreArcs.length > 0}
    <div class="stats-table">
      <table>
        <thead>
          <tr>
            <th>Genre</th>
            <th>Texts</th>
            <th>Corpora</th>
            <th>Breakpoint</th>
            <th>Peak (LOESS)</th>
            <th>Rise slope</th>
            <th>Fall slope</th>
            <th>R²</th>
            <th>Fall (SD)</th>
            <th>Start</th>
            <th>End</th>
          </tr>
        </thead>
        <tbody>
          {#each genreArcs as arc}
            {@const s = arc.stats}
            <tr>
              <td class="genre">{arc.genre}</td>
              <td>{s.n_texts.toLocaleString()}</td>
              <td>{s.n_corpora}</td>
              <td>{s.breakpoint ?? '—'}</td>
              <td>{s.peak_year ?? '—'}</td>
              <td class="slope" class:sig={s.rise_slope_p !== null && s.rise_slope_p < 0.001}>
                {s.rise_slope !== null ? (s.rise_slope > 0 ? '+' : '') + s.rise_slope.toFixed(4) + '/dec' : '—'}
                {#if s.rise_slope_p !== null}
                  <span class="p">{pStars(s.rise_slope_p)}</span>
                {/if}
              </td>
              <td class="slope" class:sig={s.fall_slope_p !== null && s.fall_slope_p < 0.001}>
                {s.fall_slope !== null ? (s.fall_slope > 0 ? '+' : '') + s.fall_slope.toFixed(4) + '/dec' : '—'}
                {#if s.fall_slope_p !== null}
                  <span class="p">{pStars(s.fall_slope_p)}</span>
                {/if}
              </td>
              <td>{s.r2 !== null ? s.r2.toFixed(3) : '—'}</td>
              <td>{s.change_sd !== null ? s.change_sd.toFixed(1) + ' SD' : '—'}</td>
              <td class="score">{s.start_score !== null ? s.start_score.toFixed(3) : '—'}</td>
              <td class="score">{s.end_score !== null ? s.end_score.toFixed(3) : '—'}</td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  {/if}
</div>


<style>
  .chart-container { flex: 1; display: flex; flex-direction: column; min-height: 0; }
  .plot { flex: 1; min-height: 500px; }
  .stats-table {
    padding: 0.5rem 1rem; border-top: 1px solid #eee;
    overflow-x: auto; flex-shrink: 0;
  }
  table { border-collapse: collapse; width: 100%; font-size: 0.8rem; }
  th { text-align: left; padding: 4px 8px; border-bottom: 2px solid #ddd; color: #555; font-weight: 600; }
  td { padding: 4px 8px; border-bottom: 1px solid #eee; }
  .genre { font-weight: 600; }
  .slope { font-family: monospace; font-size: 0.75rem; }
  .slope.sig { color: #1a1a2e; }
  .score { font-family: monospace; font-size: 0.75rem; color: #666; }
  .p { font-size: 0.7rem; color: #999; margin-left: 2px; }
</style>
