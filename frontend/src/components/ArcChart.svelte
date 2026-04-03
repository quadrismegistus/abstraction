<script lang="ts">
  import { onMount } from 'svelte';
  import { goto } from '$app/navigation';
  import { norm, selectedGenres, selectedCorpora, yearRange, periodMatched, loessSpan, adjustModel, corpusAdjusted, binSize, globalLoading, corporaList } from '$lib/stores';
  import { fetchArcByGenre } from '$lib/api';
  import type { GenreArc } from '$lib/types';

  let plotDiv: HTMLDivElement;
  let Plotly: any;
  let genreArcs: GenreArc[] = $state([]);
  let mode: 'explore' | 'print' | 'aggregate' = $state('aggregate');
  let showRawTrend = $state(false);
  let printPngUrl = $state('');

  function pStars(p: number | null): string {
    if (p === null) return '';
    if (p < 0.001) return '***';
    if (p < 0.01) return '**';
    if (p < 0.05) return '*';
    return 'n.s.';
  }

  // Shapes per genre for aggregate mode
  const genreShapes: Record<string, string> = {
    'arc_fiction': 'circle',
    'arc_poetry': 'diamond',
    'arc_periodical': 'square',
    'arc_essays': 'triangle-up',
    'Fiction': 'circle',
    'Poetry': 'diamond',
    'Periodical': 'square',
    'Essay': 'triangle-up',
    'Drama': 'triangle-down',
    'Sermon': 'pentagon',
    'Letters': 'hexagon',
  };

  // Line styles per genre/arc corpus
  const genreStyles: Record<string, { color: string; dash: string; width: number }> = {
    'arc_fiction':    { color: '#222',    dash: 'solid',    width: 3 },
    'arc_poetry':     { color: '#666',    dash: 'dashdot',  width: 2.5 },
    'arc_periodical': { color: '#aaa',    dash: 'dash',     width: 2.5 },
    'arc_essays':     { color: '#336699', dash: 'longdash',  width: 2 },
    'Fiction':    { color: '#222',    dash: 'solid',    width: 3 },
    'Poetry':     { color: '#666',    dash: 'dashdot',  width: 2.5 },
    'Periodical': { color: '#aaa',    dash: 'dash',     width: 2.5 },
    'Drama':      { color: '#cc6600', dash: 'dot',      width: 2 },
    'Essay':      { color: '#336699', dash: 'longdash',  width: 2 },
    'Treatise':   { color: '#669933', dash: 'longdashdot', width: 2 },
  };

  // Explore mode: distinct colors per corpus
  const corpusColors: string[] = [
    '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
    '#42d4f4', '#f032e6', '#bfef45', '#469990', '#dcbeff',
    '#9a6324', '#800000', '#aaffc3', '#808000', '#000075',
    '#a9a9a9', '#e6beff', '#ffe119', '#fabebe', '#7f7f7f',
  ];

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
      : ['arc_fiction'];
    p.genre = genres;
    // Only pass corpus filter if some are unchecked
    const allCorpusNames = $corporaList.map(c => c.name);
    if ($selectedCorpora.length > 0 && $selectedCorpora.length < allCorpusNames.length) {
      p.corpus = $selectedCorpora;
    }
    p.year_min = String($yearRange[0]);
    p.year_max = String($yearRange[1]);
    if ($periodMatched) p.period_matched = 'true';
    if ($corpusAdjusted) p.corpus_adjusted = 'true';
    p.loess_span = String($loessSpan);
    p.model = $adjustModel;
    p.bin_size = String($binSize);
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
    if (mode === 'aggregate') renderAggregate();
    else if (mode === 'explore') renderExplore();
    else renderPrint();
  }

  function renderAggregate() {
    const traces: any[] = [];

    for (const arc of genreArcs) {
      const gs = genreStyles[arc.genre] || { color: '#333', dash: 'solid', width: 2 };
      const scoreKey = $corpusAdjusted ? 'adjusted' : 'score';

      // Aggregate all corpus points into one point per year bin
      const bins: Record<number, { sum: number; count: number }> = {};
      for (const p of arc.points) {
        const val = scoreKey === 'adjusted' ? p.adjusted : p.score;
        const yr = p.year;
        if (!bins[yr]) bins[yr] = { sum: 0, count: 0 };
        bins[yr].sum += val * p.n_texts;
        bins[yr].count += p.n_texts;
      }

      const years = Object.keys(bins).map(Number).sort((a, b) => a - b);
      const means = years.map(y => bins[y].sum / bins[y].count);
      const counts = years.map(y => bins[y].count);
      const maxN = Math.max(...counts, 1);

      // SE ribbon from aggregate LOESS
      if (arc.loess_aggregate.length > 0) {
        const lx = arc.loess_aggregate.map(p => p.year);
        traces.push({
          x: [...lx, ...lx.slice().reverse()],
          y: [...arc.loess_aggregate.map(p => p.se_hi), ...arc.loess_aggregate.map(p => p.se_lo).reverse()],
          type: 'scatter', mode: 'lines', fill: 'toself',
          fillcolor: 'rgba(0,0,0,0.06)', line: { color: 'transparent' },
          showlegend: false, hoverinfo: 'skip',
        });
      }

      // Aggregated points — one per bin, sized by n_texts, shaped by genre
      traces.push({
        x: years,
        y: means,
        customdata: years.map((y, i) => ['', counts[i], arc.genre, y]),
        type: 'scatter',
        mode: 'markers',
        marker: {
          color: gs.color,
          symbol: genreShapes[arc.genre] || 'circle',
          size: counts.map(n => 4 + Math.sqrt(n / maxN) * 16),
          opacity: 0.5,
          line: { width: 0.5, color: gs.color },
        },
        name: `${arc.genre} (${arc.n_texts_total.toLocaleString()} texts)`,
        hovertemplate:
          `<b>${arc.genre}</b><br>` +
          'Year: %{x}<br>Mean: %{y:.3f}<br>' +
          'Texts: %{customdata[0]:,}<extra></extra>',
      });

      // Aggregate LOESS (fitted on text-weighted year bins)
      if (arc.loess_aggregate.length > 0) {
        traces.push({
          x: arc.loess_aggregate.map(p => p.year),
          y: arc.loess_aggregate.map(p => p.fitted),
          type: 'scatter', mode: 'lines',
          line: { color: gs.color, dash: gs.dash, width: gs.width },
          name: `${arc.genre} LOESS`,
          showlegend: false,
          hovertemplate: `<b>${arc.genre}</b><br>Year: %{x}<br>Score: %{y:.3f}<extra></extra>`,
        });
      }
    }

    _renderLayout(traces);
  }

  function renderExplore() {
    const traces: any[] = [];

    // Build corpus → color/shape maps
    const allCorpora = [...new Set(genreArcs.flatMap(g => g.points.map(p => p.corpus)).filter(Boolean))];
    const corpusColorMap: Record<string, string> = {};
    const corpusShapeMap: Record<string, string> = {};
    allCorpora.forEach((c, i) => {
      corpusColorMap[c!] = corpusColors[i % corpusColors.length];
      corpusShapeMap[c!] = corpusShapes[i % corpusShapes.length];
    });

    const maxN = Math.max(...genreArcs.flatMap(g => g.points.map(p => p.n_texts)), 1);

    for (const arc of genreArcs) {
      // SE ribbon
      if (arc.loess.length > 0) {
        const lx = arc.loess.map(p => p.year);
        traces.push({
          x: [...lx, ...lx.slice().reverse()],
          y: [...arc.loess.map(p => p.se_hi), ...arc.loess.map(p => p.se_lo).reverse()],
          type: 'scatter', mode: 'lines', fill: 'toself',
          fillcolor: 'rgba(0,0,0,0.06)', line: { color: 'transparent' },
          showlegend: false, hoverinfo: 'skip',
        });
      }

      // Group points by corpus
      const byCorpus: Record<string, typeof arc.points> = {};
      for (const p of arc.points) {
        const c = p.corpus || 'unknown';
        (byCorpus[c] ??= []).push(p);
      }

      for (const [corpus, cPts] of Object.entries(byCorpus)) {
        const color = corpusColorMap[corpus] || '#999';
        const symbol = corpusShapeMap[corpus] || 'circle';

        // Raw (unadjusted) points — faint background (only when adjustment is on)
        if ($corpusAdjusted) traces.push({
          x: cPts.map(p => p.year),
          y: cPts.map(p => p.score),
          customdata: cPts.map(p => [corpus, p.n_texts, arc.genre, p.year]),
          type: 'scatter', mode: 'markers',
          marker: {
            color, symbol,
            size: cPts.map(p => 4 + Math.sqrt(p.n_texts / maxN) * 8),
            opacity: 0.15,
            line: { width: 0 },
          },
          name: corpus,
          legendgroup: corpus,
          showlegend: false,
          hovertemplate:
            `<b>${corpus}</b> (${arc.genre})<br>` +
            'Decade: %{x}<br>Raw: %{y:.3f}<br>' +
            'Texts: %{customdata[1]:,}<extra>raw</extra>',
        });

        // Main points — bold foreground (raw or adjusted depending on mode)
        traces.push({
          x: cPts.map(p => p.year),
          y: cPts.map(p => $corpusAdjusted ? p.adjusted : p.score),
          customdata: cPts.map(p => [corpus, p.n_texts, arc.genre, p.year]),
          type: 'scatter', mode: 'markers',
          marker: {
            color, symbol,
            size: cPts.map(p => 6 + Math.sqrt(p.n_texts / maxN) * 12),
            opacity: 0.6,
            line: { width: 0.5, color: 'white' },
          },
          name: corpus,
          legendgroup: corpus,
          showlegend: !traces.some(t => t.legendgroup === corpus && t.showlegend),
          hovertemplate:
            `<b>${corpus}</b> (${arc.genre})<br>` +
            'Decade: %{x}<br>Adjusted: %{y:.3f}<br>' +
            'Texts: %{customdata[1]:,}<extra>adjusted</extra>',
        });
      }

      // Raw LOESS (pre-adjustment trend, thin dashed)
      if (showRawTrend && arc.loess_raw.length > 0) {
        const gs = genreStyles[arc.genre] || { color: '#333', dash: 'solid', width: 2 };
        traces.push({
          x: arc.loess_raw.map(p => p.year),
          y: arc.loess_raw.map(p => p.fitted),
          type: 'scatter', mode: 'lines',
          line: { color: gs.color, dash: 'dot', width: 1.5, },
          opacity: 0.5,
          name: `${arc.genre} raw`,
          legendgroup: `_loess_${arc.genre}`,
          showlegend: false,
          hovertemplate: `<b>${arc.genre}</b> (raw)<br>Year: %{x}<br>Score: %{y:.3f}<extra>unadjusted</extra>`,
        });
      }

      // LOESS line (adjusted, on top)
      if (arc.loess.length > 0) {
        const gs = genreStyles[arc.genre] || { color: '#333', dash: 'solid', width: 2 };
        traces.push({
          x: arc.loess.map(p => p.year),
          y: arc.loess.map(p => p.fitted),
          type: 'scatter', mode: 'lines',
          line: { color: gs.color, dash: gs.dash, width: gs.width },
          name: `${arc.genre} LOESS`,
          legendgroup: `_loess_${arc.genre}`,
          hovertemplate: `<b>${arc.genre}</b><br>Year: %{x}<br>Score: %{y:.3f}<extra>adjusted</extra>`,
        });
      }
    }

    _renderLayout(traces);
  }

  function renderPrint() {
    const traces: any[] = [];
    const maxN = Math.max(...genreArcs.flatMap(g => g.points.map(p => p.n_texts)), 1);

    for (const arc of genreArcs) {
      const style = genreStyles[arc.genre] || { color: '#999', dash: 'solid', width: 2 };

      // SE ribbon
      if (arc.loess.length > 0) {
        const lx = arc.loess.map(p => p.year);
        traces.push({
          x: [...lx, ...lx.slice().reverse()],
          y: [...arc.loess.map(p => p.se_hi), ...arc.loess.map(p => p.se_lo).reverse()],
          type: 'scatter', mode: 'lines', fill: 'toself',
          fillcolor: 'rgba(150,150,150,0.15)', line: { color: 'transparent' },
          showlegend: false, hoverinfo: 'skip',
        });
      }

      // Adjusted points — grayscale, shaped by corpus
      const byCorpus: Record<string, typeof arc.points> = {};
      for (const p of arc.points) {
        (byCorpus[p.corpus || 'unknown'] ??= []).push(p);
      }

      const corpusKeys = Object.keys(byCorpus);
      for (const [ci, [corpus, cPts]] of Object.entries(Object.entries(byCorpus))) {
        traces.push({
          x: cPts.map(p => p.year),
          y: cPts.map(p => p.adjusted),
          customdata: cPts.map(p => [corpus, p.n_texts, arc.genre, p.year]),
          type: 'scatter', mode: 'markers',
          marker: {
            color: style.color,
            symbol: corpusShapes[Number(ci) % corpusShapes.length],
            size: cPts.map(p => 6 + Math.sqrt(p.n_texts / maxN) * 12),
            opacity: 0.35,
            line: { width: 0.5, color: style.color },
          },
          name: `${arc.genre}: ${corpus}`,
          legendgroup: arc.genre,
          showlegend: false,
          hovertemplate:
            `<b>${arc.genre}</b> — ${corpus}<br>` +
            'Decade: %{x}<br>Score: %{y:.3f}<br>' +
            'Texts: %{customdata[1]:,}<extra></extra>',
        });
      }

      // Raw LOESS (pre-adjustment, thin dotted)
      if (showRawTrend && arc.loess_raw.length > 0) {
        traces.push({
          x: arc.loess_raw.map(p => p.year),
          y: arc.loess_raw.map(p => p.fitted),
          type: 'scatter', mode: 'lines',
          line: { color: style.color, dash: 'dot', width: 1.5 },
          opacity: 0.4,
          name: `${arc.genre} raw`,
          legendgroup: arc.genre,
          showlegend: false,
          hovertemplate: `<b>${arc.genre}</b> (raw)<br>Year: %{x}<br>Score: %{y:.3f}<extra>unadjusted</extra>`,
        });
      }

      // LOESS line (adjusted)
      if (arc.loess.length > 0) {
        traces.push({
          x: arc.loess.map(p => p.year),
          y: arc.loess.map(p => p.fitted),
          type: 'scatter', mode: 'lines',
          line: { color: style.color, dash: style.dash, width: style.width },
          name: `${arc.genre} (${arc.n_texts_total.toLocaleString()} texts, ${arc.n_corpora} corpora)`,
          legendgroup: arc.genre,
          hovertemplate: `<b>${arc.genre}</b><br>Year: %{x}<br>Score: %{y:.3f}<extra>adjusted</extra>`,
        });
      }
    }

    _renderLayout(traces);
  }

  function _renderLayout(traces: any[]) {
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
        font: { size: 10 },
      },
      hovermode: 'closest' as const,
      plot_bgcolor: 'white',
    }, { responsive: true, scrollZoom: true });

    plotDiv.removeAllListeners?.('plotly_click');
    plotDiv.on('plotly_click', (data: any) => {
      const point = data.points[0];
      if (point?.customdata) {
        const [corpus, nTexts, genre, yr] = point.customdata;
        const params = new URLSearchParams();
        if (corpus) params.set('corpus', corpus);
        params.set('year', String(yr));
        params.set('bin_size', String($binSize));
        params.set('genre', genre);
        goto(`/texts?${params.toString()}`);
      }
    });
  }

  let debounceTimer: ReturnType<typeof setTimeout>;

  onMount(async () => {
    Plotly = await import('plotly.js-dist-min');
    loadData();
  });

  $effect(() => {
    $norm; $selectedGenres; $selectedCorpora; $yearRange; $periodMatched; $corpusAdjusted; $loessSpan; $adjustModel; $binSize;
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      if (Plotly) loadData();
    }, 400);
  });

  // Re-render (no reload) when display toggles change
  $effect(() => {
    mode; showRawTrend;
    if (Plotly && genreArcs.length > 0) renderPlot();
  });
</script>

<div class="chart-container">
  <div class="mode-toggle">
    <button class:active={mode === 'aggregate'} onclick={() => mode = 'aggregate'}>Aggregate</button>
    <button class:active={mode === 'explore'} onclick={() => mode = 'explore'}>Explore</button>
    <button class:active={mode === 'print'} onclick={() => mode = 'print'}>Print</button>
    <button class:active={showRawTrend} onclick={() => showRawTrend = !showRawTrend}>
      {$corpusAdjusted ? 'Show raw' : 'Show adjusted'}
    </button>
    <button onclick={() => {
      if (Plotly && plotDiv) {
        Plotly.downloadImage(plotDiv, {
          format: 'png', width: 3000, height: 1800,
          filename: 'arc_plot', scale: 1,
        });
      }
    }}>Export PNG</button>
    <button onclick={async () => {
      const res = await fetch(`http://${window.location.hostname}:1709/api/arc/print`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(genreArcs),
      });
      if (res.ok) {
        const blob = await res.blob();
        printPngUrl = URL.createObjectURL(blob);
        mode = 'print';
      }
    }}>Plotnine</button>
  </div>
  {#if mode === 'print' && printPngUrl}
    <div class="print-container">
      <img src={printPngUrl} alt="Arc plot (print quality)" class="print-img" />
      <p class="print-hint">Right-click to save image</p>
    </div>
  {:else}
    <div bind:this={plotDiv} class="plot"></div>
  {/if}

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
  .print-container { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 1rem; overflow: auto; }
  .print-img { max-width: 100%; max-height: 100%; object-fit: contain; }
  .print-hint { color: #999; font-size: 0.8rem; margin-top: 0.5rem; }
  .mode-toggle {
    display: flex; gap: 4px; padding: 4px 1rem; flex-shrink: 0;
  }
  .mode-toggle button {
    padding: 3px 12px; font-size: 0.8rem; border: 1px solid #ccc;
    background: white; border-radius: 3px; cursor: pointer;
  }
  .mode-toggle button.active { background: #333; color: white; border-color: #333; }
  .plot { flex: 1 1 0; min-height: 300px; max-height: calc(100vh - 200px); }
  .stats-table {
    padding: 0.5rem 1rem; border-top: 1px solid #eee;
    overflow-x: auto; flex-shrink: 0; flex-grow: 0;
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
