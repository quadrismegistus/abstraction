<script lang="ts">
  import { norm, chunkSize } from '$lib/stores';
  import { fetchPassage } from '$lib/api';
  import type { PassageResponse } from '$lib/types';

  let { corpus, textId, chunkIndex }: {
    corpus: string; textId: string; chunkIndex: number;
  } = $props();

  let data: PassageResponse | null = $state(null);
  let loading = $state(true);
  let error = $state('');
  let mode: 'color' | 'print' = $state('color');

  // Tooltip state
  let tooltipWord = $state('');
  let tooltipZ = $state('');
  let tooltipClass = $state('');
  let showTooltip = $state(false);

  async function loadData() {
    loading = true;
    error = '';
    try {
      data = await fetchPassage(corpus, textId, chunkIndex, {
        col: $norm,
        chunk_size: String($chunkSize),
      });
    } catch (e: any) {
      error = e.message;
    }
    loading = false;
  }

  function handleMouseOver(e: MouseEvent) {
    const el = e.target as HTMLElement;
    if (!el.classList.contains('w')) return;
    const z = el.dataset.z;
    tooltipWord = el.textContent || '';
    tooltipZ = z || '';
    tooltipClass = el.classList.contains('abstract') ? 'abstract'
      : el.classList.contains('concrete') ? 'concrete'
      : el.classList.contains('unscored') ? 'unscored'
      : 'neither';
    showTooltip = true;
  }

  function handleMouseOut(e: MouseEvent) {
    const el = e.target as HTMLElement;
    if (el.classList.contains('w')) showTooltip = false;
  }

  function exportPrint() {
    if (!data) return;
    const w = window.open('', '_blank');
    if (w) {
      w.document.write(data.print_html);
      w.document.close();
    }
  }

  $effect(() => {
    $norm; $chunkSize;
    loadData();
  });
</script>

<div class="passage-container">
  {#if loading}
    <div class="loading">Loading passage...</div>
  {:else if error}
    <div class="error">Error: {error}</div>
  {:else if data}
    {@const total = data.n_abstract + data.n_concrete + data.n_neutral}
    <div class="toolbar">
      <div class="stats">
        <span class="stat abstract">{data.n_abstract} abstract ({total ? (data.n_abstract/total*100).toFixed(1) : 0}%)</span>
        <span class="stat concrete">{data.n_concrete} concrete ({total ? (data.n_concrete/total*100).toFixed(1) : 0}%)</span>
        <span class="stat neutral">{data.n_neutral} neutral</span>
      </div>
      <div class="controls">
        <button class:active={mode === 'color'} onclick={() => mode = 'color'}>Color</button>
        <button class:active={mode === 'print'} onclick={() => mode = 'print'}>Print</button>
        <button class="export-btn" onclick={exportPrint}>Export</button>
      </div>
    </div>

    <div class="legend">
      {#if mode === 'color'}
        <span class="legend-item" style="background:hsla(220,70%,55%,0.45); border-radius:2px; padding:1px 5px;">abstract</span>
        <span class="legend-arrow">&larr;</span>
        <span class="legend-item" style="padding:1px 5px;">neutral</span>
        <span class="legend-arrow">&rarr;</span>
        <span class="legend-item" style="background:hsla(25,85%,55%,0.45); border-radius:2px; padding:1px 5px;">concrete</span>
        <span class="legend-sep">|</span>
        <span class="legend-item unscored-label">plain = unscored</span>
      {:else}
        <span class="legend-item" style="outline:3px solid rgba(0,0,0,0.6); outline-offset:0; border-radius:2px; padding:0 3px;">abstract</span>
        <span class="legend-item" style="outline:1px solid rgba(0,0,0,0.15); outline-offset:0; border-radius:2px; padding:0 3px;">slightly abstract</span>
        <span class="legend-item" style="font-weight:500; background:rgba(0,0,0,0.08); border-radius:2px; padding:0 3px;">slightly concrete</span>
        <span class="legend-item" style="font-weight:800; background:rgba(0,0,0,0.30); border-radius:2px; padding:0 3px;">concrete</span>
        <span class="legend-item" style="color:#888;">plain = unscored</span>
      {/if}
    </div>

    <!-- svelte-ignore a11y_no_static_element_interactions -->
    <div
      class="passage-body"
      onmouseover={handleMouseOver}
      onmouseout={handleMouseOut}
    >
      {#if mode === 'color'}
        {@html data.body_html}
      {:else}
        {@html data.print_body_html}
      {/if}
    </div>

    {#if showTooltip}
      <div class="tooltip">
        <strong>{tooltipWord}</strong>
        {#if tooltipZ}
          <span>z = {tooltipZ}</span>
          <span class={tooltipClass}>
            {tooltipClass === 'abstract' ? 'Abstract' : tooltipClass === 'concrete' ? 'Concrete' : 'Neutral'}
          </span>
        {:else}
          <span class="unscored">Not in vocabulary</span>
        {/if}
      </div>
    {/if}
  {/if}
</div>

<style>
  .passage-container { flex: 1; padding: 1rem; overflow-y: auto; }

  .toolbar {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 0.5rem; padding: 0.5rem;
    background: #f5f5f5; border-radius: 4px;
  }
  .stats { display: flex; gap: 1rem; font-size: 0.85rem; }
  .stat.abstract { color: hsl(220, 70%, 40%); }
  .stat.concrete { color: hsl(25, 85%, 40%); }
  .stat.neutral { color: #666; }
  .controls { display: flex; gap: 4px; }
  .controls button {
    padding: 3px 10px; font-size: 0.8rem; border: 1px solid #ccc;
    background: white; border-radius: 3px; cursor: pointer;
  }
  .controls button.active { background: #333; color: white; border-color: #333; }
  .export-btn { margin-left: 8px; }

  .legend {
    display: flex; gap: 0.75rem; align-items: center;
    margin-bottom: 0.75rem; font-size: 0.8rem; color: #555;
  }
  .legend-arrow { color: #bbb; }
  .legend-sep { color: #ccc; }
  .unscored-label { color: #888; }

  /* Passage body */
  .passage-body {
    line-height: 2.2; font-size: 1.05rem;
    font-family: Georgia, "Times New Roman", serif;
    max-width: 700px;
  }
  .passage-body :global(.psg-para) { margin: 0; text-indent: 2em; }
  .passage-body :global(.psg-first) { text-indent: 0; }
  .passage-body :global(.w) { cursor: default; border-radius: 2px; }
  .passage-body :global(.w:hover) { outline: 2px solid rgba(0,0,0,0.3); outline-offset: 0; }
  .passage-body :global(.w.unscored) { color: #888; }

  /* Tooltip */
  .tooltip {
    position: fixed; bottom: 1rem; right: 1rem;
    background: white; border: 1px solid #ddd; border-radius: 6px;
    padding: 0.5rem 0.75rem; box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    display: flex; gap: 0.75rem; align-items: center; font-size: 0.85rem;
    z-index: 100;
  }
  .tooltip .abstract { color: hsl(220, 70%, 35%); font-weight: 600; }
  .tooltip .concrete { color: hsl(25, 90%, 40%); font-weight: 600; }
  .tooltip .unscored { color: #999; font-style: italic; }

  .loading, .error { padding: 2rem; text-align: center; }
  .loading { color: #666; font-style: italic; }
  .error { color: #d32f2f; }
</style>
