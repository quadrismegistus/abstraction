<script lang="ts">
  import { norm, selectedGenres, selectedCorpora, yearRange, periodMatched, loessSpan, adjustModel, corpusAdjusted, corpusSmoothed, binSize, splitBy, translatedFilter, minTexts, genresList, corporaList, normsList } from '$lib/stores';
  import type { NormInfo } from '$lib/types';

  // Group norms by source and period
  let normsBySource: Record<string, NormInfo[]> = $derived(
    $normsList.reduce((acc, n) => {
      (acc[n.source] ??= []).push(n);
      return acc;
    }, {} as Record<string, NormInfo[]>)
  );

  let currentNormParts = $derived(() => {
    const m = $norm.match(/Abs-Conc\.(.+)\.(.+)/);
    return m ? { source: m[1], period: m[2] } : { source: 'Median', period: 'median' };
  });

  function setNorm(source: string, period: string) {
    $norm = `Abs-Conc.${source}.${period}`;
  }

  function toggleGenre(g: string) {
    $selectedGenres = $selectedGenres.includes(g)
      ? $selectedGenres.filter(x => x !== g)
      : [...$selectedGenres, g];
  }

  function toggleCorpus(c: string) {
    $selectedCorpora = $selectedCorpora.includes(c)
      ? $selectedCorpora.filter(x => x !== c)
      : [...$selectedCorpora, c];
  }

</script>

<aside class="filter-panel">
  <h3>Filters</h3>

  <section>
    <h4>Norm</h4>
    <select onchange={(e) => {
      const val = (e.target as HTMLSelectElement).value;
      $norm = val;
    }}>
      {#each $normsList as n}
        <option value={n.col} selected={n.col === $norm}>{n.label}</option>
      {/each}
    </select>
    <label class="toggle">
      <input type="checkbox" checked={$periodMatched}
             onchange={() => $periodMatched = !$periodMatched} />
      Period-matched norms
    </label>
    <label class="toggle">
      <input type="checkbox" checked={$corpusSmoothed}
             onchange={() => $corpusSmoothed = !$corpusSmoothed} />
      Corpus-smoothed
    </label>
    {#if $corpusSmoothed}
      <label class="toggle">
        <input type="checkbox" checked={$corpusAdjusted}
               onchange={() => $corpusAdjusted = !$corpusAdjusted} />
        Corpus adjustment
      </label>
    {/if}
    <h4>Bin size</h4>
    <select onchange={(e) => $binSize = +(e.target as HTMLSelectElement).value}>
      <option value="1" selected={$binSize === 1}>1 year</option>
      <option value="5" selected={$binSize === 5}>5 years</option>
      <option value="10" selected={$binSize === 10}>10 years (decade)</option>
      <option value="25" selected={$binSize === 25}>25 years</option>
    </select>
    <h4>LOESS span</h4>
    <div class="slider-row">
      <input type="range" min="0.05" max="0.8" step="0.05"
             value={$loessSpan}
             oninput={(e) => $loessSpan = +(e.target as HTMLInputElement).value} />
      <span class="slider-val">{$loessSpan.toFixed(2)}</span>
    </div>
    <h4>Min texts per point</h4>
    <div class="slider-row">
      <input type="range" min="1" max="20" step="1"
             value={$minTexts}
             oninput={(e) => $minTexts = +(e.target as HTMLInputElement).value} />
      <span class="slider-val">{$minTexts}</span>
    </div>
    <h4>Split by</h4>
    <select onchange={(e) => $splitBy = (e.target as HTMLSelectElement).value}>
      <option value="" selected={$splitBy === ''}>None</option>
      <option value="genre_raw" selected={$splitBy === 'genre_raw'}>Genre (raw)</option>
      <option value="corpus_name" selected={$splitBy === 'corpus_name'}>Corpus</option>
      <option value="is_translated" selected={$splitBy === 'is_translated'}>Translated?</option>
    </select>
    <h4>Translations</h4>
    <select onchange={(e) => $translatedFilter = (e.target as HTMLSelectElement).value}>
      <option value="" selected={$translatedFilter === ''}>All texts</option>
      <option value="false" selected={$translatedFilter === 'false'}>Originals only</option>
      <option value="true" selected={$translatedFilter === 'true'}>Translations only</option>
    </select>
    <h4>Adjustment model</h4>
    <select onchange={(e) => $adjustModel = (e.target as HTMLSelectElement).value}>
      <option value="quadratic" selected={$adjustModel === 'quadratic'}>Quadratic</option>
      <option value="cubic" selected={$adjustModel === 'cubic'}>Cubic</option>
      <option value="quartic" selected={$adjustModel === 'quartic'}>Quartic</option>
      <option value="piecewise" selected={$adjustModel === 'piecewise'}>Piecewise</option>
    </select>
  </section>

  <section>
    <h4>Genre</h4>
    <div class="checkbox-group">
      {#each $genresList as g}
        <label>
          <input type="checkbox" checked={$selectedGenres.includes(g)}
                 onchange={() => toggleGenre(g)} />
          {g}
        </label>
      {/each}
    </div>
    {#if $selectedGenres.length > 0}
      <button class="clear-btn" onclick={() => $selectedGenres = []}>Clear</button>
    {/if}
  </section>

  <section>
    <h4>Year range</h4>
    <div class="range-inputs">
      <input type="number" value={$yearRange[0]} min="1000" max="2025"
             oninput={(e) => $yearRange = [+(e.target as HTMLInputElement).value, $yearRange[1]]} />
      <span>&ndash;</span>
      <input type="number" value={$yearRange[1]} min="1000" max="2025"
             oninput={(e) => $yearRange = [$yearRange[0], +(e.target as HTMLInputElement).value]} />
    </div>
  </section>

  <section>
    <h4>Corpora ({$corporaList.length})</h4>
    <div class="checkbox-group scrollable">
      {#each $corporaList as c}
        <label>
          <input type="checkbox" checked={$selectedCorpora.includes(c.name)}
                 onchange={() => toggleCorpus(c.name)} />
          {c.name} <span class="count">({c.n_texts.toLocaleString()})</span>
        </label>
      {/each}
    </div>
    <div class="btn-row">
      <button class="clear-btn" onclick={() => $selectedCorpora = $corporaList.map(c => c.name)}>All</button>
      <button class="clear-btn" onclick={() => $selectedCorpora = []}>None</button>
      <span class="count">{$selectedCorpora.length}/{$corporaList.length}</span>
    </div>
  </section>
</aside>

<style>
  .filter-panel {
    width: 240px;
    min-width: 240px;
    padding: 1rem;
    border-right: 1px solid #ddd;
    overflow-y: auto;
    font-size: 0.85rem;
  }
  h3 { margin: 0 0 1rem; font-size: 1rem; }
  h4 { margin: 0.75rem 0 0.25rem; font-size: 0.85rem; color: #555; }
  section { margin-bottom: 0.5rem; }
  select { width: 100%; padding: 0.25rem; font-size: 0.8rem; }
  .toggle { margin-top: 6px; font-size: 0.8rem; color: #666; }
  .slider-row { display: flex; align-items: center; gap: 6px; }
  .slider-row input[type="range"] { flex: 1; }
  .slider-val { font-size: 0.75rem; color: #888; min-width: 2.5em; }
  .checkbox-group { display: flex; flex-direction: column; gap: 2px; }
  .checkbox-group.scrollable { max-height: 200px; overflow-y: auto; }
  label { display: flex; align-items: center; gap: 4px; cursor: pointer; }
  .count { color: #999; font-size: 0.75rem; }
  .btn-row { display: flex; align-items: center; gap: 4px; margin-top: 4px; }
  .clear-btn {
    padding: 2px 8px; font-size: 0.75rem;
    background: #eee; border: 1px solid #ccc; border-radius: 3px; cursor: pointer;
  }
  .range-inputs { display: flex; align-items: center; gap: 4px; }
  .range-inputs input { width: 70px; padding: 2px 4px; font-size: 0.8rem; }
</style>
