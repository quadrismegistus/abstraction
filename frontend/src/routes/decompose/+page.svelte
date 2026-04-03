<script lang="ts">
  import { onMount } from 'svelte';
  import { norm, periodMatched } from '$lib/stores';

  let genre = $state('arc_fiction');
  let earlyMin = $state(1700);
  let earlyMax = $state(1780);
  let lateMin = $state(1850);
  let lateMax = $state(1950);
  let splitTarget = $state('genre_raw');  // for future: corpus, is_translated
  let loading = $state(false);
  let result: any = $state(null);
  let error = $state('');

  async function loadDecomp() {
    loading = true;
    error = '';
    try {
      const params = new URLSearchParams({
        col: $norm,
        genre,
        year_early_min: String(earlyMin),
        year_early_max: String(earlyMax),
        year_late_min: String(lateMin),
        year_late_max: String(lateMax),
        invert: 'true',
      });
      if ($periodMatched) params.set('period_matched', 'true');
      const host = typeof window !== 'undefined' ? window.location.hostname : 'localhost';
      const res = await fetch(`http://${host}:1709/api/decompose/shift-share?${params.toString()}`);
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      result = await res.json();
    } catch (e: any) {
      error = e.message;
    }
    loading = false;
  }

  onMount(loadDecomp);
</script>

<div class="decompose-page">
  <div class="header">
    <a href="/arc">&larr; Back to arc</a>
    <h2>Shift-Share Decomposition</h2>
  </div>

  <div class="controls">
    <label>
      Genre:
      <select bind:value={genre} onchange={loadDecomp}>
        <option value="arc_fiction">Fiction</option>
        <option value="arc_poetry">Poetry</option>
        <option value="arc_periodical">Periodical</option>
        <option value="arc_essays">Essays</option>
      </select>
    </label>
    <label>
      Early period:
      <input type="number" bind:value={earlyMin} style="width:60px" /> –
      <input type="number" bind:value={earlyMax} style="width:60px" />
    </label>
    <label>
      Late period:
      <input type="number" bind:value={lateMin} style="width:60px" /> –
      <input type="number" bind:value={lateMax} style="width:60px" />
    </label>
    <button onclick={loadDecomp}>Compute</button>
  </div>

  {#if loading}
    <div class="loading">Computing decomposition...</div>
  {:else if error}
    <div class="error">Error: {error}</div>
  {:else if result}
    <div class="summary">
      <div class="summary-row">
        <span>Early ({result.period_early}): <strong>{result.overall_mean_early.toFixed(3)}</strong></span>
        <span>Late ({result.period_late}): <strong>{result.overall_mean_late.toFixed(3)}</strong></span>
        <span>Change: <strong>{result.overall_change.toFixed(3)}</strong></span>
      </div>
      <div class="summary-row decomp-totals">
        <span>Composition: <strong>{(result.total_composition / Math.abs(result.overall_change) * 100).toFixed(1)}%</strong> ({result.total_composition.toFixed(4)})</span>
        <span>Within-genre: <strong>{(result.total_within / Math.abs(result.overall_change) * 100).toFixed(1)}%</strong> ({result.total_within.toFixed(4)})</span>
        <span>Interaction: <strong>{(result.total_interaction / Math.abs(result.overall_change) * 100).toFixed(1)}%</strong> ({result.total_interaction.toFixed(4)})</span>
      </div>
    </div>

    <div class="table-wrapper">
      <table>
        <thead>
          <tr>
            <th>Genre (raw)</th>
            <th>N early</th>
            <th>N late</th>
            <th>Share early</th>
            <th>Share late</th>
            <th>Mean early</th>
            <th>Mean late</th>
            <th>Composition</th>
            <th>Within</th>
            <th>Interaction</th>
            <th>Total</th>
          </tr>
        </thead>
        <tbody>
          {#each result.rows as r}
            <tr>
              <td class="genre">{r.genre}</td>
              <td>{r.n_early.toLocaleString()}</td>
              <td>{r.n_late.toLocaleString()}</td>
              <td class="num">{(r.share_early * 100).toFixed(1)}%</td>
              <td class="num">{(r.share_late * 100).toFixed(1)}%</td>
              <td class="num">{r.mean_early.toFixed(3)}</td>
              <td class="num">{r.mean_late.toFixed(3)}</td>
              <td class="num" style="color: {r.composition_effect < 0 ? 'hsl(25,70%,40%)' : 'hsl(220,60%,40%)'}">{r.composition_effect.toFixed(4)}</td>
              <td class="num" style="color: {r.within_effect < 0 ? 'hsl(25,70%,40%)' : 'hsl(220,60%,40%)'}">{r.within_effect.toFixed(4)}</td>
              <td class="num">{r.interaction.toFixed(4)}</td>
              <td class="num" style="font-weight:600; color: {r.total_effect < 0 ? 'hsl(25,70%,40%)' : 'hsl(220,60%,40%)'}">{r.total_effect.toFixed(4)}</td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  {/if}
</div>

<style>
  .decompose-page { flex: 1; display: flex; flex-direction: column; overflow: hidden; padding: 0; }
  .header {
    padding: 0.5rem 1rem; border-bottom: 1px solid #eee;
    display: flex; align-items: center; gap: 1rem; flex-shrink: 0;
  }
  .header a { color: #1565C0; text-decoration: none; font-size: 0.85rem; }
  .header h2 { margin: 0; font-size: 1.1rem; }
  .controls {
    padding: 0.5rem 1rem; display: flex; flex-wrap: wrap; gap: 1rem; align-items: center;
    border-bottom: 1px solid #eee; flex-shrink: 0; font-size: 0.85rem;
  }
  .controls label { display: flex; align-items: center; gap: 4px; }
  .controls select, .controls input { font-size: 0.85rem; padding: 2px 4px; }
  .controls button {
    padding: 4px 12px; font-size: 0.85rem; background: #333; color: white;
    border: none; border-radius: 3px; cursor: pointer;
  }
  .summary {
    padding: 0.75rem 1rem; background: #f8f8f8; border-bottom: 1px solid #eee;
    font-size: 0.9rem; flex-shrink: 0;
  }
  .summary-row { display: flex; gap: 2rem; margin-bottom: 0.25rem; }
  .decomp-totals { font-size: 0.85rem; color: #555; }
  .table-wrapper { flex: 1; overflow: auto; padding: 0 1rem; }
  table { border-collapse: collapse; width: 100%; font-size: 0.8rem; }
  th {
    text-align: left; padding: 6px 8px; border-bottom: 2px solid #ddd;
    color: #555; font-weight: 600; position: sticky; top: 0; background: white;
  }
  td { padding: 5px 8px; border-bottom: 1px solid #f0f0f0; }
  .genre { font-weight: 500; }
  .num { font-family: monospace; font-size: 0.75rem; }
  .loading, .error { padding: 2rem; text-align: center; }
  .loading { color: #666; font-style: italic; }
  .error { color: #d32f2f; }
</style>
