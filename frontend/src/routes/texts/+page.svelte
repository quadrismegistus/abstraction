<script lang="ts">
  import { page } from '$app/state';
  import { goto } from '$app/navigation';
  import { norm, periodMatched } from '$lib/stores';
  import { fetchArcTexts } from '$lib/api';
  import type { ArcText } from '$lib/types';

  let corpus = $derived(page.url.searchParams.get('corpus') || '');
  let yearBin = $derived(Number(page.url.searchParams.get('year') || '0'));
  let binSz = $derived(Number(page.url.searchParams.get('bin_size') || '10'));
  let genre = $derived(page.url.searchParams.get('genre') || '');

  let texts: ArcText[] = $state([]);
  let total = $state(0);
  let loading = $state(true);
  let sortCol = $state('score');
  let sortAsc = $state(false);

  async function loadData() {
    loading = true;
    const params: Record<string, string | string[]> = {
      col: $norm,
      year_min: String(yearBin),
      year_max: String(yearBin + binSz - 1),
      page_size: '10000',
    };
    if (corpus) params.corpus = [corpus];
    if (genre) params.genre = [genre];
    if ($periodMatched) params.period_matched = 'true';
    const result = await fetchArcTexts(params);
    texts = result.texts;
    total = result.total;
    loading = false;
  }

  let sortedTexts = $derived(
    [...texts].sort((a, b) => {
      let va: any = a[sortCol as keyof ArcText];
      let vb: any = b[sortCol as keyof ArcText];
      if (va === null) return 1;
      if (vb === null) return -1;
      if (typeof va === 'string') return sortAsc ? va.localeCompare(vb) : vb.localeCompare(va);
      return sortAsc ? va - vb : vb - va;
    })
  );

  function toggleSort(col: string) {
    if (sortCol === col) sortAsc = !sortAsc;
    else { sortCol = col; sortAsc = col === 'title' || col === 'author'; }
  }

  $effect(() => {
    corpus; yearBin; binSz; genre; $norm;
    loadData();
  });
</script>

<div class="texts-page">
  <div class="header">
    <a href="/arc">&larr; Back to arc</a>
    <h2>{genre}{corpus ? ` in ${corpus}` : ''} ({yearBin}{binSz > 1 ? `\u2013${yearBin + binSz - 1}` : ''})</h2>
    <span class="count">{total.toLocaleString()} texts</span>
  </div>

  {#if loading}
    <div class="loading">Loading texts...</div>
  {:else}
    <div class="table-wrapper">
      <table>
        <thead>
          <tr>
            <th class="sortable" onclick={() => toggleSort('corpus')}>Corpus {sortCol === 'corpus' ? (sortAsc ? '▲' : '▼') : ''}</th>
            <th class="sortable" onclick={() => toggleSort('year')}>Year {sortCol === 'year' ? (sortAsc ? '▲' : '▼') : ''}</th>
            <th class="sortable" onclick={() => toggleSort('author')}>Author {sortCol === 'author' ? (sortAsc ? '▲' : '▼') : ''}</th>
            <th class="sortable" onclick={() => toggleSort('title')}>Title {sortCol === 'title' ? (sortAsc ? '▲' : '▼') : ''}</th>
            <th class="sortable" onclick={() => toggleSort('score')}>Score {sortCol === 'score' ? (sortAsc ? '▲' : '▼') : ''}</th>
            <th>Genre</th>
            <th>Genre (raw)</th>
            <th>Translated?</th>
          </tr>
        </thead>
        <tbody>
          {#each sortedTexts as t}
            <tr class="text-row" onclick={() => {
              // t.id is _id format: _corpus/text_id — extract corpus and text_id
              const m = t.id.match(/^_([^/]+)\/(.+)$/);
              if (m) goto(`/text/${m[1]}/${m[2]}`);
              else goto(`/text/${t.corpus}/${t.id}`);
            }}>
              <td class="corpus">{t.corpus}</td>
              <td class="year">{t.year ?? '—'}</td>
              <td class="author">{t.author ?? '—'}</td>
              <td class="title">{t.title ?? t.id}</td>
              <td class="score" style="color: {t.score !== null ? (t.score < 0 ? 'hsl(220,60%,40%)' : 'hsl(25,70%,40%)') : '#999'}">
                {t.score !== null ? t.score.toFixed(3) : '—'}
              </td>
              <td class="genre">{t.genre ?? '—'}</td>
              <td class="genre">{t.genre_raw ?? '—'}</td>
              <td class="genre">{t.is_translated === true ? 'Yes' : t.is_translated === false ? 'No' : '—'}</td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  {/if}
</div>

<style>
  .texts-page { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
  .header {
    padding: 0.5rem 1rem; border-bottom: 1px solid #eee;
    display: flex; align-items: center; gap: 1rem; flex-shrink: 0;
  }
  .header a { color: #1565C0; text-decoration: none; font-size: 0.85rem; }
  .header h2 { margin: 0; font-size: 1.1rem; }
  .count { color: #888; font-size: 0.85rem; }
  .loading { padding: 2rem; text-align: center; color: #666; font-style: italic; }
  .table-wrapper { flex: 1; overflow-y: auto; padding: 0 1rem; }
  table { border-collapse: collapse; width: 100%; font-size: 0.85rem; }
  th {
    text-align: left; padding: 6px 8px; border-bottom: 2px solid #ddd;
    color: #555; font-weight: 600; position: sticky; top: 0; background: white;
  }
  th.sortable { cursor: pointer; user-select: none; }
  th.sortable:hover { color: #222; }
  td { padding: 5px 8px; border-bottom: 1px solid #f0f0f0; }
  .text-row { cursor: pointer; }
  .text-row:hover { background: #f8f8f8; }
  .year { width: 60px; color: #666; }
  .author { max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .title { max-width: 400px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .score { font-family: monospace; font-size: 0.8rem; width: 70px; }
  .genre { color: #888; font-size: 0.8rem; }
</style>
