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
  let genreRaw = $derived(page.url.searchParams.get('genre_raw') || '');

  const PAGE_SIZE = 10000;

  let texts: ArcText[] = $state([]);
  let total = $state(0);
  let pageNum = $state(0);
  let loading = $state(true);
  let sortCol = $state('score');
  let sortAsc = $state(false);
  let loadSeq = 0;

  let totalPages = $derived(Math.max(1, Math.ceil(total / PAGE_SIZE)));
  let rangeStart = $derived(total === 0 ? 0 : pageNum * PAGE_SIZE + 1);
  let rangeEnd = $derived(pageNum * PAGE_SIZE + texts.length);

  async function loadData(p: number) {
    const seq = ++loadSeq;
    loading = true;
    const params: Record<string, string | string[]> = {
      col: $norm,
      year_min: String(yearBin),
      year_max: String(yearBin + binSz - 1),
      page: String(p),
      page_size: String(PAGE_SIZE),
    };
    if (corpus) params.corpus = [corpus];
    if (genre) params.genre = [genre];
    if (genreRaw) params.genre_raw = genreRaw;
    if ($periodMatched) params.period_matched = 'true';
    const result = await fetchArcTexts(params);
    if (seq !== loadSeq) return; // a newer request superseded this one
    texts = result.texts;
    total = result.total;
    loading = false;
  }

  function gotoPage(p: number) {
    if (p < 0 || p >= totalPages) return;
    pageNum = p;
    loadData(p);
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

  function openText(t: ArcText) {
    // t.id is _id format: _corpus/text_id — extract corpus and text_id
    const m = t.id.match(/^_([^/]+)\/(.+)$/);
    if (m) goto(`/text/${m[1]}/${m[2]}`);
    else goto(`/text/${t.corpus}/${t.id}`);
  }

  // Enter/Space activation for elements given role="button"/"link"
  function activateOnKey(e: KeyboardEvent, fn: () => void) {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      fn();
    }
  }

  $effect(() => {
    corpus; yearBin; binSz; genre; genreRaw; $norm; $periodMatched;
    pageNum = 0;
    loadData(0);
  });
</script>

<div class="texts-page">
  <div class="header">
    <a href="/arc">&larr; Back to arc</a>
    <h2>{genre}{genreRaw ? ` \u2014 ${genreRaw}` : ''}{corpus ? ` in ${corpus}` : ''} ({yearBin}{binSz > 1 ? `\u2013${yearBin + binSz - 1}` : ''})</h2>
    {#if totalPages > 1}
      <span class="count">
        showing {rangeStart.toLocaleString()}&ndash;{rangeEnd.toLocaleString()} of {total.toLocaleString()}
        &mdash; page {pageNum + 1}/{totalPages} (sort applies within page)
      </span>
      <div class="pager">
        <button disabled={pageNum === 0 || loading} onclick={() => gotoPage(pageNum - 1)}>&larr; Prev</button>
        <button disabled={pageNum >= totalPages - 1 || loading} onclick={() => gotoPage(pageNum + 1)}>Next &rarr;</button>
      </div>
    {:else}
      <span class="count">{total.toLocaleString()} texts</span>
    {/if}
  </div>

  {#if loading}
    <div class="loading">Loading texts...</div>
  {:else}
    <div class="table-wrapper">
      <table>
        <thead>
          <tr>
            <th class="sortable" role="button" tabindex="0" onclick={() => toggleSort('corpus')} onkeydown={(e) => activateOnKey(e, () => toggleSort('corpus'))}>Corpus {sortCol === 'corpus' ? (sortAsc ? '▲' : '▼') : ''}</th>
            <th class="sortable" role="button" tabindex="0" onclick={() => toggleSort('year')} onkeydown={(e) => activateOnKey(e, () => toggleSort('year'))}>Year {sortCol === 'year' ? (sortAsc ? '▲' : '▼') : ''}</th>
            <th class="sortable" role="button" tabindex="0" onclick={() => toggleSort('author')} onkeydown={(e) => activateOnKey(e, () => toggleSort('author'))}>Author {sortCol === 'author' ? (sortAsc ? '▲' : '▼') : ''}</th>
            <th class="sortable" role="button" tabindex="0" onclick={() => toggleSort('title')} onkeydown={(e) => activateOnKey(e, () => toggleSort('title'))}>Title {sortCol === 'title' ? (sortAsc ? '▲' : '▼') : ''}</th>
            <th class="sortable" role="button" tabindex="0" onclick={() => toggleSort('score')} onkeydown={(e) => activateOnKey(e, () => toggleSort('score'))}>Score {sortCol === 'score' ? (sortAsc ? '▲' : '▼') : ''}</th>
            <th class="sortable" role="button" tabindex="0" onclick={() => toggleSort('n_versions')} onkeydown={(e) => activateOnKey(e, () => toggleSort('n_versions'))}>Versions {sortCol === 'n_versions' ? (sortAsc ? '▲' : '▼') : ''}</th>
            <th class="sortable" role="button" tabindex="0" onclick={() => toggleSort('score_sd')} onkeydown={(e) => activateOnKey(e, () => toggleSort('score_sd'))}>SD {sortCol === 'score_sd' ? (sortAsc ? '▲' : '▼') : ''}</th>
            <th>Genre</th>
            <th>Genre (raw)</th>
            <th>Genre source</th>
            <th>Translated?</th>
          </tr>
        </thead>
        <tbody>
          {#each sortedTexts as t}
            <tr
              class="text-row"
              role="link"
              tabindex="0"
              onclick={() => openText(t)}
              onkeydown={(e) => activateOnKey(e, () => openText(t))}
            >
              <td class="corpus">{t.corpus}</td>
              <td class="year">{t.year ?? '—'}</td>
              <td class="author">{t.author ?? '—'}</td>
              <td class="title">{t.title ?? t.id}</td>
              <td class="score" style="color: {t.score !== null ? (t.score < 0 ? 'hsl(220,60%,40%)' : 'hsl(25,70%,40%)') : '#999'}">
                {t.score !== null ? t.score.toFixed(3) : '—'}
              </td>
              <td class="versions">{t.n_versions ?? '—'}</td>
              <td class="sd">{t.score_sd !== null && t.score_sd !== undefined ? t.score_sd.toFixed(3) : '—'}</td>
              <td class="genre">{t.genre ?? '—'}</td>
              <td class="genre">{t.genre_raw ?? '—'}</td>
              <td class="genre">{t.genre_enriched_source ?? '—'}</td>
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
  .pager { display: flex; gap: 4px; }
  .pager button {
    padding: 2px 10px; font-size: 0.8rem; background: #fff; color: #333;
    border: 1px solid #ccc; border-radius: 3px; cursor: pointer;
  }
  .pager button:hover:not(:disabled) { background: #f0f0f0; }
  .pager button:disabled { color: #bbb; cursor: default; }
  .loading { padding: 2rem; text-align: center; color: #666; font-style: italic; }
  .table-wrapper { flex: 1; overflow-y: auto; padding: 0 1rem; }
  table { border-collapse: collapse; width: 100%; font-size: 0.85rem; }
  th {
    text-align: left; padding: 6px 8px; border-bottom: 2px solid #ddd;
    color: #555; font-weight: 600; position: sticky; top: 0; background: white;
  }
  th.sortable { cursor: pointer; user-select: none; }
  th.sortable:hover { color: #222; }
  th.sortable:focus-visible { outline: 2px solid #1565C0; outline-offset: -2px; color: #222; }
  td { padding: 5px 8px; border-bottom: 1px solid #f0f0f0; }
  .text-row { cursor: pointer; }
  .text-row:hover { background: #f8f8f8; }
  .text-row:focus-visible { outline: 2px solid #1565C0; outline-offset: -2px; background: #f8f8f8; }
  .year { width: 60px; color: #666; }
  .author { max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .title { max-width: 400px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .score { font-family: monospace; font-size: 0.8rem; width: 70px; }
  .versions { font-family: monospace; font-size: 0.8rem; width: 40px; color: #666; text-align: center; }
  .sd { font-family: monospace; font-size: 0.8rem; width: 60px; color: #888; }
  .genre { color: #888; font-size: 0.8rem; }
</style>
