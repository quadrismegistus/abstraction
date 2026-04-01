<script lang="ts">
  import { norm, chunkSize } from '$lib/stores';
  import { fetchPassage } from '$lib/api';
  import type { PassageResponse, ScoredWord } from '$lib/types';

  let { corpus, textId, chunkIndex }: {
    corpus: string; textId: string; chunkIndex: number;
  } = $props();

  let data: PassageResponse | null = $state(null);
  let loading = $state(true);
  let error = $state('');
  let hoveredWord: ScoredWord | null = $state(null);

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

  function wordColor(score: number | null): string {
    if (score === null) return '#888';
    if (score <= -1.0) {
      const intensity = Math.min(Math.abs(score), 3) / 3;
      return `rgba(21, 101, 192, ${0.4 + intensity * 0.6})`;  // blue
    }
    if (score >= 1.0) {
      const intensity = Math.min(score, 3) / 3;
      return `rgba(230, 81, 0, ${0.4 + intensity * 0.6})`;  // orange
    }
    return '#444';  // neutral
  }

  function wordBg(score: number | null): string {
    if (score === null) return 'transparent';
    if (score <= -1.0) return 'rgba(21, 101, 192, 0.08)';
    if (score >= 1.0) return 'rgba(230, 81, 0, 0.08)';
    return 'transparent';
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
    <div class="stats">
      {#if data.words.length > 0}
        {@const abs = data.words.filter(w => w.is_abstract).length}
        {@const conc = data.words.filter(w => w.is_concrete).length}
        {@const total = data.words.length}
        <span class="stat abstract">{abs} abstract ({(abs/total*100).toFixed(1)}%)</span>
        <span class="stat concrete">{conc} concrete ({(conc/total*100).toFixed(1)}%)</span>
        <span class="stat neutral">{total - abs - conc} neutral</span>
      {/if}
    </div>

    <div class="passage">
      {#each data.words as word}
        <span
          class="word"
          class:abstract={word.is_abstract}
          class:concrete={word.is_concrete}
          class:unscored={word.score === null}
          style="color: {wordColor(word.score)}; background: {wordBg(word.score)}"
          onmouseenter={() => hoveredWord = word}
          onmouseleave={() => hoveredWord = null}
        >{word.word}</span>{' '}
      {/each}
    </div>

    {#if hoveredWord}
      <div class="tooltip">
        <strong>{hoveredWord.word}</strong>
        {#if hoveredWord.score !== null}
          <span>z = {hoveredWord.score.toFixed(2)}</span>
          <span class:abstract={hoveredWord.is_abstract} class:concrete={hoveredWord.is_concrete}>
            {hoveredWord.is_abstract ? 'Abstract' : hoveredWord.is_concrete ? 'Concrete' : 'Neutral'}
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
  .stats {
    display: flex; gap: 1rem; margin-bottom: 1rem; padding: 0.5rem;
    background: #f5f5f5; border-radius: 4px; font-size: 0.85rem;
  }
  .stat.abstract { color: #1565C0; }
  .stat.concrete { color: #E65100; }
  .stat.neutral { color: #666; }
  .passage {
    line-height: 1.8; font-size: 1.05rem; font-family: Georgia, serif;
  }
  .word {
    cursor: default; padding: 1px 2px; border-radius: 2px;
    transition: background 0.1s;
  }
  .word:hover { outline: 2px solid rgba(0,0,0,0.3); outline-offset: 1px; }
  .word.unscored { color: #bbb; }
  .tooltip {
    position: fixed; bottom: 1rem; right: 1rem;
    background: white; border: 1px solid #ddd; border-radius: 6px;
    padding: 0.5rem 0.75rem; box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    display: flex; gap: 0.75rem; align-items: center; font-size: 0.85rem;
    z-index: 100;
  }
  .tooltip .abstract { color: #1565C0; font-weight: 600; }
  .tooltip .concrete { color: #E65100; font-weight: 600; }
  .tooltip .unscored { color: #999; font-style: italic; }
  .loading, .error { padding: 2rem; text-align: center; }
  .loading { color: #666; font-style: italic; }
  .error { color: #d32f2f; }
</style>
