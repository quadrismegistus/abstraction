<script lang="ts">
  import { norm, chunkSize } from '$lib/stores';
  import { fetchPassage } from '$lib/api';
  import type { PassageResponse, PassageToken } from '$lib/types';

  let { corpus, textId, chunkIndex }: {
    corpus: string; textId: string; chunkIndex: number;
  } = $props();

  let data: PassageResponse | null = $state(null);
  let loading = $state(true);
  let error = $state('');
  let hoveredToken: PassageToken | null = $state(null);

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

  function wordColor(token: PassageToken): string {
    if (token.is_punct || token.score === null) return '#444';
    if (token.score <= -1.0) {
      const intensity = Math.min(Math.abs(token.score), 3) / 3;
      return `rgba(21, 101, 192, ${0.5 + intensity * 0.5})`;  // blue
    }
    if (token.score >= 1.0) {
      const intensity = Math.min(token.score, 3) / 3;
      return `rgba(230, 81, 0, ${0.5 + intensity * 0.5})`;  // orange
    }
    return '#444';
  }

  function wordBg(token: PassageToken): string {
    if (token.is_punct || token.score === null) return 'transparent';
    if (token.is_abstract) return 'rgba(21, 101, 192, 0.1)';
    if (token.is_concrete) return 'rgba(230, 81, 0, 0.1)';
    return 'transparent';
  }

  /**
   * Whether to insert a space before this token.
   * Mirrors passages.py:_render_paragraph spacing logic:
   * - Space before words (not before punctuation)
   * - No space after hyphens/dashes
   */
  function needsSpaceBefore(tokens: PassageToken[], i: number): boolean {
    if (i === 0) return false;
    const cur = tokens[i];
    const prev = tokens[i - 1];
    // No space before punctuation
    if (cur.is_punct) return false;
    // No space after hyphens/dashes
    if (prev.is_punct && ['-', '\u2013', '\u2014'].includes(prev.text)) return false;
    return true;
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
    <div class="stats">
      <span class="stat abstract">{data.n_abstract} abstract ({total ? (data.n_abstract/total*100).toFixed(1) : 0}%)</span>
      <span class="stat concrete">{data.n_concrete} concrete ({total ? (data.n_concrete/total*100).toFixed(1) : 0}%)</span>
      <span class="stat neutral">{data.n_neutral} neutral</span>
    </div>

    <div class="passage">{#each data.tokens as token, i}{#if needsSpaceBefore(data.tokens, i)}{' '}{/if}{#if token.is_punct}<span class="punct">{token.text}</span>{:else}<span class="word" class:abstract={token.is_abstract} class:concrete={token.is_concrete} class:unscored={token.score === null} style="color: {wordColor(token)}; background: {wordBg(token)}" role="term" onmouseenter={() => hoveredToken = token} onmouseleave={() => hoveredToken = null}>{token.text}</span>{/if}{/each}</div>

    {#if hoveredToken && !hoveredToken.is_punct}
      <div class="tooltip">
        <strong>{hoveredToken.text}</strong>
        {#if hoveredToken.score !== null}
          <span>z = {hoveredToken.score.toFixed(2)}</span>
          <span class:abstract={hoveredToken.is_abstract} class:concrete={hoveredToken.is_concrete}>
            {hoveredToken.is_abstract ? 'Abstract' : hoveredToken.is_concrete ? 'Concrete' : 'Neutral'}
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
    line-height: 2.0; font-size: 1.05rem; font-family: Georgia, "Times New Roman", serif;
    max-width: 700px;
  }
  .punct { color: #444; }
  .word {
    cursor: default; border-radius: 2px;
    transition: background 0.1s;
  }
  .word:hover { outline: 2px solid rgba(0,0,0,0.3); outline-offset: 0px; }
  .word.unscored { color: #888; }
  .word.abstract { font-weight: 500; }
  .word.concrete { font-weight: 600; }
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
