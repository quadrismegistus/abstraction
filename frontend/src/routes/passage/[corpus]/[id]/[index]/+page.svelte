<script lang="ts">
  import { page } from '$app/state';
  import { chunkSize } from '$lib/stores';
  import PassageReader from '../../../../../components/PassageReader.svelte';

  let corpus = $derived(page.params.corpus);
  let textId = $derived(page.params.id);
  let chunkIndex = $derived(Number(page.params.index));

  // Read chunk_size from URL if present, sync to store
  $effect(() => {
    const urlChunk = page.url.searchParams.get('chunk_size');
    if (urlChunk) $chunkSize = Number(urlChunk);
  });
</script>

<div class="header">
  <a href="/text/{corpus}/{textId}">&larr; Back to trajectory</a>
  <span class="breadcrumb">{corpus} / {textId} / passage {chunkIndex} ({$chunkSize}-word chunks)</span>
</div>

<PassageReader {corpus} {textId} {chunkIndex} />

<style>
  .header {
    padding: 0.5rem 1rem; border-bottom: 1px solid #eee;
    display: flex; align-items: center; gap: 1rem;
  }
  .header a { color: #1565C0; text-decoration: none; font-size: 0.85rem; }
  .header a:hover { text-decoration: underline; }
  .breadcrumb { color: #888; font-size: 0.8rem; }
</style>
