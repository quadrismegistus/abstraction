<script lang="ts">
  import { onMount } from 'svelte';
  import favicon from '$lib/assets/favicon.svg';
  import { fetchCorpora, fetchNorms, fetchGenres } from '$lib/api';
  import { corporaList, normsList, genresList } from '$lib/stores';
  import FilterPanel from '../components/FilterPanel.svelte';

  let { children } = $props();
  let loaded = $state(false);

  onMount(async () => {
    try {
      const [corpora, norms, genres] = await Promise.all([
        fetchCorpora(), fetchNorms(), fetchGenres()
      ]);
      $corporaList = corpora;
      $normsList = norms;
      $genresList = genres;
    } catch (e) {
      console.error('Failed to load metadata:', e);
    }
    loaded = true;
  });
</script>

<svelte:head>
  <link rel="icon" href={favicon} />
  <title>Abstraction Explorer</title>
</svelte:head>

<div class="app">
  <header>
    <nav>
      <a href="/arc" class="brand">Abstraction Explorer</a>
    </nav>
  </header>

  <div class="main">
    {#if loaded}
      <FilterPanel />
    {/if}
    <main>
      {@render children()}
    </main>
  </div>
</div>

<style>
  :global(body) {
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    color: #333;
  }
  .app { display: flex; flex-direction: column; height: 100vh; }
  header {
    background: #1a1a2e; color: white; padding: 0.5rem 1rem;
    flex-shrink: 0;
  }
  nav { display: flex; align-items: center; }
  .brand {
    color: white; text-decoration: none; font-weight: 600; font-size: 1rem;
    letter-spacing: 0.5px;
  }
  .main {
    display: flex; flex: 1; min-height: 0;
  }
  main {
    flex: 1; display: flex; flex-direction: column;
    overflow: auto; padding: 0;
  }
</style>
