<script lang="ts">
  import { page } from '$app/state';
  import TrajectoryChart from '../../../../components/TrajectoryChart.svelte';

  let corpus = $derived(page.params.corpus);
  let textId = $derived(page.params.id);
</script>

<div class="header">
  <a href="/arc">&larr; Back to arc</a>
</div>

<TrajectoryChart {corpus} {textId} />

<style>
  .header { padding: 0.5rem 1rem; border-bottom: 1px solid #eee; }
  .header a { color: #1565C0; text-decoration: none; font-size: 0.85rem; }
  .header a:hover { text-decoration: underline; }
</style>
