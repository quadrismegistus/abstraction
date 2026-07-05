// SPA mode: all data comes from the FastAPI backend at runtime, so disable
// SSR and prerendering; adapter-static emits the `fallback` index.html shell.
export const ssr = false;
export const prerender = false;
