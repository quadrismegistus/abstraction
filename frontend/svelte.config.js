import adapter from '@sveltejs/adapter-static';
import { relative, sep } from 'node:path';

/** @type {import('@sveltejs/kit').Config} */
const config = {
	compilerOptions: {
		// defaults to rune mode for the project, except for `node_modules`. Can be removed in svelte 6.
		runes: ({ filename }) => {
			const relativePath = relative(import.meta.dirname, filename);
			const pathSegments = relativePath.toLowerCase().split(sep);
			const isExternalLibrary = pathSegments.includes('node_modules');

			return isExternalLibrary ? undefined : true;
		}
	},
	kit: {
		// Static SPA build: the app talks to the FastAPI backend at runtime, so
		// there is no Node server to deploy. `fallback` serves index.html for
		// dynamic routes (/text/[corpus]/[id], etc.) that cannot be prerendered;
		// point your static host's 404/rewrite rule at it.
		adapter: adapter({
			fallback: 'index.html'
		})
	}
};

export default config;
