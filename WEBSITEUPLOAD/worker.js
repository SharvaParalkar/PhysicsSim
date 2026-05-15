/**
 * Serves static assets from the Worker bundle root (/) while allowing a custom-domain
 * route such as sharvaparalkar.com/simulation/* — incoming paths are stripped of /simulation.
 * workers.dev at / is unchanged (no prefix to strip).
 */
const BASE = '/simulation';

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    let pathname = url.pathname;

    if (pathname === BASE || pathname.startsWith(BASE + '/')) {
      pathname = pathname.slice(BASE.length) || '/';
    }

    if (pathname !== '/' && !pathname.includes('.')) {
      pathname = pathname.endsWith('/') ? pathname + 'index.html' : pathname + '/index.html';
    }

    url.pathname = pathname;
    let response = await env.ASSETS.fetch(new Request(url.toString(), request));

    if (response.status === 404 && pathname !== '/index.html') {
      url.pathname = '/index.html';
      response = await env.ASSETS.fetch(new Request(url.toString(), request));
    }

    return response;
  },
};
