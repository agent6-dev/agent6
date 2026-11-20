'use strict';
// This script receives only view state from its parent, never code or markup.
window.addEventListener('message', event => {
  if (event.source !== parent || event.data?.type !== 'preview-state') return;
  const {theme, anchor} = event.data;
  document.documentElement.className = theme === 'light' || theme === 'dark' ? theme : '';
  if (typeof anchor === 'string') {
    const heading = document.getElementById(anchor);
    if (heading) heading.scrollIntoView(); else window.scrollTo(0, 0);
  }
});
document.getElementById('load-remote')?.addEventListener('click', () => {
  parent.postMessage({type: 'preview-remote'}, '*');
});
document.addEventListener('click', event => {
  const link = event.target.closest('a[href]');
  if (!link) return;
  event.preventDefault();
  parent.postMessage({type: 'preview-link', href: link.getAttribute('href')}, '*');
});
if (document.body.dataset.remoteAllowed === 'true') {
  for (const player of document.querySelectorAll('video[data-remote-src], audio[data-remote-src]')) {
    player.src = player.dataset.remoteSrc;
  }
}
for (const image of document.querySelectorAll('.md-view img')) {
  const missing = reason => {
    const box = document.createElement('span'); box.className = 'md-missing';
    const name = document.createElement('span'); name.textContent = image.alt || 'image';
    const detail = document.createElement('span'); detail.className = 'muted'; detail.textContent = reason;
    box.append(name, detail); image.replaceWith(box);
  };
  if (image.dataset.missing) { missing(image.dataset.missing); continue; }
  if (image.dataset.remoteSrc && document.body.dataset.remoteAllowed !== 'true') {
    missing('Remote image blocked.'); continue;
  }
  const failed = async () => {
    let reason = 'The browser cannot show this image.';
    if (!image.dataset.remoteSrc) {
      try {
        const response = await fetch(image.src);
        if (!response.ok) reason = await response.text();
      } catch (error) { reason = error.message; }
    }
    missing(reason);
  };
  image.addEventListener('error', failed, {once: true});
  if (image.dataset.remoteSrc) image.src = image.dataset.remoteSrc;
  else if (image.complete && !image.naturalWidth) failed();
}
