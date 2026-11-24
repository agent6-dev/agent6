// Chapter buttons for a video with a chapters track, one per cue.
// The row is built here, so a page without scripts shows the plain player.
function chapterButtons(video) {
  const source = video.querySelector('track[kind="chapters"]');
  if (!source || video.previousElementSibling?.classList.contains('a6-chapters')) return;
  const row = document.createElement('div');
  row.className = 'a6-chapters';
  row.setAttribute('role', 'group');
  row.setAttribute('aria-label', 'Parts of this video');
  const still = matchMedia('(prefers-reduced-motion: reduce)');
  const build = () => {
    const cues = [...source.track.cues];
    // The current part is the last one begun, so the row never depends on a cue's end time.
    const mark = () => {
      const current = cues.findLast((cue) => cue.startTime <= video.currentTime);
      cues.forEach((cue, i) => {
        if (cue === current) row.children[i].setAttribute('aria-current', 'true');
        else row.children[i].removeAttribute('aria-current');
      });
    };
    for (const cue of cues) {
      const button = row.appendChild(document.createElement('button'));
      button.type = 'button';
      button.textContent = cue.text;
      button.addEventListener('click', () => {
        video.currentTime = cue.startTime;
        // Under reduced motion a press shows the part's first frame and waits for Play.
        if (!still.matches) video.play();
      });
    }
    source.track.addEventListener('cuechange', mark);
    video.addEventListener('seeked', mark);
    mark();
    video.before(row);
  };
  // A chapters track loads its cues only once its mode leaves "disabled".
  source.track.mode = 'hidden';
  if (source.readyState === HTMLTrackElement.LOADED) build();
  else source.addEventListener('load', build, { once: true });
}

// Material swaps the page body on navigation and announces each page on document$.
const eachPage = () => document.querySelectorAll('video').forEach(chapterButtons);
if (window.document$) window.document$.subscribe(eachPage);
else eachPage();
