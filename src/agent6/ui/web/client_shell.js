"use strict";
const view = document.getElementById('view');
const crumb = document.getElementById('crumb');

const el = (t, cls, txt) => { const e = document.createElement(t); if (cls) e.className = cls; if (txt != null) e.textContent = txt; return e; };
const esc = s => (s == null ? '' : String(s));
function withCode(text) {
  const frag = document.createDocumentFragment();
  String(text).split(/`([^`\n]+)`/).forEach((part, i) => { if (part) frag.append(i % 2 ? el('code', null, part) : part); });
  return frag;
}

function firstLine(s, n) { const t = String(s == null ? '' : s).split('\n')[0]; return t.length > n ? t.slice(0, n - 1) + '…' : t; }

// Removing an override lets a later system theme change apply.
const systemDark = matchMedia('(prefers-color-scheme: dark)');
function themeIsDark() {
  const c = document.documentElement.classList;
  return c.contains('dark') || (!c.contains('light') && systemDark.matches);
}
function toggleTheme() {
  const next = themeIsDark() ? 'light' : 'dark';
  document.documentElement.classList.remove('light', 'dark');
  if (next === (systemDark.matches ? 'dark' : 'light')) forget('a6-theme');
  else { document.documentElement.classList.add(next); remember('a6-theme', next); }
  syncTheme();
}
function syncTheme() {
  const c = document.documentElement.classList;
  const forced = c.contains('dark') ? 'dark' : c.contains('light') ? 'light' : '';
  for (const m of document.querySelectorAll('meta[name=theme-color]')) {
    m.content = (forced || (m.media.includes('dark') ? 'dark' : 'light')) === 'dark' ? '#0e1116' : '#f6f8fa';
  }
  const name = themeIsDark() ? 'Switch to light theme' : 'Switch to dark theme';
  for (const b of document.querySelectorAll('.theme-toggle')) { b.setAttribute('aria-label', name); b.title = name; }
}
for (const b of document.querySelectorAll('.theme-toggle')) b.onclick = toggleTheme;
systemDark.addEventListener('change', syncTheme);
syncTheme();

const railToggle = document.getElementById('rail-toggle');
const railShow = document.getElementById('rail-show');
const railRoom = matchMedia('(width >= 64em)');
const RAIL_PRESS = { labels: 'Collapse the sidebar', icons: 'Hide the sidebar', hidden: 'Show the sidebar' };
function railState() {
  const c = document.documentElement.classList;
  return c.contains('rail-hidden') ? 'hidden' : c.contains('rail-min') || !railRoom.matches ? 'icons' : 'labels';
}
function syncRail() {
  const state = railState();
  for (const b of [railToggle, railShow]) {
    b.setAttribute('aria-expanded', String(state === 'labels'));
    b.setAttribute('aria-label', RAIL_PRESS[state]);
    b.title = RAIL_PRESS[state];
  }
}
function cycleRail() {
  const next = { labels: 'min', icons: 'hidden', hidden: '' }[railState()];
  const pressed = document.activeElement;
  document.documentElement.classList.remove('rail-min', 'rail-hidden');
  if (next) document.documentElement.classList.add('rail-' + next);
  remember('a6-rail', next);
  syncRail();
  // Hiding the active rail button must not lose keyboard focus.
  if (pressed === railToggle && next === 'hidden') railShow.focus();
  else if (pressed === railShow) railToggle.focus();
}
railToggle.onclick = railShow.onclick = cycleRail;
railRoom.addEventListener('change', syncRail);
syncRail();

function setCrumb(t, statePill) {
  crumb.replaceChildren(...(statePill ? [statePill] : []), el('span', null, t || ''));
  const words = statePill ? statePill.textContent + ' · ' + t : t;
  document.title = words ? words + ' · agent6' : 'agent6';
}

function pill(level, label) { return el('span', 'pill ' + esc(level || 'neutral'), esc(label)); }

function actionable(elem, activate, label) {
  elem.setAttribute('role', 'button');
  elem.tabIndex = 0;
  if (label) elem.setAttribute('aria-label', label);
  elem.onclick = activate;
  elem.onkeydown = (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); activate(); }
  };
}

function setTab(name) {
  let label = '';
  for (const a of document.querySelectorAll('nav.tabs a, aside.rail .rail-nav a')) {
    if (a.dataset.tab === name) { a.setAttribute('aria-current', 'page'); label = a.title; }
    else a.removeAttribute('aria-current');
  }
  setCrumb(label);
}
