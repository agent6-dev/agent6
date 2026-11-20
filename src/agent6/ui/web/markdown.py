"""Render repository markdown without executable HTML or unsafe link destinations."""

from __future__ import annotations

import dataclasses
import html
import posixpath
import re
from collections.abc import Sequence
from importlib import resources
from urllib import parse

import markdown_it
from markdown_it import token
from markdown_it.common import normalize_url, utils

MEDIA_TYPES = {
    "mp3": ("audio", "audio/mpeg"),
    "mp4": ("video", "video/mp4"),
    "ogg": ("audio", "audio/ogg"),
    "wav": ("audio", "audio/wav"),
    "webm": ("video", "video/webm"),
}


@dataclasses.dataclass(frozen=True, slots=True)
class Rendered:
    """A markdown snapshot and its remote image and media addresses."""

    html: str
    images: tuple[str, ...]
    media: tuple[str, ...]


def remote_origin(url: str) -> str:
    """Return a CSP-safe HTTP resource origin, or an empty string."""
    try:
        parts = parse.urlsplit(url)
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            return ""
        if not re.fullmatch(r"[A-Za-z0-9.\-:\[\]]+", parts.netloc):
            return ""
        if parts.port is not None and not 0 < parts.port < 65536:
            return ""
    except ValueError:
        return ""
    return f"{parts.scheme}://{parts.netloc}"


def _valid_link(url: str) -> bool:
    # The library permits raster data URLs; repository markdown permits none.
    return normalize_url.validateLink(url) and not url.strip().lower().startswith("data:")


def _raw_block(_renderer: object, tokens: Sequence[token.Token], idx: int, *_: object) -> str:
    return f'<pre class="md-raw"><code>{utils.escapeHtml(tokens[idx].content)}</code></pre>\n'


def _raw_inline(_renderer: object, tokens: Sequence[token.Token], idx: int, *_: object) -> str:
    return f'<code class="md-raw">{utils.escapeHtml(tokens[idx].content)}</code>'


def _media(_renderer: object, tokens: Sequence[token.Token], idx: int, *_: object) -> str:
    item = tokens[idx]
    label = utils.escapeHtml(item.content)
    source = " ".join(
        f'{name}="{utils.escapeHtml(str(value))}"'
        for name, value in item.attrs.items()
        if name in {"src", "data-remote-src"}
    )
    inline = " playsinline" if item.tag == "video" else ""
    return (
        f'<{item.tag} {source} controls preload="metadata" aria-label="{label}"{inline}>'
        f"{label}</{item.tag}>"
    )


def _rewrite_link(item: token.Token, rev: str, path: str, prefix: str) -> str | None:
    image = item.type == "image"
    attribute = "src" if image else "href"
    url = str(item.attrs[attribute])
    if image:
        try:
            suffix = parse.unquote(parse.urlsplit(url).path).rsplit(".", 1)[-1].lower()
        except ValueError:
            suffix = ""
        media = MEDIA_TYPES.get(suffix)
        if media:
            item.type, item.tag = "media", media[0]
    if re.match(r"^(?:[a-zA-Z][a-zA-Z0-9+.-]*:|//)", url):
        url = "https:" + url if url.startswith("//") else url
        if image:
            item.attrs.pop("src")
            if remote_origin(url):
                item.attrSet("data-remote-src", url)
                return url
            item.attrSet("data-missing", "Unsupported image address.")
        else:
            item.attrSet("href", url)
            item.attrSet("target", "_blank")
            item.attrSet("rel", "noopener noreferrer")
        return None
    destination, _, fragment = url.partition("#")
    destination = parse.unquote(destination.partition("?")[0])
    target = posixpath.normpath(
        posixpath.join(posixpath.dirname(path), destination) if destination else path
    ).lstrip("/")
    if target == ".." or target.startswith("../"):
        item.attrs.pop(attribute)
        return None
    if image:
        href = prefix + parse.quote(target, safe="/")
    else:
        href = f"#/files/{parse.quote(rev, safe='')}/{parse.quote(target, safe='/')}"
        if fragment:
            href += "?" + parse.urlencode({"anchor": parse.unquote(fragment)})
    item.attrSet(attribute, href)
    return None


def _checkbox(item: token.Token, *, editable: bool) -> bool:
    """Replace a task marker in a list item's first paragraph with a labelled control.

    Returns:
        Whether the paragraph contains a checkbox.
    """
    if not item.map or not item.children or not re.match(r"^\[[ xX]\](?=\s|$)", item.content):
        return False
    first = item.children[0]
    if first.type != "text" or not first.content.startswith(item.content[:3]):
        return False
    first.content = first.content[3:]
    box = token.Token("checkbox", "input", 0)
    box.attrSet("type", "checkbox")
    box.attrSet("data-task-line", item.map[0])
    label = "".join(c.content for c in item.children if c.type in {"text", "image", "code_inline"})
    box.attrSet("aria-label", label.strip() or f"Task on line {item.map[0] + 1}")
    if item.content[1] != " ":
        box.attrSet("checked", "checked")
    if not editable:
        box.attrSet("disabled", "disabled")
    item.children.insert(0, box)
    return True


def render(text: str, *, rev: str, path: str, prefix: str) -> Rendered:
    """Render repository markdown with escaped HTML and revision-aware destinations.

    Raw HTML shows as its source, never as elements: a block as a code block, a tag as code.

    Args:
        text: The complete markdown source.
        rev: The resolved revision, or WORKTREE.
        path: The repository-relative source path.
        prefix: The preview capability's repository resource path.

    Returns:
        HTML with heading ids, Files links, confined paths and remote resource addresses.
    """
    parser = markdown_it.MarkdownIt("commonmark", {"html": True}).enable(["table", "strikethrough"])
    parser.validateLink = _valid_link
    # The parser finds raw HTML so these rules can show it as escaped source.
    parser.add_render_rule("html_block", _raw_block)
    parser.add_render_rule("html_inline", _raw_inline)
    parser.add_render_rule("media", _media)
    tokens = parser.parse(text)
    ids: dict[str, int] = {}
    images: set[str] = set()
    media: set[str] = set()
    checklist = False
    for index, item in enumerate(tokens):
        if (
            item.type == "inline"
            and index >= 2
            and tokens[index - 1].type == "paragraph_open"
            and tokens[index - 2].type == "list_item_open"
        ):
            checklist |= _checkbox(item, editable=rev == "WORKTREE")
        if item.type == "heading_open":
            title = "".join(
                child.content
                for child in tokens[index + 1].children or []
                if child.type in {"text", "code_inline", "image"}
            )
            slug = re.sub(r"\s+", "-", re.sub(r"[^\w\s-]", "", title.lower()).strip())
            slug = slug or "heading"
            heading_id = slug
            while heading_id in ids:
                ids[slug] += 1
                heading_id = f"{slug}-{ids[slug]}"
            ids[heading_id] = 0
            item.attrSet("id", heading_id)
        for child in item.children or []:
            if child.type in {"link_open", "image"}:
                remote = _rewrite_link(child, rev, path, prefix)
                if remote:
                    (media if child.type == "media" else images).add(remote)
    body = parser.renderer.render(tokens, parser.options, {})
    if checklist and rev == "WORKTREE":
        # Literal carriage returns would normalize to LF in the HTML parser.
        source = html.escape(text, quote=False).replace("\r", "&#13;")
        name = html.escape(posixpath.basename(path))
        body += f'<template data-file="{name}">{source}</template>'
    return Rendered(body, tuple(sorted(images)), tuple(sorted(media)))


def document(rendered: Rendered, nonce: str, *, remote_allowed: bool) -> bytes:
    """Build the sandbox document with only the owned script and nonce-authorized styles.

    Returns:
        UTF-8 HTML with the consent bar when remote images or media are blocked.
    """
    assets = resources.files(__package__)
    style = (
        assets.joinpath("styles.css").read_text()
        + assets.joinpath("styles_markdown.css").read_text()
    )
    script = assets.joinpath("markdown.js").read_text()
    blocked = ""
    if (rendered.images or rendered.media) and not remote_allowed:
        addresses = "".join(
            f"<li>{html.escape(url)}</li>" for url in sorted((*rendered.images, *rendered.media))
        )
        blocked = (
            '<section class="md-remote" aria-label="Remote files">'
            "<p>Remote files are blocked.</p><ul>" + addresses + "</ul>"
            '<button id="load-remote" type="button">Load</button></section>'
        )
    allowed = "true" if remote_allowed else "false"
    return (
        '<!doctype html><html><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f'<style nonce="{nonce}">{style}</style><title>Markdown preview</title></head>'
        f'<body data-remote-allowed="{allowed}">{blocked}'
        f'<main class="md-view">{rendered.html}</main>'
        f'<script nonce="{nonce}">{script}</script></body></html>'
    ).encode()
