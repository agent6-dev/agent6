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
    """A markdown snapshot and the remote image addresses it asks to load."""

    html: str
    images: tuple[str, ...]


def image_origin(url: str) -> str:
    """Return a CSP-safe HTTP image origin, or an empty string."""
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
    source = utils.escapeHtml(str(item.attrs["src"]))
    inline = " playsinline" if item.tag == "video" else ""
    return (
        f'<{item.tag} src="{source}" controls preload="metadata" aria-label="{label}"{inline}>'
        f"{label}</{item.tag}>"
    )


def _rewrite_link(item: token.Token, rev: str, path: str, prefix: str) -> str | None:
    image = item.type == "image"
    attribute = "src" if image else "href"
    url = str(item.attrs[attribute])
    if re.match(r"^(?:[a-zA-Z][a-zA-Z0-9+.-]*:|//)", url):
        url = "https:" + url if url.startswith("//") else url
        if image:
            item.attrs.pop("src")
            if image_origin(url):
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
        media = MEDIA_TYPES.get(target.rsplit(".", 1)[-1].lower())
        if media:
            item.type, item.tag = "media", media[0]
    else:
        href = f"#/files/{parse.quote(rev, safe='')}/{parse.quote(target, safe='/')}"
        if fragment:
            href += "?" + parse.urlencode({"anchor": parse.unquote(fragment)})
    item.attrSet(attribute, href)
    return None


def render(text: str, *, rev: str, path: str, prefix: str) -> Rendered:
    """Render repository markdown with escaped HTML and revision-aware destinations.

    Raw HTML shows as its source, never as elements: a block as a code block, a tag as code.

    Args:
        text: The complete markdown source.
        rev: The resolved revision, or WORKTREE.
        path: The repository-relative source path.
        prefix: The preview capability's repository resource path.

    Returns:
        HTML with heading ids, Files links, confined media paths and remote image addresses.
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
    for index, item in enumerate(tokens):
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
                    images.add(remote)
    return Rendered(parser.renderer.render(tokens, parser.options, {}), tuple(sorted(images)))


def document(rendered: Rendered, nonce: str, *, images_allowed: bool) -> bytes:
    """Build the sandbox document with only the owned script and nonce-authorized styles.

    Returns:
        UTF-8 HTML with the image-consent bar when remote images are blocked.
    """
    assets = resources.files(__package__)
    style = (
        assets.joinpath("styles.css").read_text()
        + assets.joinpath("styles_markdown.css").read_text()
    )
    script = assets.joinpath("markdown.js").read_text()
    blocked = ""
    if rendered.images and not images_allowed:
        addresses = "".join(f"<li>{html.escape(url)}</li>" for url in rendered.images)
        blocked = (
            '<section class="md-images" aria-label="Remote images">'
            "<p>Remote images are blocked.</p><ul>" + addresses + "</ul>"
            '<button id="load-images" type="button">Load</button></section>'
        )
    allowed = "true" if images_allowed else "false"
    return (
        '<!doctype html><html><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f'<style nonce="{nonce}">{style}</style><title>Markdown preview</title></head>'
        f'<body data-images-allowed="{allowed}">{blocked}'
        f'<main class="md-view">{rendered.html}</main>'
        f'<script nonce="{nonce}">{script}</script></body></html>'
    ).encode()
