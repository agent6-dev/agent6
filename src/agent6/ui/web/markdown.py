"""Render repository markdown without executable HTML or unsafe link destinations."""

from __future__ import annotations

import posixpath
import re
from urllib import parse

import markdown_it
from markdown_it import token
from markdown_it.common import normalize_url


def _valid_link(url: str) -> bool:
    # The library permits raster data URLs; repository markdown permits none.
    return normalize_url.validateLink(url) and not url.strip().lower().startswith("data:")


def _rewrite_link(item: token.Token, rev: str, path: str, ignored: bool) -> None:
    image = item.type == "image"
    attribute = "src" if image else "href"
    url = str(item.attrs[attribute])
    if re.match(r"^(?:[a-zA-Z][a-zA-Z0-9+.-]*:|//)", url):
        if not image and url.lower().startswith(("http:", "https:", "//")):
            item.attrSet("target", "_blank")
            item.attrSet("rel", "noopener")
        return
    destination, _, fragment = url.partition("#")
    destination = parse.unquote(destination.partition("?")[0])
    target = posixpath.normpath(
        posixpath.join(posixpath.dirname(path), destination) if destination else path
    ).lstrip("/")
    if target == ".." or target.startswith("../"):
        item.attrs.pop(attribute)
        return
    if image:
        href = "/api/git/blob?" + parse.urlencode({"rev": rev, "path": target})
        if ignored:
            href += "&ignored=1"
        item.attrs.pop("src")
        attribute = "data-blob-src"
    else:
        href = f"#/files/{parse.quote(rev, safe='')}/{parse.quote(target, safe='/')}"
        if fragment:
            href += "?" + parse.urlencode({"anchor": parse.unquote(fragment)})
    item.attrSet(attribute, href)


def render(text: str, *, rev: str, path: str, ignored: bool = False) -> str:
    """Render repository markdown with escaped HTML and revision-aware destinations.

    Args:
        text: The complete markdown source.
        rev: The resolved revision, or WORKTREE.
        path: The repository-relative source path.
        ignored: Permit ignored working-tree images through the blob route.

    Returns:
        HTML with heading ids and Files links whose anchor query names a heading id.
        Relative images carry data-blob-src for the client's authenticated blob fetch.
    """
    parser = markdown_it.MarkdownIt("commonmark", {"html": False}).enable(
        ["table", "strikethrough"]
    )
    parser.validateLink = _valid_link
    tokens = parser.parse(text)
    ids: dict[str, int] = {}
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
                _rewrite_link(child, rev, path, ignored)
    return parser.renderer.render(tokens, parser.options, {})
