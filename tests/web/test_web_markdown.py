from __future__ import annotations

import pytest

from agent6.ui.web import markdown


def test_raw_html_is_text_and_markdown_extensions_render() -> None:
    """Repository HTML cannot create elements; tables, strikes and fences still render."""
    rendered = markdown.render(
        "<script>alert(1)</script>\n\nInline <img src=x onerror=alert(1)>\n\n"
        "| name | value |\n| --- | --- |\n| safe | ~~old~~ |\n\n"
        '```html\n<img onerror="bad">\n```\n',
        rev="abc123",
        path="docs/readme.md",
    )
    assert "<script" not in rendered and "<img" not in rendered
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in rendered
    assert "&lt;img src=x onerror=alert(1)&gt;" in rendered
    assert "<table>" in rendered and "<s>old</s>" in rendered
    assert '<pre><code class="language-html">&lt;img onerror=&quot;bad&quot;&gt;' in rendered


@pytest.mark.parametrize(
    "url",
    [
        "javascript:alert(1)",
        "JaVaScRiPt:alert(1)",
        "jav&#x61;script:alert(1)",
        "vbscript:msgbox(1)",
        "data:text/html;base64,PHNjcmlwdD4=",
        "data:image/png;base64,aGVsbG8=",
    ],
)
def test_unsafe_links_and_images_are_not_emitted(url: str) -> None:
    """Link validation applies to links and images, including raster data URLs."""
    rendered = markdown.render(f"[bad]({url})\n\n![bad]({url})", rev="HEAD", path="README.md")
    assert "<a " not in rendered and "<img " not in rendered
    assert "bad" in rendered


def test_links_and_images_keep_the_revision_and_encoded_repository_path() -> None:
    """Relative destinations resolve from the document, not the server's web root."""
    rendered = markdown.render(
        "[read](../other%20%23%25.markdown#next-step)\n\n"
        "![picture](images/a%20b.png)\n\n[contents](#read-me)\n\n"
        "[root](/README.md)\n\n[external](https://example.com/a?q=1&b=2)\n\n"
        "[outside](../../private.md)",
        rev="refs/heads/topic",
        path="docs/readme.md",
    )
    assert (
        'href="#/files/refs%2Fheads%2Ftopic/other%20%23%25.markdown?anchor=next-step"' in rendered
    )
    assert (
        'data-blob-src="/api/git/blob?rev=refs%2Fheads%2Ftopic&amp;path=docs%2Fimages%2Fa+b.png"'
        in rendered
    )
    assert 'href="#/files/refs%2Fheads%2Ftopic/docs/readme.md?anchor=read-me"' in rendered
    assert 'href="#/files/refs%2Fheads%2Ftopic/README.md"' in rendered
    assert 'href="https://example.com/a?q=1&amp;b=2" target="_blank" rel="noopener"' in rendered
    assert "<a>outside</a>" in rendered
    assert ' src="/api/' not in rendered


def test_heading_ids_follow_visible_text_and_do_not_collide() -> None:
    """Formatting does not enter heading ids, and repeated headings get distinct anchors."""
    rendered = markdown.render(
        "# Read *me* & `code`!\n\n## Read me & code!\n\n## Read me code-1\n",
        rev="WORKTREE",
        path="README.md",
    )
    assert '<h1 id="read-me-code">' in rendered
    assert '<h2 id="read-me-code-1">' in rendered
    assert '<h2 id="read-me-code-1-1">' in rendered
