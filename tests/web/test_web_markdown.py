from __future__ import annotations

import pytest

from agent6.ui.web import markdown


@pytest.mark.parametrize(
    ("suffix", "tag"),
    [("mp4", "video"), ("webm", "video"), ("mp3", "audio"), ("wav", "audio"), ("ogg", "audio")],
)
def test_repository_media_uses_native_controls(suffix: str, tag: str) -> None:
    """Image syntax for repository media becomes a labelled player, never autoplay."""
    rendered = markdown.render(
        f'![A & B "clip"](media/a%20b.{suffix.upper()})',
        rev="abc123",
        path="docs/guide.md",
        prefix="/preview/key/",
    )
    assert f'<{tag} src="/preview/key/docs/media/a%20b.{suffix.upper()}"' in rendered.html
    assert 'controls preload="metadata"' in rendered.html
    assert 'aria-label="A &amp; B &quot;clip&quot;"' in rendered.html
    assert ("playsinline" in rendered.html) == (tag == "video")
    assert f"</{tag}>" in rendered.html
    assert "autoplay" not in rendered.html and "<img" not in rendered.html
    assert rendered.images == ()


def test_raw_html_is_text_and_markdown_extensions_render() -> None:
    """Repository HTML cannot create elements; tables, strikes and fences still render."""
    rendered = markdown.render(
        "<script>alert(1)</script>\n\nInline <img src=x onerror=alert(1)>\n\n"
        "| name | value |\n| --- | --- |\n| safe | ~~old~~ |\n\n"
        '```html\n<img onerror="bad">\n```\n',
        rev="abc123",
        path="docs/readme.md",
        prefix="/preview/key/",
    ).html
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
    rendered = markdown.render(
        f"[bad]({url})\n\n![bad]({url})", rev="HEAD", path="README.md", prefix="/preview/key/"
    ).html
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
        prefix="/preview/key/",
    ).html
    assert (
        'href="#/files/refs%2Fheads%2Ftopic/other%20%23%25.markdown?anchor=next-step"' in rendered
    )
    assert 'src="/preview/key/docs/images/a%20b.png"' in rendered
    assert 'href="#/files/refs%2Fheads%2Ftopic/docs/readme.md?anchor=read-me"' in rendered
    assert 'href="#/files/refs%2Fheads%2Ftopic/README.md"' in rendered
    assert (
        'href="https://example.com/a?q=1&amp;b=2" target="_blank" rel="noopener noreferrer"'
        in rendered
    )
    assert "<a>outside</a>" in rendered
    assert ' src="/api/' not in rendered


def test_heading_ids_follow_visible_text_and_do_not_collide() -> None:
    """Formatting does not enter heading ids, and repeated headings get distinct anchors."""
    rendered = markdown.render(
        "# Read *me* & `code`!\n\n## Read me & code!\n\n## Read me code-1\n",
        rev="WORKTREE",
        path="README.md",
        prefix="/preview/key/",
    ).html
    assert '<h1 id="read-me-code">' in rendered
    assert '<h2 id="read-me-code-1">' in rendered
    assert '<h2 id="read-me-code-1-1">' in rendered


@pytest.mark.parametrize(
    "url",
    [
        "https://user:password@example.test/a",
        "https://a.test:99999/a",
        "https://a.test;script-src/a",
        "https://[bad]/a",
        "ftp://example.test/a",
    ],
)
def test_invalid_image_origins_cannot_become_policy_sources(url: str) -> None:
    """CSP delimiters, credentials and unsupported image schemes cannot widen a grant."""
    assert markdown.image_origin(url) == ""


def test_remote_images_are_listed_without_a_fetching_attribute() -> None:
    """The renderer records full addresses but gives only repository images a source."""
    rendered = markdown.render(
        "![A](https://images.test/a.png?text=one&more=two)\n\n![B](//other.test/b.svg)\n\n![A again](https://images.test/a.png?text=one&more=two)",
        rev="WORKTREE",
        path="docs/plan.md",
        prefix="/preview/key/",
    )
    assert rendered.images == (
        "https://images.test/a.png?text=one&more=two",
        "https://other.test/b.svg",
    )
    assert 'data-remote-src="https://images.test/a.png?text=one&amp;more=two"' in rendered.html
    assert " src=" not in rendered.html


def test_raw_html_shows_as_its_source() -> None:
    """Raw HTML ran into the prose as text; a block reads as source and a tag as code."""
    rendered = markdown.render(
        "Text\n\n<table>\n  <tr><td>cell</td></tr>\n</table>\n\nA <b>bold</b> word\n",
        rev="HEAD",
        path="README.md",
        prefix="/preview/key/",
    ).html
    raw = '<code class="md-raw">'
    assert (
        '<pre class="md-raw"><code>&lt;table&gt;\n'
        "  &lt;tr&gt;&lt;td&gt;cell&lt;/td&gt;&lt;/tr&gt;\n&lt;/table&gt;\n</code></pre>"
        in rendered
    )
    assert f"<p>A {raw}&lt;b&gt;</code>bold{raw}&lt;/b&gt;</code> word</p>" in rendered
