from __future__ import annotations

import asyncio
import pathlib
import subprocess

import pytest
from textual import pilot as textual_pilot
from textual import widgets

from agent6.ui.tui import files, home, menubar
from tests.tui import waits


@pytest.fixture
def repo(tmp_path: pathlib.Path) -> pathlib.Path:
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    return root


@pytest.mark.parametrize("close_key", ["escape", "q"])
def test_files_opens_from_the_hub_and_returns(repo: pathlib.Path, close_key: str) -> None:
    """F opens the working tree from the hub; Esc and q return to the hub."""

    async def scenario() -> None:
        app = home.Agent6HomeApp(repo.parent / "state", repo)
        async with app.run_test(size=(160, 40)) as pilot:
            await pilot.press("F")
            await pilot.pause()
            assert type(app.screen).__name__ == "FilesScreen"
            tree = app.screen.query_one(widgets.DirectoryTree)
            assert pathlib.Path(tree.path) == repo
            assert app.screen.query(menubar.MenuBar)
            assert app.screen.query(widgets.Footer)
            await pilot.press(close_key)
            await pilot.pause()
            assert isinstance(app.screen, home.HomeScreen)

    asyncio.run(scenario())


async def _select(pilot: textual_pilot.Pilot[pathlib.Path | None], name: str) -> None:
    tree = pilot.app.screen.query_one(widgets.DirectoryTree)
    node = tree.root
    for part in pathlib.Path(name).parts:
        await waits.wait_for(
            pilot,
            lambda node=node, part=part: any(str(child.label) == part for child in node.children),
            f"tree entry {part}",
        )
        node = next(child for child in node.children if str(child.label) == part)
        tree.move_cursor(node)
        tree.focus()
        await pilot.press("enter")
        await pilot.pause()


def _source_text(source: widgets.RichLog) -> str:
    return "\n".join(line.text for line in source.lines)


def test_files_shortcut_is_in_the_hub_footer_and_help(repo: pathlib.Path) -> None:
    """The hub advertises the Files key through its shared menu registry."""

    async def scenario() -> None:
        app = home.Agent6HomeApp(repo.parent / "state", repo)
        async with app.run_test(size=(160, 40)) as pilot:
            await pilot.pause()
            binding = app.screen.active_bindings.get("F")
            assert binding is not None
            assert binding.binding.show
            assert binding.binding.description == "Files"
            await pilot.press("question_mark")
            await pilot.pause()
            assert isinstance(app.screen, menubar.HelpScreen)
            rows = [str(widget.content) for widget in app.screen.query(widgets.Static)]
            assert any("Open files" in row and row.endswith("F") for row in rows)

    asyncio.run(scenario())


@pytest.mark.parametrize("suffix", [".md", ".markdown"])
def test_markdown_renders_a_heading_and_toggles_source(repo: pathlib.Path, suffix: str) -> None:
    """Markdown opens without an outline; s switches to literal source and back."""
    name = f"README{suffix}"
    body = "# Workspace guide\n\nRead **this** plan.\n"
    (repo / name).write_text(body, encoding="utf-8")

    async def scenario() -> None:
        app = home.Agent6HomeApp(repo.parent / "state", repo)
        async with app.run_test(size=(120, 36)) as pilot:
            await pilot.press("F")
            await _select(pilot, name)
            assert app.screen.query(widgets.MarkdownViewer)
            viewer = app.screen.query_one(widgets.MarkdownViewer)
            assert viewer.is_on_screen
            assert not viewer.show_table_of_contents
            assert not viewer.table_of_contents.is_on_screen
            assert viewer.document.region.x == viewer.region.x
            headings = viewer.document.query("MarkdownH1")
            assert headings
            heading = headings.first()
            assert "Workspace guide" in "".join(
                heading.render_line(y).text for y in range(heading.size.height)
            )
            source = app.screen.query_one(widgets.RichLog)
            assert not source.is_on_screen
            await pilot.press("s")
            await pilot.pause()
            assert source.is_on_screen and not viewer.is_on_screen
            assert "# Workspace guide" in _source_text(source)
            assert "**this**" in _source_text(source)
            source.focus()
            await pilot.press("s")
            await pilot.pause()
            assert viewer.is_on_screen and not source.is_on_screen

    asyncio.run(scenario())


def test_subtitle_follows_a_file_then_a_folder_and_root(repo: pathlib.Path) -> None:
    """Selecting a folder cannot leave the previous file's name in the subtitle."""
    (repo / "docs").mkdir()
    (repo / "example.py").write_text("answer = 42\n", encoding="utf-8")

    async def scenario() -> None:
        app = home.Agent6HomeApp(repo.parent / "state", repo)
        async with app.run_test(size=(120, 36)) as pilot:
            await pilot.press("F")
            assert app.sub_title == "Files · repo"
            await _select(pilot, "example.py")
            assert app.sub_title == "Files · example.py"
            await _select(pilot, "docs")
            assert app.sub_title == "Files · docs"
            tree = app.screen.query_one(widgets.DirectoryTree)
            tree.move_cursor(tree.root)
            await pilot.press("enter")
            await pilot.pause()
            assert app.sub_title == "Files · repo"

    asyncio.run(scenario())


def test_python_displays_highlighted_source(repo: pathlib.Path) -> None:
    """A Python file paints syntax colours instead of rendering as Markdown."""
    (repo / "example.py").write_text("def answer():\n    return 42\n", encoding="utf-8")

    async def scenario() -> None:
        app = home.Agent6HomeApp(repo.parent / "state", repo)
        async with app.run_test(size=(120, 36)) as pilot:
            await pilot.press("F")
            await _select(pilot, "example.py")
            assert app.screen.query(widgets.RichLog)
            source = app.screen.query_one(widgets.RichLog)
            assert source.is_on_screen
            assert "def answer():" in _source_text(source)
            assert "return 42" in _source_text(source)
            colours = {
                segment.style.color
                for line in source.lines
                for segment in line
                if segment.text.strip() and segment.style
            }
            assert len(colours) > 1
            await pilot.press("s")
            assert source.is_on_screen

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("name", "data", "kind"),
    [
        ("data.bin", b"abc\0xyz", "Binary"),
        ("invalid.dat", b"\xff\xfe\x01", "Binary"),
        ("image.png", b"\x89PNG\r\n\x1a\n", "image/png"),
        ("diagram.svg", b"<svg xmlns='http://www.w3.org/2000/svg'/>", "image/svg+xml"),
    ],
)
def test_binary_and_images_show_size_and_kind(
    repo: pathlib.Path, name: str, data: bytes, kind: str
) -> None:
    """Non-text previews show file facts rather than undecodable or image source bytes."""
    (repo / name).write_bytes(data)

    async def scenario() -> None:
        app = home.Agent6HomeApp(repo.parent / "state", repo)
        async with app.run_test(size=(120, 36)) as pilot:
            await pilot.press("F")
            await _select(pilot, name)
            text = "\n".join(str(w.content) for w in app.screen.query(widgets.Static))
            assert f"{len(data)} bytes" in text
            assert kind in text

    asyncio.run(scenario())


def test_large_text_shows_only_the_first_mib(repo: pathlib.Path) -> None:
    """The byte cap leaves a note and does not mistake a cut UTF-8 character for binary."""
    prefix = ("x" * 1023 + "\n") * 1023 + "y" * 1023
    (repo / "large.txt").write_text(prefix + "éNOT IN PREVIEW", encoding="utf-8")

    async def scenario() -> None:
        app = home.Agent6HomeApp(repo.parent / "state", repo)
        async with app.run_test(size=(120, 36)) as pilot:
            await pilot.press("F")
            await _select(pilot, "large.txt")
            text = "\n".join(str(w.content) for w in app.screen.query(widgets.Static))
            assert "first 1 MiB" in text
            source = app.screen.query_one(widgets.RichLog)
            assert source.is_on_screen
            shown = _source_text(source)
            assert "x" * 1023 in shown and "y" * 1023 in shown
            assert "NOT IN PREVIEW" not in shown

    asyncio.run(scenario())


def test_git_metadata_and_ignored_paths_are_absent(repo: pathlib.Path) -> None:
    """The tree honours ignored directories, nested rules, negations and quoted names."""
    (repo / ".gitignore").write_text("cache/\n*.tmp\n!keep.tmp\n", encoding="utf-8")
    (repo / "cache").mkdir()
    (repo / "cache" / "hidden.md").write_text("hidden", encoding="utf-8")
    (repo / "docs").mkdir()
    (repo / "docs" / ".gitignore").write_text("private.md\n", encoding="utf-8")
    for name in (
        "drop.tmp",
        "space é.tmp",
        "keep.tmp",
        "README.md",
        "docs/private.md",
        "docs/ok.md",
    ):
        (repo / name).write_text("visible or ignored", encoding="utf-8")
    (repo / ".git" / "info" / "exclude").write_text("local.md\n", encoding="utf-8")
    (repo / "local.md").write_text("ignored locally", encoding="utf-8")

    async def scenario() -> None:
        app = home.Agent6HomeApp(repo.parent / "state", repo)
        async with app.run_test(size=(120, 36)) as pilot:
            await pilot.press("F")
            tree = app.screen.query_one(widgets.DirectoryTree)
            await waits.wait_for(pilot, lambda: bool(tree.root.children), "root entries")
            names = {str(child.label) for child in tree.root.children}
            assert names == {".gitignore", "docs", "keep.tmp", "README.md"}
            await _select(pilot, "docs/ok.md")
            docs = next(child for child in tree.root.children if str(child.label) == "docs")
            assert {str(child.label) for child in docs.children} == {".gitignore", "ok.md"}

    asyncio.run(scenario())


def test_files_uses_the_project_root_from_a_subdirectory(repo: pathlib.Path) -> None:
    """Launching the hub below the root still browses the whole working tree."""
    subdir = repo / "docs"
    subdir.mkdir()

    async def scenario() -> None:
        app = home.Agent6HomeApp(repo.parent / "state", subdir)
        async with app.run_test() as pilot:
            await pilot.press("F")
            assert pathlib.Path(app.screen.query_one(widgets.DirectoryTree).path) == repo

    asyncio.run(scenario())


def test_file_read_failure_is_shown(repo: pathlib.Path) -> None:
    """A vanished tree entry reports the read failure instead of keeping a stale preview."""
    path = repo / "vanished.txt"
    path.write_text("gone", encoding="utf-8")

    async def scenario() -> None:
        app = home.Agent6HomeApp(repo.parent / "state", repo)
        async with app.run_test() as pilot:
            await pilot.press("F")
            tree = app.screen.query_one(widgets.DirectoryTree)
            await waits.wait_for(pilot, lambda: bool(tree.root.children), "root entries")
            path.unlink()
            await _select(pilot, path.name)
            text = "\n".join(str(w.content) for w in app.screen.query(widgets.Static))
            assert "Could not read" in text and "vanished.txt" in text

    asyncio.run(scenario())


def test_files_refuses_a_failed_git_listing(tmp_path: pathlib.Path) -> None:
    """A Git listing failure stays on the hub and reports its reason."""

    async def scenario() -> None:
        app = home.Agent6HomeApp(tmp_path / "state", tmp_path)
        async with app.run_test() as pilot:
            await pilot.press("F")
            await pilot.pause()
            assert isinstance(app.screen, home.HomeScreen)
            assert "Could not open files" in "\n".join(n.message for n in app._notifications)

    asyncio.run(scenario())


def test_markdown_links_do_not_replace_the_selected_file(
    repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Markdown links cannot bypass the tree's selected file and bounded reader."""
    target = repo / "other.md"
    target.write_text("# Unselected\n", encoding="utf-8")
    (repo / "README.md").write_text(f"# Selected\n\n[Other]({target})\n", encoding="utf-8")

    async def scenario() -> None:
        app = home.Agent6HomeApp(repo.parent / "state", repo)
        opened: list[str] = []
        monkeypatch.setattr(app, "open_url", opened.append)
        async with app.run_test(size=(120, 36)) as pilot:
            await pilot.press("F")
            await _select(pilot, "README.md")
            assert await pilot.click("MarkdownParagraph", offset=(1, 0))
            await pilot.pause()
            assert not opened
            viewer = app.screen.query_one(widgets.MarkdownViewer)
            assert viewer.document.query("MarkdownH1")
            heading = viewer.document.query("MarkdownH1").first()
            assert "Selected" in "".join(
                heading.render_line(y).text for y in range(heading.size.height)
            )

    asyncio.run(scenario())


def test_terminal_controls_in_a_file_never_reach_the_viewer(repo: pathlib.Path) -> None:
    """A file carrying an OSC 52 sequence cannot write the clipboard when it is opened."""
    (repo / "notes.txt").write_text(
        "before \x1b]52;c;SGVsbG8=\x1b\\ after \x1b[31mred\x1b[0m\n", encoding="utf-8"
    )

    async def scenario() -> None:
        app = home.Agent6HomeApp(repo.parent / "state", repo)
        async with app.run_test(size=(120, 36)) as pilot:
            await pilot.press("F")
            await _select(pilot, "notes.txt")
            text = _source_text(app.screen.query_one(widgets.RichLog))
            assert "\x1b" not in text
            assert "before" in text and "after" in text and "red" in text

    asyncio.run(scenario())


def test_a_file_behind_a_directory_symlink_out_of_the_tree_is_refused(repo: pathlib.Path) -> None:
    """Textual lists a directory symlink's real children; one outside the root must not crash."""
    outside = repo.parent / "outside"
    outside.mkdir()
    (outside / "readme.txt").write_text("elsewhere\n", encoding="utf-8")
    (repo / "link").symlink_to(outside, target_is_directory=True)

    async def scenario() -> None:
        app = home.Agent6HomeApp(repo.parent / "state", repo)
        async with app.run_test(size=(120, 36)) as pilot:
            await pilot.press("F")
            await _select(pilot, "link/readme.txt")
            assert type(app.screen).__name__ == "FilesScreen"
            message = app.screen.query_one("#files-message", widgets.Static)
            assert "Outside the working tree" in str(message.content)
            assert "elsewhere" not in _source_text(app.screen.query_one(widgets.RichLog))

    asyncio.run(scenario())


def test_a_newly_selected_file_opens_at_its_top(repo: pathlib.Path) -> None:
    """The source viewer forgets the previous file's scroll offset."""
    for name in ("first.py", "second.py"):
        (repo / name).write_text("".join(f"x{i} = {i}\n" for i in range(300)), encoding="utf-8")

    async def scenario() -> None:
        app = home.Agent6HomeApp(repo.parent / "state", repo)
        async with app.run_test(size=(120, 36)) as pilot:
            await pilot.press("F")
            await _select(pilot, "first.py")
            source = app.screen.query_one(widgets.RichLog)
            source.scroll_to(y=100, animate=False)
            await pilot.pause()
            assert source.scroll_y == 100
            await _select(pilot, "second.py")
            assert source.scroll_y == 0

    asyncio.run(scenario())


def test_markdown_choice_survives_file_selection(repo: pathlib.Path) -> None:
    """The Files screen keeps its source choice across Markdown and plain files."""
    for name in ("one.md", "two.md", "plain.txt"):
        (repo / name).write_text(f"# {name}\n", encoding="utf-8")

    async def scenario() -> None:
        app = home.Agent6HomeApp(repo.parent / "state", repo)
        async with app.run_test(size=(120, 36)) as pilot:
            await pilot.press("F")
            await _select(pilot, "one.md")
            switcher = app.screen.query_one(widgets.ContentSwitcher)
            assert switcher.current == "files-markdown"
            await pilot.press("s")
            await pilot.pause()
            assert switcher.current == "files-source"
            for name in ("two.md", "plain.txt", "one.md"):
                await _select(pilot, name)
                assert switcher.current == "files-source"
            await pilot.press("s")
            await pilot.pause()
            assert switcher.current == "files-markdown"
            await _select(pilot, "two.md")
            assert switcher.current == "files-markdown"

    asyncio.run(scenario())


@pytest.mark.parametrize("size", [(80, 24), (140, 40)])
@pytest.mark.parametrize("name", ["README.md", "example.py"])
def test_tree_toggle_gives_the_file_full_width_and_restores_selection(
    repo: pathlib.Path, size: tuple[int, int], name: str
) -> None:
    """Folding the tree focuses the full-width file without losing selection or scroll."""
    (repo / name).write_text("# File contents\n\n" + "Read this file.\n\n" * 100, encoding="utf-8")

    async def scenario() -> None:
        app = home.Agent6HomeApp(repo.parent / "state", repo)
        async with app.run_test(size=size) as pilot:
            await pilot.press("F")
            await _select(pilot, name)
            tree = app.screen.query_one(widgets.DirectoryTree)
            selected = tree.cursor_node
            preview = app.screen.query_one("#files-preview")
            initial_width = preview.size.width
            assert 0 < initial_width < size[0]
            reader = (
                app.screen.query_one(widgets.MarkdownViewer)
                if name.endswith(".md")
                else app.screen.query_one(widgets.RichLog)
            )
            focus = reader.document if isinstance(reader, widgets.MarkdownViewer) else reader
            await pilot.press("t")
            await pilot.pause()
            assert not tree.display
            assert preview.region.x == 0 and preview.size.width == size[0]
            assert reader.region.x == 0 and reader.size.width == size[0]
            assert focus.region.x == 0
            assert app.focused is focus
            await pilot.press("pagedown")
            await pilot.pause()
            assert reader.scroll_y > 0
            offset = reader.scroll_y
            await pilot.press("t")
            await pilot.pause()
            assert tree.display and app.focused is tree
            assert tree.cursor_node is selected
            assert preview.size.width == initial_width
            await pilot.press("t")
            await pilot.pause()
            assert app.focused is focus
            assert reader.scroll_y == offset
            assert preview.size.width == size[0]

    asyncio.run(scenario())


@pytest.mark.parametrize("name", [None, "image.png"])
def test_tree_toggle_works_without_a_text_preview(repo: pathlib.Path, name: str | None) -> None:
    """An empty or image preview can fold the tree and restore keyboard navigation."""
    (repo / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    async def scenario() -> None:
        app = home.Agent6HomeApp(repo.parent / "state", repo)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.press("F")
            if name is not None:
                await _select(pilot, name)
            message = app.screen.query_one("#files-message", widgets.Static)
            content = str(message.content)
            await pilot.press("t")
            await pilot.pause()
            tree = app.screen.query_one(widgets.DirectoryTree)
            assert not tree.display
            assert message.is_on_screen and str(message.content) == content
            assert message.region.x == 0 and message.size.width == 80
            await pilot.press("t")
            await pilot.pause()
            assert tree.display and app.focused is tree

    asyncio.run(scenario())


def test_outline_is_optional_and_keeps_its_choice_across_files(repo: pathlib.Path) -> None:
    """The outline toggles only on rendered Markdown and keeps its choice across previews."""
    for name in ("one.md", "two.md", "plain.txt"):
        (repo / name).write_text("# First\n\n" + "Text\n\n" * 40 + "## Last\n", encoding="utf-8")

    async def scenario() -> None:
        app = home.Agent6HomeApp(repo.parent / "state", repo)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.press("F")
            await _select(pilot, "one.md")
            viewer = app.screen.query_one(widgets.MarkdownViewer)
            await pilot.press("o")
            await pilot.pause()
            assert viewer.table_of_contents.is_on_screen
            outline = viewer.table_of_contents.query_one(widgets.Tree)
            outline.focus()
            await pilot.press("end", "enter")
            await pilot.pause()
            assert viewer.scroll_y > 0
            await pilot.press("o")
            await pilot.pause()
            assert not viewer.show_table_of_contents
            assert not viewer.table_of_contents.is_on_screen
            assert app.focused is viewer.document
            await pilot.press("o", "s")
            await pilot.pause()
            assert not app.screen.active_bindings["o"].enabled
            await pilot.press("o", "s")
            await pilot.pause()
            assert viewer.table_of_contents.is_on_screen
            for name in ("plain.txt", "two.md"):
                await _select(pilot, name)
                assert app.screen.active_bindings["o"].enabled == name.endswith(".md")
                if name.endswith(".txt"):
                    await pilot.press("o")
            assert viewer.table_of_contents.is_on_screen
            await pilot.press("o")
            await _select(pilot, "one.md")
            assert not viewer.table_of_contents.is_on_screen

    asyncio.run(scenario())


def test_files_sidebar_keys_are_in_the_footer_and_help(repo: pathlib.Path) -> None:
    """Both sidebar keys remain discoverable at 80 columns through the menu registry."""
    (repo / "README.md").write_text("# Guide\n", encoding="utf-8")

    async def scenario() -> None:
        app = home.Agent6HomeApp(repo.parent / "state", repo)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.press("F")
            await _select(pilot, "README.md")
            footer = app.screen._compositor.render_strips()[-1].text
            assert "t Tree" in footer and "o Outline" in footer
            await pilot.press("question_mark")
            await pilot.pause()
            assert isinstance(app.screen, menubar.HelpScreen)
            rows = [str(widget.content) for widget in app.screen.query(widgets.Static)]
            assert any("Show / hide tree" in row and row.endswith("t") for row in rows)
            assert any("Show / hide outline" in row and row.endswith("o") for row in rows)

    asyncio.run(scenario())


@pytest.mark.parametrize("size", [(80, 24), (140, 40)])
def test_the_tree_is_as_wide_as_its_names_up_to_a_limit(
    repo: pathlib.Path, size: tuple[int, int]
) -> None:
    """A fixed share of the screen left short names beside empty columns the file could use."""
    (repo / "README.md").write_text("# Guide\n", encoding="utf-8")

    async def scenario() -> None:
        app = home.Agent6HomeApp(repo.parent / "state", repo)
        async with app.run_test(size=size) as pilot:
            await pilot.press("F")
            await _select(pilot, "README.md")
            tree = app.screen.query_one(widgets.DirectoryTree)
            assert tree.region.width == len("├── 📄 README.md") + 2
            (repo / ("long-name-" * 8 + ".txt")).write_text("x\n", encoding="utf-8")
            await tree.reload()
            await pilot.pause()
            assert tree.region.width == size[0] * 2 // 5
            assert app.screen.query_one("#files-preview").region.width == size[0] * 3 // 5

    asyncio.run(scenario())


def test_the_file_starts_under_the_menu_and_a_cut_file_says_so_above_it(
    repo: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty note held a blank row above every file."""
    monkeypatch.setattr(files, "_MAX_BYTES", 128)
    (repo / "README.md").write_text("# Guide\n", encoding="utf-8")
    (repo / "big.txt").write_text("line\n" * 30, encoding="utf-8")

    async def scenario() -> None:
        app = home.Agent6HomeApp(repo.parent / "state", repo)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.press("F")
            await _select(pilot, "README.md")
            preview = app.screen.query_one("#files-preview")
            note = app.screen.query_one("#files-note", widgets.Static)
            assert app.screen.query_one(widgets.MarkdownViewer).region.y == preview.region.y
            await _select(pilot, "big.txt")
            source = app.screen.query_one(widgets.RichLog)
            assert note.region.y == preview.region.y and note.region.height == 1
            assert source.region.y == preview.region.y + 1
            await _select(pilot, "README.md")
            assert app.screen.query_one(widgets.MarkdownViewer).region.y == preview.region.y

    asyncio.run(scenario())
