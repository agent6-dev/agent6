"""Read the working tree with Textual's file tree and viewers."""

from __future__ import annotations

import codecs
import mimetypes
import pathlib
import stat
from collections.abc import Iterable
from typing import ClassVar

from rich import syntax
from textual import app, containers, screen, widgets

from agent6 import terminal
from agent6.ui.tui import menubar, screen_chrome
from agent6.vcs.git import files as git_files
from agent6.vcs.git import read as git_read

_MAX_BYTES = 1 << 20


class RepositoryTree(widgets.DirectoryTree):
    """Leave Git metadata and the opening ignore snapshot out of the tree."""

    def __init__(self, root: pathlib.Path, ignored: frozenset[pathlib.Path]) -> None:
        super().__init__(root, id="files-tree")
        self._ignored = ignored

    def filter_paths(self, paths: Iterable[pathlib.Path]) -> Iterable[pathlib.Path]:
        """Return entries that are neither .git nor ignored by Git."""
        return (path for path in paths if path.name != ".git" and path not in self._ignored)


class PreviewMarkdown(widgets.MarkdownViewer):
    """Keep file selection in the tree so links cannot bypass the bounded reader."""

    async def go(self, location: str | pathlib.PurePath) -> None:
        """Leave navigation to the file tree and the built-in table of contents."""
        self.notify("Select files in the tree.")


class FilesScreen(screen_chrome.ScreenChrome, screen.Screen[None]):
    """Browse the repository's working-tree files."""

    MENUS: ClassVar = (
        menubar.Menu("File", (menubar.MenuItem("Back", "close"),)),
        menubar.Menu("View", (menubar.MenuItem("Markdown / source", "toggle_source"),)),
        menubar.Menu(
            "Help",
            (
                menubar.MenuItem("Keys & actions", "help"),
                menubar.MenuItem("Command palette", "command_palette"),
            ),
        ),
    )
    FOOTER: ClassVar = (("toggle_source", "Source"), ("help", "Help"), ("close", "Back"))
    BINDINGS: ClassVar = menubar.menu_bindings("files", MENUS, footer=FOOTER)
    COMMANDS: ClassVar = screen.Screen.COMMANDS | {screen_chrome.MenuCommands}
    CSS = """
    #files-tree { width: 30%; }
    #files-preview { width: 1fr; }
    #files-note { height: auto; }
    """

    def __init__(self, root: pathlib.Path) -> None:
        super().__init__()
        self._root = (git_read.toplevel(root) or root).resolve()
        self._ignored = git_files.ignored_paths(self._root)
        self._is_markdown = False
        self._show_source = False

    def compose(self) -> app.ComposeResult:
        """Yield the tree beside the file preview, with the shared screen chrome."""
        yield menubar.MenuBar(self.MENUS)
        with containers.Horizontal():
            yield RepositoryTree(self._root, self._ignored)
            with containers.Vertical(id="files-preview"):
                yield widgets.Static("", id="files-note", markup=False)
                with widgets.ContentSwitcher(initial="files-message"):
                    yield widgets.Static("Select a file.", id="files-message", markup=False)
                    yield PreviewMarkdown(id="files-markdown", open_links=False)
                    yield widgets.RichLog(id="files-source", auto_scroll=False, min_width=0)
        yield widgets.Footer()

    def on_mount(self) -> None:
        """Name the working tree and focus its file tree."""
        self.app.sub_title = f"Files · {self._root.name}"
        self.query_one(widgets.DirectoryTree).focus()

    def on_directory_tree_directory_selected(
        self, event: widgets.DirectoryTree.DirectorySelected
    ) -> None:
        """Name the selected folder, or the tree root."""
        event.stop()
        path = event.path
        name = str(path.relative_to(self._root)) if path.is_relative_to(self._root) else str(path)
        label = terminal.printable_name(self._root.name if name == "." else name)
        self.app.sub_title = f"Files · {label}"

    async def on_directory_tree_file_selected(
        self, event: widgets.DirectoryTree.FileSelected
    ) -> None:
        """Preview at most 1 MiB as Markdown, highlighted source or file facts."""
        event.stop()
        path = event.path
        switcher = self.query_one(widgets.ContentSwitcher)
        switcher.current = "files-message"
        message = self.query_one("#files-message", widgets.Static)
        note = self.query_one("#files-note", widgets.Static)
        note.update("")
        self._is_markdown = False
        self.refresh_bindings()
        if not path.is_relative_to(self._root):
            # A directory symlink lists its real children, wherever it points.
            message.update(f"Outside the working tree: {terminal.printable_name(str(path))}")
            return
        self.app.sub_title = f"Files · {path.relative_to(self._root)}"
        try:
            info = path.stat()
            if not stat.S_ISREG(info.st_mode):
                message.update("Not a regular file.")
                return
            kind = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            if kind.startswith("image/"):
                message.update(f"Image · {kind} · {info.st_size:,} bytes")
                return
            with path.open("rb") as stream:
                data = stream.read(_MAX_BYTES + 1)
        except OSError as exc:
            message.update(f"Could not read {path.name}: {exc}")
            return
        truncated = len(data) > _MAX_BYTES
        data = data[:_MAX_BYTES]
        try:
            body = codecs.getincrementaldecoder("utf-8")().decode(data, final=not truncated)
        except UnicodeDecodeError:
            body = ""
        if b"\0" in data or (data and not body):
            message.update(f"Binary · {kind} · {info.st_size:,} bytes")
            return
        # A file is untrusted text: an escape sequence in it would reach the terminal as is.
        body = terminal.scrub_terminal_controls(body)
        if truncated:
            note.update(f"Showing first 1 MiB of {info.st_size:,} bytes.")
        source = self.query_one(widgets.RichLog)
        source.clear()
        highlighted = syntax.Syntax(body, syntax.Syntax.guess_lexer(str(path), code=body))
        source.write(highlighted.highlight(body))
        source.scroll_home(animate=False)
        self._is_markdown = path.suffix.lower() in (".md", ".markdown")
        if self._is_markdown:
            viewer = self.query_one(widgets.MarkdownViewer)
            await viewer.document.update(body)
            viewer.scroll_home(animate=False)
        switcher.current = (
            "files-markdown" if self._is_markdown and not self._show_source else "files-source"
        )
        self.refresh_bindings()

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Return None to grey out the source toggle unless the preview is Markdown."""
        if action == "toggle_source":
            return True if self._is_markdown else None
        return True

    def action_toggle_source(self) -> None:
        """Switch the selected Markdown file between its rendered view and source."""
        if not self._is_markdown:
            return
        switcher = self.query_one(widgets.ContentSwitcher)
        self._show_source = not self._show_source
        if self._show_source:
            switcher.current = "files-source"
            self.query_one(widgets.RichLog).focus()
        else:
            switcher.current = "files-markdown"
            self.query_one(widgets.MarkdownViewer).document.focus()

    def action_close(self) -> None:
        """Close a menu first, otherwise return to the hub."""
        if not self.close_open_list():
            self.app.pop_screen()
