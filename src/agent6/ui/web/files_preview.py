"""Serve short-lived file previews and hold form submissions for the signed-in Files page."""

from __future__ import annotations

import dataclasses
import json
import pathlib
import re
import secrets
import threading
import time
from typing import Any
from urllib import parse

import pydantic

from agent6 import errors
from agent6.ui.web import files_edit, git_model, markdown

_BOUNDARY = "default-src 'none'; frame-ancestors 'self'; base-uri 'none'; object-src 'none'"
DENY_CSP = f"sandbox; {_BOUNDARY}; form-action 'none'; script-src 'none'"
REFUSAL = "Previews open inside the Files page. Previews need https or localhost."
LEASE_SECONDS = 30
_RESOURCE_TYPES = {
    **markdown.MEDIA_TYPES,
    "css": ("style", "text/css; charset=utf-8"),
    "js": ("script", "text/javascript; charset=utf-8"),
    "mjs": ("script", "text/javascript; charset=utf-8"),
    "otf": ("font", "font/otf"),
    "ttf": ("font", "font/ttf"),
    "woff": ("font", "font/woff"),
    "woff2": ("font", "font/woff2"),
}


class PreviewError(Exception):
    """A preview request cannot be served."""

    def __init__(self, message: str, status: int = 422) -> None:
        super().__init__(message)
        self.status = status


class OpenPreviewBody(pydantic.BaseModel):
    """The signed-in UI opens a file at the revision its Files tree shows."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)
    rev: str = pydantic.Field(description="The Files revision, or WORKTREE.")
    path: str = pydantic.Field(description="The repository-relative file to preview.")
    ignored: bool = pydantic.Field(default=False, description="Include ignored working-tree files.")
    digest: str = pydantic.Field(
        default="", description="The displayed digest, required for markdown."
    )


class PreviewKeyBody(pydantic.BaseModel):
    """The signed-in UI renews, closes or grants permission to one preview."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)
    key: str = pydantic.Field(description="The key returned when this preview opened.")


@dataclasses.dataclass(frozen=True, slots=True)
class Preview:
    """One file's revision, read permission and UI-owned lease."""

    rev: str
    path: str
    ignored: bool
    expires: float
    rendered: markdown.Rendered | None = None
    remote_allowed: bool = False
    scripts_allowed: bool = False
    pending: str | None = None

    def accepts_form(self, path: str) -> bool:
        """Return whether this is the opened working-tree HTML document."""
        return self.rendered is None and self.rev == "WORKTREE" and self.path == path


@dataclasses.dataclass(frozen=True, slots=True)
class Response:
    """A preview response and its document policy."""

    data: bytes
    content_type: str
    csp: str


def sources(host: str, prefix: str) -> str:
    """Return CSP sources scoped to one path on the HTTP or HTTPS request authority."""
    if not re.fullmatch(r"[A-Za-z0-9.\-:\[\]]+", host):
        return "'none'"
    return f"http://{host}{prefix} https://{host}{prefix}"


def _visible(path: str) -> None:
    if any(not part or part.startswith(".") for part in path.split("/")):
        raise PreviewError("previews refuse hidden or escaping paths")


class Store:
    """Keep preview capabilities on the server until the signed-in UI closes them."""

    def __init__(self, repo: pathlib.Path) -> None:
        self.repo = repo
        self._pages: dict[str, Preview] = {}
        self._lock = threading.Lock()

    def open(self, body: OpenPreviewBody) -> dict[str, Any]:
        """Pin the revision and check the document through the Files blob reader.

        Args:
            body: The file, revision and displayed digest the signed-in UI asks to preview.

        Attributes:
            key: The page's random read capability.
            url: Its frame address.
            rev: The resolved commit, or WORKTREE.
            kind: The preview document kind.
            images: The full remote image addresses in this markdown snapshot.
            media: The full remote video and audio addresses in this markdown snapshot.

        Returns:
            The frame address and its lease key.

        Raises:
            PreviewError: The path is hidden, unsupported, or the preview limit is reached.
        """
        _visible(body.path)
        suffix = pathlib.PurePosixPath(body.path).suffix.lower()
        if suffix not in {".htm", ".html", ".md", ".markdown"}:
            raise PreviewError("this file has no preview")
        rev = str(git_model.tree_payload(self.repo, body.rev, ignored=body.ignored)["rev"])
        key = secrets.token_urlsafe(32)
        rendered = None
        if suffix in {".md", ".markdown"}:
            text = git_model.text_payload(
                self.repo, rev, body.path, whole=True, ignored=body.ignored
            )
            if text["binary"]:
                raise PreviewError("this file is binary")
            if text["digest"] != body.digest:
                raise PreviewError("The file changed. Reopen it in Files.", 409)
            rendered = markdown.render(
                text["text"], rev=rev, path=body.path, prefix=f"/preview/{key}/"
            )
        else:
            git_model.blob_payload(self.repo, rev, body.path, ignored=body.ignored)
        now = time.monotonic()
        with self._lock:
            self._pages = {k: p for k, p in self._pages.items() if p.expires > now}
            if len(self._pages) >= 128:
                raise PreviewError("close a preview before opening another", 429)
            self._pages[key] = Preview(rev, body.path, body.ignored, now + LEASE_SECONDS, rendered)
        return {
            "key": key,
            "url": f"/preview/{key}/{parse.quote(body.path, safe='/')}",
            "rev": rev,
            "kind": "markdown" if rendered is not None else "html",
            "images": list(rendered.images) if rendered is not None else [],
            "media": list(rendered.media) if rendered is not None else [],
        }

    def close(self, key: str) -> None:
        """Revoke one capability, including its resources."""
        with self._lock:
            self._pages.pop(key, None)

    def renew(self, key: str) -> None:
        """Extend an unexpired lease only for the signed-in UI."""
        with self._lock:
            page = self._get(key)
            self._pages[key] = dataclasses.replace(page, expires=time.monotonic() + LEASE_SECONDS)

    def allow_remote(self, key: str) -> None:
        """Grant only the remote origins captured when the signed-in UI opened the page.

        Args:
            key: The preview's key.

        Raises:
            PreviewError: The key is closed or the preview is not markdown.
        """
        with self._lock:
            page = self._get(key)
            if page.rendered is None:
                raise PreviewError("Only markdown previews can load remote files.")
            self._pages[key] = dataclasses.replace(page, remote_allowed=True)

    def _get(self, key: str) -> Preview:
        page = self._pages.get(key)
        if page is None or page.expires <= time.monotonic():
            self._pages.pop(key, None)
            raise PreviewError("This preview is closed. Open it again in Files.", 403)
        return page

    def allow_scripts(self, key: str) -> None:
        """Grant scripts to this HTML page until it closes.

        Args:
            key: The preview's key.

        Raises:
            PreviewError: The preview is closed or is not HTML.
        """
        with self._lock:
            page = self._get(key)
            if page.rendered is not None:
                raise PreviewError("Only HTML previews can run page scripts.")
            self._pages[key] = dataclasses.replace(page, scripts_allowed=True)

    def read(self, route: str, destination: str, host: str) -> Response:
        """Return confined bytes only for an allowed browser fetch destination.

        Args:
            route: The request path, `/preview/<key>/<file>`.
            destination: The request's `Sec-Fetch-Dest` header.
            host: The request's `Host` header, which scopes the policy's sources.

        Returns:
            The resource bytes, media type and content policy.

        Raises:
            PreviewError: The capability, path or fetch destination is refused.
        """
        if destination not in {
            "audio",
            "empty",
            "font",
            "iframe",
            "image",
            "script",
            "style",
            "track",
            "video",
        }:
            raise PreviewError(REFUSAL, 403)
        key, _, path = route.removeprefix("/preview/").partition("/")
        with self._lock:
            page = self._get(key)
        _visible(path)
        if destination == "script" and not page.scripts_allowed:
            raise PreviewError(REFUSAL, 403)
        parent, slash, _ = page.path.rpartition("/")
        folder = parent + slash if page.scripts_allowed else ""
        if not path.startswith(folder) or (
            page.scripts_allowed and destination == "iframe" and path != page.path
        ):
            raise PreviewError("This grant serves only the open page and files in its folder.", 403)
        allowed = sources(host, f"/preview/{key}/{parse.quote(folder)}")
        if destination == "iframe" and page.rendered is not None and path == page.path:
            nonce = secrets.token_urlsafe(32)
            remote = {"img": "", "media": ""}
            if page.remote_allowed:
                for kind, urls in (("img", page.rendered.images), ("media", page.rendered.media)):
                    remote[kind] = " ".join(sorted({markdown.remote_origin(url) for url in urls}))
            policy = (
                f"sandbox allow-scripts; {_BOUNDARY}; form-action 'none'; frame-src 'none'; "
                f"script-src 'nonce-{nonce}'; style-src 'nonce-{nonce}'; "
                f"img-src {allowed} {remote['img']}; connect-src {allowed}; "
                f"media-src {allowed} {remote['media']}"
            )
            return Response(
                markdown.document(page.rendered, nonce, remote_allowed=page.remote_allowed),
                "text/html; charset=utf-8",
                policy,
            )
        extension = pathlib.PurePosixPath(path).suffix.lower().lstrip(".")
        if destination == "iframe":
            if extension not in {"htm", "html"}:
                raise PreviewError("this file has no preview", 403)
            content_type = "text/html; charset=utf-8"
        elif destination == "empty":
            content_type = "text/plain; charset=utf-8"
        elif destination == "image" and extension in git_model.IMAGE_TYPES:
            content_type = git_model.IMAGE_TYPES[extension]
        else:
            expected, content_type = _RESOURCE_TYPES.get(extension, ("", ""))
            if expected != destination:
                raise PreviewError("this resource type is not allowed", 403)
        blob = git_model.blob_payload(self.repo, page.rev, path, ignored=page.ignored)
        document = destination == "iframe"
        scripts = document and page.scripts_allowed
        forms = document and page.accepts_form(path)
        sandbox = (
            "sandbox" + (" allow-forms" if forms else "") + (" allow-scripts" if scripts else "")
        )
        action = sources(host, parse.quote(route)) if forms else "'none'"
        code = f"'unsafe-inline' {allowed}" if scripts else "'none'"
        connections = allowed if scripts else "'none'"
        policy = (
            f"{sandbox}; {_BOUNDARY}; form-action {action}; script-src {code}; "
            f"style-src 'unsafe-inline' {allowed}; img-src {allowed}; font-src {allowed}; "
            f"media-src {allowed}; frame-src 'none'; worker-src 'none'; connect-src {connections}"
        )
        return Response(blob.data, content_type, policy)

    def submit(self, route: str, data: bytes) -> None:
        """Keep one pending form without reading or writing the answers file.

        Args:
            route: The request path, `/preview/<key>/<file>`.
            data: The URL-encoded form body.

        Raises:
            PreviewError: The form is not for this page or its fields are invalid.
        """
        key, _, path = route.removeprefix("/preview/").partition("/")
        with self._lock:
            page = self._get(key)
            if not page.accepts_form(path):
                raise PreviewError("Only the open working-tree HTML page can submit answers.", 403)
            try:
                fields = parse.parse_qs(
                    data.decode("utf-8"),
                    keep_blank_values=True,
                    strict_parsing=True,
                    errors="strict",
                    max_num_fields=1024,
                )
            except ValueError as exc:
                raise PreviewError("Use at most 1024 UTF-8 form fields.", 400) from exc
            answers = {k: v[0] if len(v) == 1 else v for k, v in fields.items()}
            text = json.dumps(answers, ensure_ascii=False)
            self._pages[key] = dataclasses.replace(page, pending=text)

    def take_answers(self, key: str) -> dict[str, Any]:
        """Return the pending Edit proposal once, only to the signed-in UI.

        Args:
            key: The preview's key.

        Attributes:
            pending: A sibling path, the JSON text and its current disk digest, or null.

        Returns:
            The proposal, or a null `pending` when no form was submitted.
        """
        with self._lock:
            page = self._get(key)
            if page.pending is None:
                return {"pending": None}
            self._pages[key] = dataclasses.replace(page, pending=None)
        path = pathlib.PurePosixPath(page.path).with_suffix(".answers.json").as_posix()
        digest = files_edit.read(self.repo, path)["digest"]
        return {"pending": {"path": path, "text": page.pending, "digest": digest}}


READ_ERRORS = (PreviewError, git_model.GitRequestError, errors.GitError)
