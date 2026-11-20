"""Serve short-lived, read-only preview capabilities for the signed-in Files page."""

from __future__ import annotations

import dataclasses
import pathlib
import re
import secrets
import threading
import time
from urllib import parse

import pydantic

from agent6 import errors
from agent6.ui.web import git_model

DENY_CSP = (
    "sandbox; default-src 'none'; frame-ancestors 'self'; form-action 'none'; "
    "base-uri 'none'; object-src 'none'; script-src 'none'"
)
REFUSAL = "Previews open inside the Files page. Previews need https or localhost."
LEASE_SECONDS = 30
_RESOURCE_TYPES = {
    "css": ("style", "text/css; charset=utf-8"),
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


class PreviewKeyBody(pydantic.BaseModel):
    """The signed-in UI renews or closes one preview capability."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)
    key: str = pydantic.Field(description="The key returned when this preview opened.")


@dataclasses.dataclass(frozen=True, slots=True)
class Preview:
    """One file's revision, read permission and UI-owned lease."""

    rev: str
    path: str
    ignored: bool
    expires: float


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

    def open(self, body: OpenPreviewBody) -> dict[str, str]:
        """Pin the revision and check the document through the Files blob reader.

        Attributes:
            key: The page's random read capability.
            url: Its frame address.
            rev: The resolved commit, or WORKTREE.
            kind: The preview document kind.

        Returns:
            The frame address and its lease key.

        Raises:
            PreviewError: The path is hidden, unsupported, or the preview limit is reached.
        """
        _visible(body.path)
        if pathlib.PurePosixPath(body.path).suffix.lower() not in {".htm", ".html"}:
            raise PreviewError("this file has no preview")
        rev = str(git_model.tree_payload(self.repo, body.rev, ignored=body.ignored)["rev"])
        git_model.blob_payload(self.repo, rev, body.path, ignored=body.ignored)
        key = secrets.token_urlsafe(32)
        now = time.monotonic()
        with self._lock:
            self._pages = {k: p for k, p in self._pages.items() if p.expires > now}
            if len(self._pages) >= 128:
                raise PreviewError("close a preview before opening another", 429)
            self._pages[key] = Preview(rev, body.path, body.ignored, now + LEASE_SECONDS)
        return {
            "key": key,
            "url": f"/preview/{key}/{parse.quote(body.path, safe='/')}",
            "rev": rev,
            "kind": "html",
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

    def _get(self, key: str) -> Preview:
        page = self._pages.get(key)
        if page is None or page.expires <= time.monotonic():
            self._pages.pop(key, None)
            raise PreviewError("This preview is closed. Open it again in Files.", 403)
        return page

    def read(self, route: str, destination: str, host: str) -> Response:
        """Return confined bytes only for an allowed browser fetch destination.

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
            "style",
            "track",
            "video",
        }:
            raise PreviewError(REFUSAL, 403)
        key, _, path = route.removeprefix("/preview/").partition("/")
        with self._lock:
            page = self._get(key)
        _visible(path)
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
        allowed = sources(host, f"/preview/{key}/")
        policy = (
            f"{DENY_CSP}; style-src 'unsafe-inline' {allowed}; img-src {allowed}; "
            f"font-src {allowed}; media-src {allowed}; frame-src 'none'; connect-src 'none'"
        )
        return Response(blob.data, content_type, policy)


READ_ERRORS = (PreviewError, git_model.GitRequestError, errors.GitError)
