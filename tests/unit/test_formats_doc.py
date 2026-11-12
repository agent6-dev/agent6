from __future__ import annotations

import importlib.util
import inspect
import pathlib
import re
import sys
import types

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _generator() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("gen_formats", _ROOT / "docs/gen_formats.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_formats_page_is_not_stale() -> None:
    """The committed public reference matches its serialized producers."""
    generated = _generator().build_markdown()
    assert generated == (_ROOT / "docs/formats.md").read_text(encoding="utf-8"), (
        "Regenerate with: uv run python docs/gen_formats.py"
    )


@pytest.mark.parametrize(
    ("heading", "producer"),
    (
        ("`logs.jsonl` event envelope", "sessions.event_log.EventSink.emit"),
        ("`sessions list --json` row", "viewmodel.listing.summary_row"),
        ("`sessions list --json` group fields", "viewmodel.listing.row_json"),
        ("`sessions show --json`", "ui.cli.sessions_show._cmd_status"),
        (
            "`attach --json` approval and commit fields",
            "viewmodel.state_view.session_state_as_dict",
        ),
        ("`/api/meta` payload", "ui.web.server._Handler._route"),
        ("`/api/git/diff` payload", "ui.web.git_model.diff_payload"),
        ("`agent6 ps` JSON row", "agent6.ui.cli.ps_cmd._Row"),
        ("Token + cost summary", "budget.BudgetTracker.format_summary"),
    ),
)
def test_surface_headings_keep_the_producer_on_the_source_line(heading: str, producer: str) -> None:
    """Readers find a surface heading without losing its producer's source link."""
    page = _generator().build_markdown()
    marker = f"### {heading}\n\n"
    assert marker in page
    section = page.split(marker, 1)[1].split("\n##", 1)[0]
    assert (
        f"Source: [`{producer}`](https://github.com/agent6-dev/agent6/blob/master/src/" in section
    )


def test_transcript_reference_lists_the_written_keys(tmp_path: pathlib.Path) -> None:
    """The transcript table follows the writer, including its call and request ids."""
    import json

    from agent6 import kinds
    from agent6.providers import provider_types

    path = provider_types.TranscriptSink(tmp_path, objects_dir=tmp_path / "objects").record(
        request_headers={},
        request_body={},
        response_status=200,
        response_body={},
        call_context=kinds.CallContext(),
        request_id="attempt",
    )
    page = _generator().build_markdown()
    assert "## Provider transcripts\n" in page
    section = page.split("## Provider transcripts\n", 1)[1].split("\n## ", 1)[0]
    assert "**Support: durable but private.**" in section
    assert all(f"| `{key}` |" in section for key in json.loads(path.read_text()))


def test_source_links_name_the_tree_not_the_import_location(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CI tests the installed wheel: every source link still names `src/agent6/`."""

    def elsewhere(obj: object) -> str:
        return "/elsewhere/site-packages/agent6/x.py"

    monkeypatch.setattr(inspect, "getfile", elsewhere)
    page = _generator().build_markdown()
    links = re.findall(r"\(https://github\.com/agent6-dev/agent6/blob/master/([^)]*)\)", page)
    assert links and all(link.startswith("src/agent6/") for link in links)


def test_type_text_spells_a_private_module_by_its_public_name() -> None:
    """Python 3.13 prints `pathlib._local.Path`; the page spells `pathlib.Path` on every Python."""
    assert _generator()._type_text("pathlib._local.Path | None") == "pathlib.Path \\| None"


def test_event_meanings_come_from_the_records(monkeypatch: pytest.MonkeyPatch) -> None:
    """Event field meanings and optionality survive generation."""
    from typing import ClassVar

    import pydantic

    from agent6.sessions import journal

    class ExampleEvent(journal.Event):
        """An example event."""

        TYPE: ClassVar[str] = "example.event"
        count: int | None = pydantic.Field(default=None, description="An optional count.")

    monkeypatch.setattr(journal, "EVENT_TYPES", {"example.event": ExampleEvent})
    page = _generator().build_markdown()
    assert "### example.event\n\n" in page
    assert "An optional count." in page
    assert "optional" in page


def test_dictionary_keys_are_read_from_code(tmp_path: pathlib.Path) -> None:
    """A new serialized key changes the reference without a generator field list."""
    generator = _generator()
    source = tmp_path / "producer.py"
    source.write_text(
        "def payload():\n"
        '    """A payload.\n\n'
        "    Attributes:\n"
        "        added: A newly serialized value.\n"
        '    """\n'
        '    return {"added": 1}\n',
        encoding="utf-8",
    )
    fields = generator.dictionary_fields(source, "payload")
    assert [(f.name, f.meaning) for f in fields] == [("added", "A newly serialized value.")]


def test_local_nested_payload_excludes_unreturned_dicts(tmp_path: pathlib.Path) -> None:
    """Only serialized dictionaries contribute keys, including locally built nested rows."""
    generator = _generator()
    source = tmp_path / "producer.py"
    source.write_text(
        "def payload():\n"
        '    """A payload.\n\n'
        "    Attributes:\n"
        "        rows: The rows.\n"
        "        rows[].added: The added value.\n"
        '    """\n'
        '    internal = {"private": 1}\n'
        '    rows = [{"added": item} for item in internal]\n'
        '    return {"rows": rows}\n',
        encoding="utf-8",
    )
    fields = generator.dictionary_fields(source, "payload")
    assert [f.name for f in fields] == ["rows", "rows[].added"]


def test_session_fields_follow_serialization() -> None:
    """The snapshot reference excludes popped state and includes the serialized additions."""
    from agent6.viewmodel import state_types, state_view

    generator = _generator()
    producers = generator.SESSION[:1] + generator.SESSION[2:]
    fields = generator.composed_fields(state_types.SessionState, producers)
    names = {f.name for f in fields}
    actual = state_view.session_state_as_dict(state_types.SessionState())
    assert actual.keys() <= names
    assert "task_label" not in names
    assert {"as_of.turn", "as_of.sha", "placement.line", "budget.usd_text"} <= names


def test_ps_fields_document_separate_status_and_reason() -> None:
    """The ps reference exposes status and reason as separate string fields."""
    generator = _generator()
    row_type = next(cls for cls in generator._models() if cls.__module__ == "agent6.ui.cli.ps_cmd")
    fields = {field.name: field.kind for field in generator.model_fields(row_type)}
    assert fields["directory"] == "str \\| None"
    assert fields["status"] == fields["reason"] == "str"


def test_model_fields_follow_the_serialized_model() -> None:
    """A model description and an added field come from the model rather than a docs registry."""
    import pydantic

    class Payload(pydantic.BaseModel):
        added: int = pydantic.Field(description="An added serialized value.")

    fields = _generator().model_fields(Payload)
    assert [(f.name, f.meaning) for f in fields] == [("added", "An added serialized value.")]


def test_missing_meaning_refuses_generation(tmp_path: pathlib.Path) -> None:
    """A new field cannot silently acquire an invented description."""
    generator = _generator()
    source = tmp_path / "producer.py"
    source.write_text('def payload():\n    return {"undocumented": 1}\n', encoding="utf-8")
    with pytest.raises(generator.FormatError, match="undocumented"):
        generator.dictionary_fields(source, "payload")


def test_public_row_types_are_not_json_placeholders() -> None:
    """Listing and status fields publish their declared types."""
    generator = _generator()
    for module, symbol in (
        ("viewmodel.listing", "summary_row"),
        ("ui.cli.sessions_show", "_cmd_status"),
    ):
        fields = generator.dictionary_fields(generator.module_path(module), symbol)
        assert fields
        assert all(f.kind != "JSON" for f in fields)
