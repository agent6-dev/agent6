# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A completer offers exactly what its argument accepts.

Offering less is a lie by omission: the operator tabs, sees no plan or ask, and
concludes the verb does not take one -- when it does. Offering more is worse,
since the suggestion is refused on Enter.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent6.paths import state_dir
from agent6.sessions.layout import bucket_dir
from agent6.ui.cli import completers

_TINY_MACHINE = """
machine = "tiny"
version = 1
initial = "route"

[budget]
max_transitions = 10

[vars.code]
n = { type = "int", default = 0 }

[states.route]
kind = "branch"
when = [
  { if = "n == 0", goto = "done" },
  { else = true, goto = "done" },
]

[states.done]
kind = "terminal"
status = "ok"
reason = "routed"
"""


def _seed(tmp_path: Path) -> None:
    state = state_dir(tmp_path)
    for bucket, mode, sid in (
        ("runs", "run", "runny-one-AAAAAA"),
        ("plans", "plan", "planny-two-BBBBB"),
        ("asks", "ask", "asky-three-CCCCC"),
        ("machines", "machine", "drafty-four-DDDD"),
    ):
        session = bucket_dir(state, bucket) / sid
        session.mkdir(parents=True)
        (session / "logs.jsonl").write_text(
            json.dumps({"type": "session.start", "mode": mode}) + "\n", encoding="utf-8"
        )
    (state / "machines" / "live-machine").mkdir(parents=True)


def test_every_session_id_is_offered_where_any_is_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`sessions show|diff|transcript|...` resolve across every bucket."""
    monkeypatch.chdir(tmp_path)
    _seed(tmp_path)
    offered = set(completers._complete_session_ids(""))  # pyright: ignore[reportPrivateUsage]
    assert offered == {
        "runny-one-AAAAAA",
        "planny-two-BBBBB",
        "asky-three-CCCCC",
        "drafty-four-DDDD",
    }


def test_resume_offers_only_what_it_can_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A machine draft is a session, but `resume` refuses it -- so suggesting it
    would be a suggestion the operator cannot act on."""
    monkeypatch.chdir(tmp_path)
    _seed(tmp_path)
    offered = set(completers._complete_resumable_ids(""))  # pyright: ignore[reportPrivateUsage]
    assert offered == {"runny-one-AAAAAA", "planny-two-BBBBB", "asky-three-CCCCC"}


def test_attach_offers_every_session_and_every_machine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _seed(tmp_path)
    offered = set(completers._complete_watch_targets(""))  # pyright: ignore[reportPrivateUsage]
    assert "live-machine" in offered
    assert {"runny-one-AAAAAA", "planny-two-BBBBB", "asky-three-CCCCC"} <= offered


def test_enum_value_completion_is_derived_from_the_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hand-kept enum table drifts: leaves whose type is a Literal got
    nothing on TAB because nobody added them. The choices come from the schema
    the config view already reads, so a new enum leaf completes for free."""
    import argparse

    from agent6.config import Config
    from agent6.config.layer import load_effective
    from agent6.viewmodel.config_view import build_config_view

    monkeypatch.chdir(tmp_path)
    schema_enums = {
        s.key
        for s in build_config_view(load_effective(tmp_path)).settings
        if s.choices and s.py_type == "choice"
    }
    assert len(schema_enums) > 10, "expected many enum leaves in the schema"
    for key in sorted(schema_enums):
        offered = completers._complete_config_values(  # pyright: ignore[reportPrivateUsage]
            "", argparse.Namespace(key=key)
        )
        assert offered, f"{key} offers no values on TAB"

    # A bool is as closed a set as any enum, and `config set` takes exactly
    # `true` or `false` there: the 17 bool leaves completed to nothing while
    # every enum completed, and `True` and `yes` are both refused.
    bools = {
        s.key for s in build_config_view(load_effective(tmp_path)).settings if s.py_type == "bool"
    }
    assert len(bools) > 10, "expected many bool leaves in the schema"
    for key in sorted(bools):
        offered = completers._complete_config_values(  # pyright: ignore[reportPrivateUsage]
            "", argparse.Namespace(key=key)
        )
        assert set(offered) == {"true", "false"}, f"{key} offers {offered}"

    # sandbox.isolation keeps its deliberate omission: TAB must not put
    # "disable the sandbox" one keystroke away.
    iso = completers._complete_config_values(  # pyright: ignore[reportPrivateUsage]
        "", argparse.Namespace(key="sandbox.isolation")
    )
    assert "none" not in iso and {"auto", "strict", "hardened"} <= set(iso)
    assert Config()  # the schema loaded


def test_live_only_verbs_offer_only_live_sessions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """steer, answer, exec, forward and sessions stop refuse a finished run,
    so offering every session offered four suggestions that fail on Enter."""
    import argparse
    import os

    from agent6.sessions.ipc import write_worker_pid
    from agent6.ui.cli.parser import build_parser

    monkeypatch.chdir(tmp_path)
    _seed(tmp_path)
    live = bucket_dir(state_dir(tmp_path), "runs") / "runny-one-AAAAAA"
    write_worker_pid(live, os.getpid())
    offered = completers._complete_live_session_ids("")  # pyright: ignore[reportPrivateUsage]
    assert offered == ["runny-one-AAAAAA"]

    parser = build_parser()
    subs = next(
        a
        for a in parser._actions  # pyright: ignore[reportPrivateUsage]
        if isinstance(a, argparse._SubParsersAction)  # pyright: ignore[reportPrivateUsage]
    )
    for verb in ("steer", "answer", "forward"):
        target = next(
            a
            for a in subs.choices[verb]._actions  # pyright: ignore[reportPrivateUsage]
            if a.dest == "target"
        )
        completer = getattr(target, "completer", None)
        assert completer is completers._complete_live_session_ids, verb  # pyright: ignore[reportPrivateUsage]


def test_live_only_verbs_do_not_offer_a_finished_run_in_its_teardown_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The verbs gate on `session_is_live` (the affordance question); the
    completer gated on `worker_is_alive`, so a run that had ended while its
    worker pid was still up was offered and then refused on Enter."""
    import json
    import os

    from agent6.sessions.ipc import write_worker_pid

    monkeypatch.chdir(tmp_path)
    ended = bucket_dir(state_dir(tmp_path), "runs") / "ended-one-EEEEEE"
    ended.mkdir(parents=True)
    (ended / "logs.jsonl").write_text(
        json.dumps({"type": "session.start", "mode": "run"})
        + "\n"
        + json.dumps({"type": "session.end", "all_passed": True, "reason": "finish_session"})
        + "\n",
        encoding="utf-8",
    )
    write_worker_pid(ended, os.getpid())
    assert completers._complete_live_session_ids("") == []  # pyright: ignore[reportPrivateUsage]


def test_config_list_edit_completion_offers_only_list_leaves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`config add/remove` reject scalar leaves, so their shared key completer
    must be narrowed to the list fields those verbs edit."""
    import argparse

    monkeypatch.chdir(tmp_path)
    for verb in ("add", "remove"):
        offered = completers._complete_config_keys(  # pyright: ignore[reportPrivateUsage]
            prefix="sandbox.",
            parsed_args=argparse.Namespace(config_command=verb, config=None, machine_file=None),
            action=object(),
        )
        assert "sandbox.extra_read_paths" in offered
        assert "sandbox.network" not in offered


def test_config_list_edit_value_completion_omits_scalar_choices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A scalar key typed by hand on `config add/remove` must not get enum
    suggestions that those list-only verbs reject."""
    import argparse

    monkeypatch.chdir(tmp_path)
    offered = completers._complete_config_values(  # pyright: ignore[reportPrivateUsage]
        prefix="",
        parsed_args=argparse.Namespace(
            config_command="add", key="sandbox.network", config=None, machine_file=None
        ),
        action=object(),
    )

    assert offered == []


def test_config_show_completion_offers_accepted_section_prefixes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`config show KEY...` accepts a whole section, so TAB must not force an
    operator typing `sand` past the valid `sandbox` candidate to `sandbox.`."""
    import argparse

    monkeypatch.chdir(tmp_path)
    offered = completers._complete_config_keys(  # pyright: ignore[reportPrivateUsage]
        prefix="sand",
        parsed_args=argparse.Namespace(config_command="show", config=None),
        action=object(),
        settable=False,
        sections=True,
    )

    assert "sandbox" in offered


def test_machine_overlay_key_completion_omits_operator_only_leaves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`config set --machine-file` refuses sandbox leaves, so its key
    completer must not offer them as writable machine-overlay inputs; `config
    get --machine-file` reads them, so its completer keeps every key."""
    import argparse

    monkeypatch.chdir(tmp_path)
    machine = tmp_path / "m.asm.toml"
    offered = completers._complete_config_keys(  # pyright: ignore[reportPrivateUsage]
        prefix="sandbox.",
        parsed_args=argparse.Namespace(config=None, machine_file=machine, config_command="set"),
        action=object(),
    )
    assert offered == []

    readable = completers._complete_config_keys(  # pyright: ignore[reportPrivateUsage]
        prefix="sandbox.",
        parsed_args=argparse.Namespace(config=None, machine_file=machine, config_command="get"),
        action=object(),
        settable=False,
    )
    assert "sandbox.network" in readable


def test_machine_overlay_value_completion_omits_operator_only_leaves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A manually typed protected machine-overlay key must not get a suggested
    value that the write command will reject on Enter."""
    import argparse

    monkeypatch.chdir(tmp_path)
    offered = completers._complete_config_values(  # pyright: ignore[reportPrivateUsage]
        prefix="",
        parsed_args=argparse.Namespace(
            key="sandbox.network", config=None, machine_file=tmp_path / "m.asm.toml"
        ),
        action=object(),
    )

    assert offered == []


def test_config_key_completion_reads_user_presets_from_the_typed_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A preset declared only by `--config FILE` is a writable key namespace
    for that invocation and must be completed from the same layer stack."""
    import argparse

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.chdir(tmp_path)
    explicit = tmp_path / "explicit.toml"
    explicit.write_text('[presets.team.review]\ntrigger = "before_finish"\n', encoding="utf-8")

    offered = completers._complete_config_keys(  # pyright: ignore[reportPrivateUsage]
        prefix="presets.",
        parsed_args=argparse.Namespace(config=explicit),
        action=object(),
    )

    assert "presets.team.review.trigger" in offered


def test_config_value_completion_under_a_preset_uses_the_leafs_choices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A generated `presets.<name>.<leaf>` key accepts the same closed values
    as that schema leaf, rather than losing completion at the preset prefix."""
    import argparse

    monkeypatch.chdir(tmp_path)
    offered = completers._complete_config_values(  # pyright: ignore[reportPrivateUsage]
        prefix="s",
        parsed_args=argparse.Namespace(key="presets.team.sandbox.network", config=None),
        action=object(),
    )

    assert offered == ["session"]


def test_mcp_remove_offers_only_servers_in_the_selected_layer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`mcp remove` edits one layer, so its completion must not offer a server
    that the effective config inherits only from the other layer."""
    import argparse

    from agent6.paths import global_config_path, repo_config_path

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.chdir(tmp_path)
    global_path = global_config_path()
    global_path.parent.mkdir(parents=True)
    global_path.write_text('[mcp.servers.global_only]\ncommand = ["true"]\n', encoding="utf-8")
    repo_path = repo_config_path(tmp_path)
    repo_path.parent.mkdir(parents=True)
    repo_path.write_text('[mcp.servers.repo_only]\ncommand = ["true"]\n', encoding="utf-8")

    global_names = completers._complete_mcp_servers(  # pyright: ignore[reportPrivateUsage]
        prefix="", parsed_args=argparse.Namespace(to_repo=False, config=None), action=object()
    )
    repo_names = completers._complete_mcp_servers(  # pyright: ignore[reportPrivateUsage]
        prefix="", parsed_args=argparse.Namespace(to_repo=True, config=None), action=object()
    )

    assert global_names == ["global_only"]
    assert repo_names == ["repo_only"]


def test_model_provider_completion_reads_the_typed_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`_cmd_model` threads `--config FILE` into every provider lookup, so a
    provider only that file declares is settable; completion read the default
    global and repo config instead and offered nothing."""
    import argparse

    from agent6.ui.cli.model import _connected_providers  # pyright: ignore[reportPrivateUsage]

    monkeypatch.chdir(tmp_path)
    custom = tmp_path / "custom.toml"
    custom.write_text(
        '[providers.myprovider]\napi_format = "openai"\nbase_url = "https://example.invalid/v1"\n',
        encoding="utf-8",
    )
    assert "myprovider" in _connected_providers(custom)
    offered = completers._complete_model_provider(  # pyright: ignore[reportPrivateUsage]
        "my", argparse.Namespace(role="worker", config=custom)
    )
    assert offered == ["myprovider"]


def test_model_completion_reads_the_typed_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`agent6 --config F model worker <provider> <TAB>` offers F's catalog;
    the completer dropped `parsed_args.config` and answered for the default
    layers, a config the command was not going to run under."""
    import argparse

    from agent6.ui.cli import model as model_mod

    seen: list[Path | None] = []

    def _catalog(config_path: Path | None, provider: str) -> list[str]:
        seen.append(config_path)
        return ["from-typed-config"] if config_path is not None else ["from-default-config"]

    monkeypatch.setattr(model_mod, "_models_for", _catalog)
    custom = tmp_path / "custom.toml"
    custom.write_text('[models.worker]\nprovider = "anthropic"\n', encoding="utf-8")
    offered = completers._complete_models(  # pyright: ignore[reportPrivateUsage]
        "from-", parsed_args=argparse.Namespace(provider="anthropic", config=custom)
    )
    assert offered == ["from-typed-config"], seen


def test_forward_offers_the_newest_sessions_ports_in_its_first_slot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bare number means that port on the newest session, so the second
    optional positional's completer must offer those ports in the first slot."""
    import argparse

    monkeypatch.chdir(tmp_path)
    session = bucket_dir(state_dir(tmp_path), "runs") / "runny-one-AAAAAA"
    session.mkdir(parents=True)
    (session / "logs.jsonl").write_text("{}\n", encoding="utf-8")

    def _ports(_path: Path) -> list[int]:
        return [8000, 9000]

    monkeypatch.setattr("agent6.sessions.ipc.listening_ports", _ports)

    offered = completers._complete_session_ports(  # pyright: ignore[reportPrivateUsage]
        prefix="8",
        parsed_args=argparse.Namespace(target=""),
        action=object(),
    )

    assert offered == ["8000"]


def test_state_restricted_machine_verbs_offer_only_machines_they_accept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`machine poke` takes only an open wait and `machine stop` takes only a
    running instance, so every suggestion works on Enter. `status` and `replay`
    still take any instance."""
    import argparse
    from collections.abc import Callable
    from typing import cast

    from agent6.machine.journal import MachineEnd, MachineJournal, PendingWait
    from agent6.sessions.layout import machines_root
    from agent6.ui.cli.parser import build_parser

    monkeypatch.chdir(tmp_path)
    _seed(tmp_path)
    waiting = machines_root(state_dir(tmp_path)) / "live-machine"
    (waiting / "machine.asm.toml").write_text(_TINY_MACHINE, encoding="utf-8")
    MachineJournal(waiting).begin(machine="tiny", version=1)
    MachineJournal(waiting).write_pending_wait(PendingWait(state="route", wake_epoch=None))
    ended = machines_root(state_dir(tmp_path)) / "ended-machine-FFFFF"
    ended.mkdir(parents=True)
    journal = MachineJournal(ended)
    journal.begin(machine="demo", version=1)
    journal.append(
        MachineEnd(
            ts="2026-01-01T00:00:00Z",
            status="ok",
            reason="finish_machine",
            state="done",
            transitions=1,
        )
    )

    def offered(verb: str) -> list[str]:
        parser = build_parser()
        subs = next(
            a
            for a in parser._actions  # pyright: ignore[reportPrivateUsage]
            if isinstance(a, argparse._SubParsersAction)  # pyright: ignore[reportPrivateUsage]
        )
        machine_subs = next(
            a
            for a in subs.choices["machine"]._actions  # pyright: ignore[reportPrivateUsage]
            if isinstance(a, argparse._SubParsersAction)  # pyright: ignore[reportPrivateUsage]
        )
        action = next(
            a
            for a in machine_subs.choices[verb]._actions  # pyright: ignore[reportPrivateUsage]
            if a.dest == "machine_id"
        )
        completer = cast(
            Callable[[str], list[str]],
            action.completer,  # pyright: ignore[reportAttributeAccessIssue]
        )
        return sorted(completer(""))

    assert offered("status") == ["ended-machine-FFFFF", "live-machine"]
    assert offered("replay") == ["ended-machine-FFFFF", "live-machine"]
    assert offered("poke") == ["live-machine"]
    assert offered("stop") == []


def test_machine_files_complete_relative_to_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--machine-file` takes a path as typed, so a file under cwd is offered
    relative to it: an absolute suggestion never matches the relative prefix
    the operator is typing, and TAB offered nothing."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "demo.asm.toml").write_text("", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "inner.asm.toml").write_text("", encoding="utf-8")
    instance = state_dir(tmp_path) / "machines" / "demo-ok"
    instance.mkdir(parents=True)
    (instance / "machine.asm.toml").write_text("", encoding="utf-8")

    assert completers._complete_machine_files("de") == ["demo.asm.toml"]  # pyright: ignore[reportPrivateUsage]
    assert completers._complete_machine_files("sub/") == ["sub/inner.asm.toml"]  # pyright: ignore[reportPrivateUsage]
    assert completers._complete_machine_files("./de") == ["./demo.asm.toml"]  # pyright: ignore[reportPrivateUsage]
    assert completers._complete_machine_files(str(tmp_path / "de")) == [  # pyright: ignore[reportPrivateUsage]
        str(tmp_path / "demo.asm.toml")
    ]
    assert completers._complete_machine_files(str(instance)) == [  # pyright: ignore[reportPrivateUsage]
        str(instance / "machine.asm.toml")
    ]
