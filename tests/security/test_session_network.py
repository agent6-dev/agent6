# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The run's session network: what it must let through, and what it must not.

A jailed child joins the machine's network (`host`), the run's own (`private`)
or one of its own (`none`). The point of `private` is that a dev server one
tool starts answers the next tool AND the MCP server driving it, with no route
off the box. Every test here runs real children and prints what they managed,
so a pass cannot be vacuous.
"""

from __future__ import annotations

import subprocess
import tempfile
import time
from pathlib import Path

import pytest

from agent6.config import Config
from agent6.sandbox.jail import SessionNetwork, spawn_in_jail
from agent6.tools.dispatch import ToolDispatcher
from agent6.tools.policy import jail_policy
from agent6.tools.results import ExecResult
from agent6.types import NetworkMode

pytestmark = pytest.mark.needs_namespaces

_PORT = 47901


def _net_of(cwd: Path, network: NetworkMode, session_net: SessionNetwork | None) -> str:
    """The network namespace a child with this policy lands in."""
    argv = ("/usr/bin/python3", "-c", "import os;print(os.readlink('/proc/self/ns/net'))")
    policy = jail_policy(cwd, Config(), "strict", argv, network=network)
    proc = spawn_in_jail(
        policy,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        session_net=session_net,
    )
    out, _ = proc.communicate(timeout=30)
    return out.decode().strip()


def test_private_children_share_one_network_and_none_children_do_not(tmp_path: Path) -> None:
    """The whole mechanism in one assertion: `private` means the SAME namespace
    for every child that asks, and `none` means a fresh one each time."""
    net = SessionNetwork.open()
    try:
        first = _net_of(tmp_path, "session", net)
        second = _net_of(tmp_path, "session", net)
        alone_a = _net_of(tmp_path, "none", net)
        alone_b = _net_of(tmp_path, "none", net)
        host = Path("/proc/self/ns/net").readlink().name
        assert first and first == second, f"private children landed apart: {first} {second}"
        assert host not in first, "the session network IS the host network"
        assert first not in (alone_a, alone_b), "a `none` child joined the shared network"
        # Sequential `none` children can recycle an inode, so this is only
        # meaningful as "not the shared one" -- which is what matters.
    finally:
        net.close()


def test_a_private_child_reaches_a_sibling_and_never_the_internet(tmp_path: Path) -> None:
    """The dev-server case, at the jail level: one child listens, another
    connects, and neither can leave the box."""
    net = SessionNetwork.open()
    listener = None
    try:
        script = (
            "import socket,time;s=socket.socket();"
            "s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1);"
            f"s.bind(('127.0.0.1',{_PORT}));s.listen(1);print('UP',flush=True);time.sleep(60)"
        )
        argv = ("/usr/bin/python3", "-u", "-c", script)
        listener = spawn_in_jail(
            jail_policy(tmp_path, Config(), "strict", argv, network="session"),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            session_net=net,
        )
        assert listener.stdout is not None
        assert b"UP" in listener.stdout.readline()

        probe = (
            "import socket\n"
            "try:\n"
            f"    socket.create_connection(('127.0.0.1',{_PORT}),timeout=5);print('SIBLING OK')\n"
            "except OSError as e:\n"
            "    print('SIBLING FAIL',type(e).__name__)\n"
            "try:\n"
            "    socket.create_connection(('1.1.1.1',53),timeout=3);print('EGRESS OK')\n"
            "except OSError as e:\n"
            "    print('NO EGRESS',type(e).__name__)\n"
        )
        argv = ("/usr/bin/python3", "-c", probe)
        client = spawn_in_jail(
            jail_policy(tmp_path, Config(), "strict", argv, network="session"),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            session_net=net,
        )
        out, err = client.communicate(timeout=30)
        text = out.decode() + err.decode()
        assert "SIBLING OK" in text, f"a private child could not reach its sibling: {text}"
        assert "NO EGRESS" in text, f"the session network reached the internet: {text}"
    finally:
        if listener is not None:
            listener.kill()
            listener.wait(timeout=10)
        net.close()


def test_an_isolated_child_cannot_reach_the_private_network(tmp_path: Path) -> None:
    """`none` is not a weaker `private`: a server left on the default must not
    see the dev server the tools are sharing."""
    net = SessionNetwork.open()
    listener = None
    try:
        script = (
            "import socket,time;s=socket.socket();"
            "s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1);"
            f"s.bind(('127.0.0.1',{_PORT + 1}));s.listen(1);print('UP',flush=True);time.sleep(60)"
        )
        argv = ("/usr/bin/python3", "-u", "-c", script)
        listener = spawn_in_jail(
            jail_policy(tmp_path, Config(), "strict", argv, network="session"),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            session_net=net,
        )
        assert listener.stdout is not None
        assert b"UP" in listener.stdout.readline()

        probe = (
            "import socket\n"
            "try:\n"
            f"    socket.create_connection(('127.0.0.1',{_PORT + 1}),timeout=4);print('REACHED')\n"
            "except OSError as e:\n"
            "    print('REFUSED',type(e).__name__)\n"
        )
        argv = ("/usr/bin/python3", "-c", probe)
        outsider = spawn_in_jail(
            jail_policy(tmp_path, Config(), "strict", argv, network="none"),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            session_net=net,
        )
        out, _ = outsider.communicate(timeout=30)
        assert b"REACHED" not in out, f"an isolated child reached the session network: {out!r}"
        assert b"REFUSED" in out, out
    finally:
        if listener is not None:
            listener.kill()
            listener.wait(timeout=10)
        net.close()


def test_a_private_child_cannot_re_enter_a_network_after_the_run_drops_it(tmp_path: Path) -> None:
    """The descriptors are the run's, not the child's: nothing is inherited, and
    seccomp blocks setns anyway, so a child cannot rejoin or reach sideways."""
    net = SessionNetwork.open()
    try:
        probe = (
            "import os\n"
            "fds=[fd for fd in range(3,64) if os.path.exists(f'/proc/self/fd/{fd}')]\n"
            "print('FDS',[os.readlink(f'/proc/self/fd/{fd}') for fd in fds])\n"
            "import ctypes\n"
            "libc=ctypes.CDLL('libc.so.6',use_errno=True)\n"
            "fd=os.open('/proc/self/ns/net',os.O_RDONLY)\n"
            "print('SETNS',libc.setns(fd,0),ctypes.get_errno())\n"
        )
        argv = ("/usr/bin/python3", "-c", probe)
        proc = spawn_in_jail(
            jail_policy(tmp_path, Config(), "strict", argv, network="session"),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            session_net=net,
        )
        out, err = proc.communicate(timeout=30)
        text = out.decode() + err.decode()
        assert "FDS []" in text, f"the child inherited a descriptor: {text}"
        # setns is on the seccomp deny list, so the call fails rather than
        # returning 0. Measured: -1 with EPERM.
        assert "SETNS -1" in text, f"a jailed child re-entered a namespace: {text}"
    finally:
        net.close()


def test_the_network_is_the_runs_and_dies_with_it() -> None:
    """Closing the run's descriptors is what releases the namespace; nothing
    outlives the run holding it open."""
    net = SessionNetwork.open()
    userns, netns = net.fds()
    assert Path(f"/proc/self/fd/{userns}").exists()
    net.close()
    assert not Path(f"/proc/self/fd/{netns}").exists(), "the run still holds its network"
    net.close()  # idempotent: teardown runs on every exit path


def test_a_private_policy_without_a_network_refuses_rather_than_running_alone(
    tmp_path: Path,
) -> None:
    """The failure that would be invisible: a child that asked for the shared
    network and silently got its own would look confined and be isolated."""
    from agent6.sandbox.jail import JailUnavailableError, run_in_jail

    policy = jail_policy(tmp_path, Config(), "strict", ("/usr/bin/true",), network="session")
    with pytest.raises(JailUnavailableError, match="session"):
        run_in_jail(policy)


def test_a_run_only_builds_a_network_when_something_would_join_it(tmp_path: Path) -> None:
    """No speculative holder: a run whose commands and servers all take the
    host network never creates one."""
    from agent6.app._setup import wants_session_network

    host_only = Config.model_validate({"sandbox": {"network": "host"}})
    assert not wants_session_network(host_only, "strict")
    assert not wants_session_network(Config(), "hardened"), "hardened has none to give"
    assert wants_session_network(Config(), "strict"), "the default puts commands on one"

    with_server = Config.model_validate(
        {
            "sandbox": {"network": "host"},
            "mcp": {
                "enabled": True,
                "servers": {"b": {"command": ["true"], "sandbox": {"network": "session"}}},
            },
        }
    )
    assert wants_session_network(with_server, "strict"), "a private server needs one too"


def test_the_dev_server_case_end_to_end(tmp_path: Path) -> None:
    """What the feature is for, through the dispatcher a run uses: a background
    dev server answers the next command, and the run still has no egress."""
    cfg = Config.model_validate({"sandbox": {"run_commands": "yes"}})
    sess = Path(tempfile.mkdtemp(prefix="privnet-", dir=tmp_path))
    d = ToolDispatcher(
        root=tmp_path, config=cfg, isolation="strict", session_dir=sess, use_jail_session=True
    )
    try:
        serve = (
            "import http.server,socketserver;"
            f"socketserver.TCPServer(('127.0.0.1',{_PORT + 2}),"
            "http.server.SimpleHTTPRequestHandler).serve_forever()"
        )
        d.dispatch("run_background", {"argv": ["/usr/bin/python3", "-c", serve]})
        time.sleep(2.0)
        got = d.dispatch(
            "run_command",
            {
                "argv": [
                    "/usr/bin/python3",
                    "-c",
                    "import urllib.request;print(urllib.request.urlopen("
                    f"'http://127.0.0.1:{_PORT + 2}/', timeout=5).status)",
                ]
            },
        )
        assert isinstance(got, ExecResult)
        assert got.returncode == 0 and "200" in got.stdout, (got.returncode, got.stderr[-300:])
        out = d.dispatch(
            "run_command",
            {
                "argv": [
                    "/usr/bin/python3",
                    "-c",
                    "import socket;socket.create_connection(('1.1.1.1',53),timeout=3)",
                ]
            },
        )
        assert isinstance(out, ExecResult)
        assert out.returncode != 0, "the run's session network reached the internet"
    finally:
        d.close()


def test_a_launcher_that_never_reports_ready_is_refused_not_waited_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A launcher too old to know `--hold-netns` reads a policy from stdin
    instead, so the handshake would block the run at startup with nothing to
    explain it. Bounded, and the refusal names the likely cause. (Reading its
    stderr to EOF hangs the same way when it left a child on the pipe, so that
    read is bounded too -- this fake keeps one alive to prove it.)"""
    from agent6.sandbox.jail import JailUnavailableError

    fake = tmp_path / "stale-jail"
    fake.write_text("#!/bin/sh\ncat > /dev/null\n", encoding="utf-8")
    fake.chmod(0o755)
    monkeypatch.setenv("AGENT6_JAIL_BIN", str(fake))
    monkeypatch.setattr("agent6.sandbox.jail._HOLDER_READY_TIMEOUT_S", 1.0)

    started = time.monotonic()
    with pytest.raises(JailUnavailableError, match="stale AGENT6_JAIL_BIN"):
        SessionNetwork.open()
    assert time.monotonic() - started < 20.0, "the run hung on the handshake"


def test_a_private_server_gets_one_even_when_the_commands_are_on_the_host(
    tmp_path: Path,
) -> None:
    """`private` means the same thing however many children ask for it, so
    there is no cross-key refusal to write: `sandbox.network = "host"` with one
    private server is simply a session network with one member."""
    from agent6.app._setup import mcp_server_policy, wants_session_network

    cfg = Config.model_validate(
        {
            "sandbox": {"network": "host"},
            "mcp": {
                "enabled": True,
                "servers": {"b": {"command": ["true"], "sandbox": {"network": "session"}}},
            },
        }
    )
    assert wants_session_network(cfg, "strict")
    server = mcp_server_policy(cfg, tmp_path, "strict", cfg.mcp.servers["b"])
    assert server is not None and server.network == "session"
    command = jail_policy(tmp_path, cfg, "strict", ("true",))
    assert command.network == "host", "the operator put the commands on the host network"


_BROWSER_SERVER = """
import json, sys, urllib.request
TOOLS = [{"name": "load", "description": "GET a url", "inputSchema":
          {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}}]
def reply(i, r):
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": i, "result": r}) + "\\n")
    sys.stdout.flush()
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    m = json.loads(line)
    if m.get("method") == "initialize":
        reply(m["id"], {"protocolVersion": "2024-11-05", "capabilities": {},
                        "serverInfo": {"name": "b", "version": "1"}})
    elif m.get("method") == "tools/list":
        reply(m["id"], {"tools": TOOLS})
    elif m.get("method") == "tools/call":
        url = (m.get("params") or {}).get("arguments", {}).get("url", "")
        try:
            out = f"HTTP {urllib.request.urlopen(url, timeout=4).status}"
        except Exception as exc:
            out = f"FAILED {type(exc).__name__}"
        reply(m["id"], {"content": [{"type": "text", "text": out}]})
    elif "id" in m:
        reply(m["id"], {})
"""


@pytest.mark.parametrize(
    ("server_network", "sees_dev_server", "sees_host"),
    [("session", True, False), ("auto", False, False), ("host", False, True)],
)
def test_an_mcp_server_reaches_the_dev_server_only_on_the_private_network(
    tmp_path: Path, server_network: str, sees_dev_server: bool, sees_host: bool
) -> None:
    """The case this feature exists for, and its boundaries, against two real
    listeners: one INSIDE the run's network (a `run_background` dev server) and
    one on the machine's (this test process). A server sees exactly one of
    them, and `auto` sees neither -- which is also the proof that the two
    networks are distinct in both directions, without needing the internet."""
    import http.server
    import socketserver
    import threading

    from agent6.app._setup import mcp_server_policy, wants_session_network
    from agent6.tools.mcp_client import MCPManager, MCPServerSpec

    script = tmp_path / "browser_server.py"
    script.write_text(_BROWSER_SERVER, encoding="utf-8")
    port = _PORT + 3
    cfg = Config.model_validate(
        {
            "sandbox": {"run_commands": "yes"},
            "mcp": {
                "enabled": True,
                "servers": {
                    "browser": {
                        "command": ["/usr/bin/python3", str(script)],
                        "approve": "yes",
                        "sandbox": {"network": server_network},
                    }
                },
            },
        }
    )
    # A listener on the MACHINE's network, which only a `host` server can see.
    httpd = socketserver.TCPServer(("127.0.0.1", 0), http.server.SimpleHTTPRequestHandler)
    host_port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    net = SessionNetwork.open() if wants_session_network(cfg, "strict") else None
    srv = cfg.mcp.servers["browser"]
    mgr = MCPManager.start(
        [
            MCPServerSpec(
                name="browser",
                command=srv.command,
                startup_timeout_s=15.0,
                call_timeout_s=20.0,
                policy=mcp_server_policy(cfg, tmp_path, "strict", srv),
            )
        ],
        session_net=net,
    )
    sess = Path(tempfile.mkdtemp(prefix="browser-", dir=tmp_path))
    d = ToolDispatcher(
        root=tmp_path,
        config=cfg,
        isolation="strict",
        session_dir=sess,
        use_jail_session=True,
        mcp_manager=mgr,
        session_net=net,
    )
    try:
        assert not mgr.failures, [f.error for f in mgr.failures]
        serve = (
            "import http.server,socketserver;"
            f"socketserver.TCPServer(('127.0.0.1',{port}),"
            "http.server.SimpleHTTPRequestHandler).serve_forever()"
        )
        d.dispatch("run_background", {"argv": ["/usr/bin/python3", "-c", serve]})
        time.sleep(2.0)
        dev = str(d.dispatch("mcp__browser__load", {"url": f"http://127.0.0.1:{port}/"}).to_wire())
        host = str(
            d.dispatch("mcp__browser__load", {"url": f"http://127.0.0.1:{host_port}/"}).to_wire()
        )
        assert ("HTTP 200" in dev) is sees_dev_server, f"{server_network}: dev server -> {dev}"
        assert ("HTTP 200" in host) is sees_host, f"{server_network}: host listener -> {host}"
    finally:
        d.close()
        mgr.close()
        if net is not None:
            net.close()
        httpd.shutdown()
        httpd.server_close()


def test_members_of_the_private_network_cannot_see_or_signal_each_other(tmp_path: Path) -> None:
    """Sharing a network means sharing a user namespace (entering one needs
    capabilities in its owner), so the question is what ELSE that shares. Each
    member still unshares its own PID namespace, so it cannot even name a
    sibling, let alone signal it."""
    net = SessionNetwork.open()
    victim = None
    try:
        argv = ("/usr/bin/python3", "-u", "-c", "import time;print('UP',flush=True);time.sleep(60)")
        victim = spawn_in_jail(
            jail_policy(tmp_path, Config(), "strict", argv, network="session"),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            session_net=net,
        )
        assert victim.stdout is not None
        assert b"UP" in victim.stdout.readline()

        probe = (
            "import os, glob\n"
            "print('PIDS', sorted(p.split('/')[-1] for p in glob.glob('/proc/[0-9]*')))\n"
            "hits = []\n"
            "for pid in range(2, 500):\n"
            "    try:\n"
            "        os.kill(pid, 0)\n"
            "        hits.append(pid)\n"
            "    except OSError:\n"
            "        pass\n"
            "print('SIGNALABLE', hits)\n"
        )
        argv = ("/usr/bin/python3", "-c", probe)
        attacker = spawn_in_jail(
            jail_policy(tmp_path, Config(), "strict", argv, network="session"),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            session_net=net,
        )
        out, _ = attacker.communicate(timeout=30)
        text = out.decode()
        # PID 1 is its own launcher-init, 2 is itself: nobody else exists here.
        assert "PIDS ['1', '2']" in text, f"a sibling was visible: {text}"
        assert "SIGNALABLE [2]" in text, f"a sibling was signalable: {text}"
    finally:
        if victim is not None:
            victim.kill()
            victim.wait(timeout=10)
        net.close()


def test_two_runs_can_each_hold_the_same_port(tmp_path: Path) -> None:
    """A property that falls out of per-run networks and that people will lean
    on: two runs (or two `--parallel` lanes) each start a dev server on the
    conventional port, and neither collides with the other or with the host."""
    cfg = Config.model_validate({"sandbox": {"run_commands": "yes"}})
    port = _PORT + 4
    serve = (
        "import http.server,socketserver;"
        f"socketserver.TCPServer(('127.0.0.1',{port}),"
        "http.server.SimpleHTTPRequestHandler).serve_forever()"
    )
    check = (
        "import urllib.request;print(urllib.request.urlopen("
        f"'http://127.0.0.1:{port}/', timeout=4).status)"
    )
    runs = [
        ToolDispatcher(
            root=tmp_path,
            config=cfg,
            isolation="strict",
            session_dir=Path(tempfile.mkdtemp(prefix=f"run{i}-", dir=tmp_path)),
            use_jail_session=True,
        )
        for i in (1, 2)
    ]
    try:
        for run in runs:
            run.dispatch("run_background", {"argv": ["/usr/bin/python3", "-c", serve]})
        time.sleep(2.5)
        for i, run in enumerate(runs, 1):
            got = run.dispatch("run_command", {"argv": ["/usr/bin/python3", "-c", check]})
            assert isinstance(got, ExecResult)
            assert got.returncode == 0 and "200" in got.stdout, (
                f"run {i} could not reach its own dev server: {got.stderr[-200:]}"
            )
    finally:
        for run in runs:
            run.close()


def test_a_member_cannot_retune_the_network_everyone_shares(tmp_path: Path) -> None:
    """Sharing a network namespace shares its sysctls. Tampering used to hurt
    only yourself; it would now hurt every sibling, so pin that the jail's
    read-only /proc still refuses it."""
    net = SessionNetwork.open()
    try:
        probe = (
            "import pathlib\n"
            "p = pathlib.Path('/proc/sys/net/ipv4/ip_local_port_range')\n"
            "print('READ', p.read_text().strip())\n"
            "try:\n"
            "    p.write_text('20000 20001\\n')\n"
            "    print('RETUNED')\n"
            "except OSError as exc:\n"
            "    print('REFUSED', type(exc).__name__)\n"
        )
        argv = ("/usr/bin/python3", "-c", probe)
        proc = spawn_in_jail(
            jail_policy(tmp_path, Config(), "strict", argv, network="session"),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            session_net=net,
        )
        out, _ = proc.communicate(timeout=30)
        text = out.decode()
        assert "READ" in text, f"the probe never ran: {text}"
        assert "RETUNED" not in text, f"a member retuned the shared network: {text}"
        assert "REFUSED" in text, text
    finally:
        net.close()


def test_joining_a_network_costs_no_other_layer(tmp_path: Path) -> None:
    """A joined child enters someone else's user namespace instead of making
    its own, which is the one thing that could quietly weaken the rest. It does
    not: the private dirs are still masked, the host is still read-only, and it
    is still PID 2 in a namespace of its own."""
    from agent6.paths import private_dirs

    net = SessionNetwork.open()
    try:
        probe = (
            "import os, pathlib\n"
            f"p = pathlib.Path({str(private_dirs()[0])!r})\n"
            "print('SECRETS', 'VISIBLE' if p.exists() and any(p.iterdir()) else 'MASKED')\n"
            "try:\n"
            "    pathlib.Path('/etc/agent6-escape-probe').write_text('x')\n"
            "    print('ETC WRITABLE')\n"
            "except OSError as exc:\n"
            "    print('ETC', type(exc).__name__)\n"
            "print('PID', os.getpid())\n"
        )
        argv = ("/usr/bin/python3", "-c", probe)
        proc = spawn_in_jail(
            jail_policy(tmp_path, Config(), "strict", argv, network="session"),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            session_net=net,
        )
        out, err = proc.communicate(timeout=30)
        text = out.decode() + err.decode()
        assert "SECRETS MASKED" in text, f"a joined child saw agent6's private dirs: {text}"
        assert "ETC WRITABLE" not in text, f"a joined child wrote the host: {text}"
        assert "PID 2" in text, f"a joined child kept the host PID namespace: {text}"
        assert not Path("/etc/agent6-escape-probe").exists()
    finally:
        net.close()


def test_closing_a_network_releases_every_descriptor(tmp_path: Path) -> None:
    """A run's network costs nothing once the run ends.

    The holder is a live process with pipes, and `close()` released the two
    namespace descriptors while leaving its stdout and stderr to garbage
    collection -- two per run that a long-lived web or hub process would
    accumulate. Measured against the process's own fd table, so the assertion
    is the resource, not the code path.
    """

    def open_fds() -> int:
        return len(list(Path("/proc/self/fd").iterdir()))

    baseline = open_fds()
    held = [SessionNetwork.open() for _ in range(8)]
    assert open_fds() > baseline, "the probe is not measuring anything"
    for net in held:
        net.close()
    assert open_fds() <= baseline + 1, (
        f"descriptors leaked: {baseline} before, {open_fds()} after 8 open/close"
    )


def test_one_runs_network_cannot_reach_another_runs(tmp_path: Path) -> None:
    """Runs are isolated from each other, not just from the machine.

    Two runs on one box -- two `--parallel` lanes, or two terminals -- each get
    a network of their own, so a server in one cannot reach a dev server in the
    other even though both are agent6 and both are the same user.
    """
    port = _PORT + 5
    first, second = SessionNetwork.open(), SessionNetwork.open()
    listener = None
    try:
        script = (
            "import socket,time;s=socket.socket();"
            "s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1);"
            f"s.bind(('127.0.0.1',{port}));s.listen(1);print('UP',flush=True);time.sleep(60)"
        )
        listener = spawn_in_jail(
            jail_policy(
                tmp_path,
                Config(),
                "strict",
                ("/usr/bin/python3", "-u", "-c", script),
                network="session",
            ),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            session_net=first,
        )
        assert listener.stdout is not None
        assert b"UP" in listener.stdout.readline()

        probe = (
            "import socket\n"
            "try:\n"
            f"    socket.create_connection(('127.0.0.1',{port}),timeout=4);print('REACHED')\n"
            "except OSError as exc:\n"
            "    print('REFUSED',type(exc).__name__)\n"
        )
        intruder = spawn_in_jail(
            jail_policy(
                tmp_path, Config(), "strict", ("/usr/bin/python3", "-c", probe), network="session"
            ),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            session_net=second,  # the OTHER run's network
        )
        out, _ = intruder.communicate(timeout=30)
        assert b"REACHED" not in out, f"a run reached another run's service: {out!r}"
        assert b"REFUSED" in out, out
    finally:
        if listener is not None:
            listener.kill()
            listener.wait(timeout=10)
        first.close()
        second.close()
