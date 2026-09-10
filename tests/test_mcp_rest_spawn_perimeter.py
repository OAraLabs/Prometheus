"""The MCP REST write surface spawns processes — who may, and what may be said.

THE DEFECT THESE GUARD
----------------------
``POST /api/mcp/servers`` stored whatever definition it was handed and applied
it live. Presence of a ``command`` or a ``url`` was the ONLY validation:
``command``, ``args``, ``env`` and ``cwd`` were never checked, and the chain

    POST → McpServerStore.upsert → _mcp_apply_live → McpRuntime.connect_server
         → resolve_stdio_config → stdio_client(StdioServerParameters(...))

ends in a process running as the daemon user. Any valid bearer reached it,
including an enrolled DEVICE token, and no config key could switch the surface
off. It was a strictly shorter path to code execution than the coding route,
which is gated.

WHAT THE FIX CLAIMS, AND WHAT IT DOES NOT
-----------------------------------------
It does NOT claim that defining an MCP server is now safe. Launching a stdio
MCP server *is* arbitrary code execution — ``npx -y <package>`` runs whatever
that package does, and no allowlist of command names changes that. Pretending
otherwise would be the reports-itself-working failure this repo keeps finding.

What it claims is narrower and checkable:

  * WHO — the write verbs need the GLOBAL token (``TestWhoMaySpawn``).
  * WHETHER — ``mcp.rest_management`` switches the surface off (``TestConfigGate``).
  * WHAT — a stored definition means what it says (``TestDefinitionMeansWhatItSays``).
  * VISIBLE — the launch lands in the security trail (``TestAuditTrail``).

CATEGORY
--------
Purely remote-chosen: the agent cannot reach this surface at all, so the fix
removes no agent capability. ``TestTheAgentCannotReachThisSurface`` pins that
rather than asserting it in prose — if it ever fails, the finding moved from
"remote-chosen" to "agent-chosen but remotely steerable" and the perimeter has
to be re-decided, not merely re-passed.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("mcp")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from prometheus.mcp.runtime import McpRuntime, _McpSession  # noqa: E402
from prometheus.mcp.store import McpServerStore, McpStoreError  # noqa: E402
from prometheus.tools.base import ToolRegistry  # noqa: E402
from prometheus.web.server import create_app  # noqa: E402

GLOBAL_TOKEN = "global-token-for-tests-only-not-a-real-credential"
ENV_SECRET = "mcp-env-value-that-must-never-be-logged"

#: ⚠ THE ENV NAME HERE IS LOAD-BEARING AND MUST NOT BE "PRETTIER".
#:
#: AuditLogger._redact is NAME-based: it masks a value whose key looks like a
#: credential (``*TOKEN``, ``*API_KEY``) and passes everything else through
#: verbatim. Measured, not assumed::
#:
#:     DOCS_TOKEN     leaked=False  -> {"DOCS_TOKEN=***"}
#:     DOCS_API_KEY   leaked=False  -> {"DOCS_API_KEY=***"}
#:     DOCS_ENDPOINT  leaked=True   -> {"DOCS_ENDPOINT": "<the value>"}
#:
#: So a leak test written with ``DOCS_TOKEN`` passes whether or not the route
#: sends env values — it measures the redactor, not this fix. The first draft
#: of this file did exactly that, and the mutant that put the whole env dict
#: into the audit row SURVIVED it. This name is one the redactor does NOT
#: catch, which makes the route's own projection the only thing between the
#: value and the log.
UNREDACTED_ENV_NAME = "DOCS_ENDPOINT"
UNREDACTED_ENV_VALUE = "https://internal.example.invalid/secret-path"


# --------------------------------------------------------------------------- #
# Rig
# --------------------------------------------------------------------------- #


def _offered(name: str) -> SimpleNamespace:
    return SimpleNamespace(
        name=name, description=f"{name} desc",
        inputSchema={"type": "object", "properties": {}},
        annotations=SimpleNamespace(readOnlyHint=True),
    )


@pytest.fixture()
def rig(monkeypatch):
    """App with a REAL api token and a REAL device store, transport stubbed.

    The existing suite (test_api_mcp_servers.py) builds an app with NO api
    token, where the middleware is deliberately inert. That rig cannot see an
    authorization defect at all — which is why this file builds its own.
    """
    async def _fake_connect_stdio(self, server_name, config):  # noqa: ANN001
        inner = MagicMock()
        inner.send_ping = AsyncMock()
        return _McpSession(server_name=server_name, session=inner,
                           transport_type="stdio", _exit_stack=AsyncMock())

    async def _fake_list(session):  # noqa: ANN001
        return [_offered("lookup")]

    monkeypatch.setattr(McpRuntime, "_connect_stdio", _fake_connect_stdio)
    monkeypatch.setattr("prometheus.mcp.runtime._list_all_tools", _fake_list)

    from prometheus.config.device_store import DeviceStore

    devices = DeviceStore()
    phone = devices.mint("a-phone", "ios")

    def _build(config_extra: dict | None = None) -> TestClient:
        config = {"web": {"api_token": GLOBAL_TOKEN}, "mcp_servers": {}}
        config.update(config_extra or {})
        app = create_app(config, device_store=devices)
        app.state.mcp_runtime = McpRuntime({})
        app.state.ws_bridge = SimpleNamespace(
            loop_context=SimpleNamespace(tool_registry=ToolRegistry())
        )
        return TestClient(app)

    return SimpleNamespace(build=_build, device_token=phone["token"])


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


DEFINITION = {"name": "docs", "command": "npx", "args": ["-y", "docs-mcp"],
              "env": {"DOCS_TOKEN": ENV_SECRET,
                      UNREDACTED_ENV_NAME: UNREDACTED_ENV_VALUE}}


# --------------------------------------------------------------------------- #
# WHO
# --------------------------------------------------------------------------- #


class TestWhoMaySpawn:
    """A device token is the wrong credential for starting a process."""

    def test_device_token_is_refused_and_writes_nothing(self, rig) -> None:
        client = rig.build()
        resp = client.post("/api/mcp/servers", json=DEFINITION,
                           headers=_auth(rig.device_token))

        assert resp.status_code == 401, resp.text
        assert "global token" in resp.json()["error"]
        # The REFUSAL is not the claim — the absence of the side effect is.
        # A 401 with the server stored anyway would still have spawned it.
        assert McpServerStore().load() == {}, (
            f"refused, but the store holds {McpServerStore().load()!r}")

    def test_device_token_refused_on_patch_and_delete_too(self, rig) -> None:
        client = rig.build()
        client.post("/api/mcp/servers", json=DEFINITION,
                    headers=_auth(GLOBAL_TOKEN))
        before = McpServerStore().load()
        assert before["docs"]["command"] == "npx"

        patch = client.patch("/api/mcp/servers/docs",
                             json={"command": "/bin/sh"},
                             headers=_auth(rig.device_token))
        delete = client.delete("/api/mcp/servers/docs",
                               headers=_auth(rig.device_token))

        assert (patch.status_code, delete.status_code) == (401, 401)
        # Contents, not existence: a PATCH that 401s but rewrote `command`
        # would be the whole defect wearing a 401.
        assert McpServerStore().load() == before, (
            f"expected {before!r}, got {McpServerStore().load()!r}")

    def test_global_token_still_works(self, rig) -> None:
        client = rig.build()
        resp = client.post("/api/mcp/servers", json=DEFINITION,
                           headers=_auth(GLOBAL_TOKEN))
        assert resp.status_code == 200, resp.text
        assert resp.json()["applies"] == "live"
        assert McpServerStore().load()["docs"]["command"] == "npx"

    def test_reading_is_deliberately_not_narrowed(self, rig) -> None:
        """A phone may SEE the connectors; the read exposes no credential."""
        client = rig.build()
        client.post("/api/mcp/servers", json=DEFINITION,
                    headers=_auth(GLOBAL_TOKEN))

        listing = client.get("/api/mcp/servers", headers=_auth(rig.device_token))
        assert listing.status_code == 200, listing.text
        card = {s["name"]: s for s in listing.json()["servers"]}["docs"]
        assert card["env_names"] == ["DOCS_ENDPOINT", "DOCS_TOKEN"]
        assert ENV_SECRET not in listing.text
        assert UNREDACTED_ENV_VALUE not in listing.text

    def test_no_token_at_all_is_refused(self, rig) -> None:
        client = rig.build()
        assert client.post("/api/mcp/servers", json=DEFINITION).status_code == 401
        assert McpServerStore().load() == {}


# --------------------------------------------------------------------------- #
# WHETHER
# --------------------------------------------------------------------------- #


class TestConfigGate:
    def test_rest_management_false_disables_the_write_verbs(self, rig) -> None:
        client = rig.build({"mcp": {"rest_management": False}})
        resp = client.post("/api/mcp/servers", json=DEFINITION,
                           headers=_auth(GLOBAL_TOKEN))
        assert resp.status_code == 403, resp.text
        assert "rest_management" in resp.json()["error"]
        assert McpServerStore().load() == {}

    def test_gate_off_leaves_reading_alone(self, rig) -> None:
        client = rig.build({
            "mcp": {"rest_management": False},
            "mcp_servers": {"yaml-srv": {"command": "cfg"}},
        })
        listing = client.get("/api/mcp/servers", headers=_auth(GLOBAL_TOKEN))
        assert listing.status_code == 200, listing.text
        assert [s["name"] for s in listing.json()["servers"]] == ["yaml-srv"]

    def test_default_is_on_so_an_absent_section_changes_nothing(self, rig) -> None:
        """`mcp:` absent must not be read as `rest_management: false`.

        Absent and false are different facts (the config-pin defect, #414);
        this pins that the ABSENT case still permits, so the guard above is
        testing the value rather than the section's existence.
        """
        client = rig.build()          # no `mcp:` key at all
        assert client.post("/api/mcp/servers", json=DEFINITION,
                           headers=_auth(GLOBAL_TOKEN)).status_code == 200

    def test_shipped_template_documents_the_key_and_agrees_with_the_code(self) -> None:
        """The template's value and the code's fallback must be the same fact."""
        import yaml

        template = Path(__file__).resolve().parents[1] / "config" / "prometheus.yaml.default"
        shipped = yaml.safe_load(template.read_text(encoding="utf-8"))
        assert shipped["mcp"]["rest_management"] is True


# --------------------------------------------------------------------------- #
# WHAT — the fields that were never checked
# --------------------------------------------------------------------------- #


class TestDefinitionMeansWhatItSays:
    """Each case is a shape the store accepted and stored before this change."""

    @pytest.mark.parametrize("name", [
        "LD_PRELOAD", "LD_AUDIT", "DYLD_INSERT_LIBRARIES",
        "NODE_OPTIONS", "PYTHONSTARTUP", "PYTHONPATH", "BASH_ENV",
        "PERL5OPT", "RUBYOPT", "GIT_SSH_COMMAND",
    ])
    def test_env_that_changes_which_program_runs_is_refused(self, name) -> None:
        """`command: npx` plus one of these is not `npx` any more.

        This is the check that makes the card and the process the same thing.
        It is NOT a claim that a caller who may POST cannot run code — they
        can, by naming it in `command`, and that is what an MCP server is.
        """
        with pytest.raises(McpStoreError, match="changes WHICH program runs"):
            McpServerStore.validate("ok", {
                "command": "npx", "args": ["-y", "docs-mcp"],
                "env": {name: "/tmp/evil.so"},
            })

    def test_the_hijack_check_is_case_insensitive(self) -> None:
        with pytest.raises(McpStoreError, match="changes WHICH program runs"):
            McpServerStore.validate("ok", {"command": "npx",
                                           "env": {"ld_preload": "/tmp/x.so"}})

    def test_ordinary_env_still_passes(self) -> None:
        """The denylist must not be so wide that real servers stop working."""
        out = McpServerStore.validate("ok", {
            "command": "npx", "env": {"DOCS_TOKEN": "abc", "HOME": "/home/x",
                                      "PATH": "/usr/bin", "NODE_ENV": "production"},
        })
        assert out["env"]["NODE_ENV"] == "production"

    def test_env_name_must_be_a_posix_name(self) -> None:
        with pytest.raises(McpStoreError, match="POSIX environment name"):
            McpServerStore.validate("ok", {"command": "x",
                                           "env": {"not a name": "v"}})

    def test_command_shape(self) -> None:
        with pytest.raises(McpStoreError, match="non-empty string"):
            McpServerStore.validate("ok", {"command": "   ", "url": "http://x/y"})
        with pytest.raises(McpStoreError, match="control characters"):
            McpServerStore.validate("ok", {"command": "npx\x00/bin/sh"})

    def test_args_shape(self) -> None:
        with pytest.raises(McpStoreError, match="list of strings"):
            McpServerStore.validate("ok", {"command": "npx", "args": "-y docs"})
        with pytest.raises(McpStoreError, match="list of strings"):
            McpServerStore.validate("ok", {"command": "npx", "args": ["-y", 7]})
        with pytest.raises(McpStoreError, match="control characters"):
            McpServerStore.validate("ok", {"command": "npx", "args": ["a\x00b"]})

    def test_cwd_must_be_absolute(self) -> None:
        """A relative cwd resolves against the DAEMON's cwd, not the author's."""
        with pytest.raises(McpStoreError, match="absolute path"):
            McpServerStore.validate("ok", {"command": "npx", "cwd": "../../etc"})
        with pytest.raises(McpStoreError, match="absolute path"):
            McpServerStore.validate("ok", {"command": "npx",
                                           "workingDirectory": "rel/path"})

    def test_cwd_that_is_a_file_is_refused(self, tmp_path) -> None:
        f = tmp_path / "not-a-dir"
        f.write_text("x")
        with pytest.raises(McpStoreError, match="is a file, not a directory"):
            McpServerStore.validate("ok", {"command": "npx", "cwd": str(f)})

    def test_absolute_cwd_that_does_not_exist_yet_is_allowed(self, tmp_path) -> None:
        """Deliberately not an existence check — see the store's comment."""
        later = tmp_path / "created-later"
        out = McpServerStore.validate("ok", {"command": "npx", "cwd": str(later)})
        assert out["cwd"] == str(later)

    def test_patch_cannot_smuggle_past_validation(self, rig) -> None:
        """PATCH merges then validates; the merged result is what must be legal."""
        client = rig.build()
        client.post("/api/mcp/servers", json=DEFINITION, headers=_auth(GLOBAL_TOKEN))
        resp = client.patch("/api/mcp/servers/docs",
                            json={"env": {"LD_PRELOAD": "/tmp/evil.so"}},
                            headers=_auth(GLOBAL_TOKEN))
        assert resp.status_code == 400, resp.text
        assert "LD_PRELOAD" not in json.dumps(McpServerStore().load())


# --------------------------------------------------------------------------- #
# VISIBLE
# --------------------------------------------------------------------------- #


class TestAuditTrail:
    def _rows(self) -> list[dict]:
        from prometheus.config.paths import get_data_dir

        log = get_data_dir() / "permission_audit.jsonl"
        if not log.exists():
            return []
        return [json.loads(line) for line in
                log.read_text(encoding="utf-8").splitlines() if line.strip()]

    def _db_rows(self) -> list[tuple]:
        """The SQLite half — this is the copy ``audit_query`` reads back."""
        import sqlite3

        from prometheus.config.paths import get_data_dir

        db = get_data_dir() / "audit.db"
        if not db.exists():
            return []
        with sqlite3.connect(db) as conn:
            return conn.execute(
                "SELECT tool_name, user_id, tool_input_summary FROM "
                "permission_audit WHERE tool_name = 'mcp.rest_define_server'"
            ).fetchall()

    def test_a_rest_defined_launch_reaches_the_security_trail(self, rig) -> None:
        client = rig.build()
        assert client.post("/api/mcp/servers", json=DEFINITION,
                           headers=_auth(GLOBAL_TOKEN)).status_code == 200

        rows = [r for r in self._rows()
                if r["tool_name"] == "mcp.rest_define_server"]
        assert len(rows) == 1, f"expected one row, got {self._rows()!r}"
        summary = rows[0]["tool_input_summary"]
        assert "npx" in summary and "docs-mcp" in summary
        # Both sinks, because they are read by different surfaces.
        db = self._db_rows()
        assert len(db) == 1, db
        assert db[0][1] == "global", db                # on whose authority
        assert "npx" in db[0][2], db

    def test_the_trail_records_env_names_but_never_env_values(self, rig) -> None:
        """`audit_query` feeds this log back into model context (open finding).

        So the row must carry enough to answer "what was allowed to run" and
        nothing that turns the audit trail into a credential source.
        """
        client = rig.build()
        client.post("/api/mcp/servers", json=DEFINITION, headers=_auth(GLOBAL_TOKEN))

        rows = [r for r in self._rows()
                if r["tool_name"] == "mcp.rest_define_server"]
        summary = rows[0]["tool_input_summary"]
        assert "DOCS_TOKEN" in summary          # the name is the useful half
        assert UNREDACTED_ENV_NAME in summary

        # THE ASSERTION THAT DOES THE WORK is the one on a value the redactor
        # does NOT mask — see UNREDACTED_ENV_NAME. ENV_SECRET is checked too,
        # but it proves nothing on its own: `DOCS_TOKEN=...` is masked
        # downstream whatever this route sends.
        both_sinks = json.dumps(self._rows()) + json.dumps(self._db_rows())
        assert UNREDACTED_ENV_VALUE not in summary, summary
        assert UNREDACTED_ENV_VALUE not in both_sinks, both_sinks
        assert ENV_SECRET not in both_sinks

    def test_a_refused_request_spawns_nothing_to_record(self, rig) -> None:
        client = rig.build()
        client.post("/api/mcp/servers", json=DEFINITION,
                    headers=_auth(rig.device_token))
        assert [r for r in self._rows()
                if r["tool_name"] == "mcp.rest_define_server"] == []


# --------------------------------------------------------------------------- #
# CATEGORY — the pin that says which perimeter this needs
# --------------------------------------------------------------------------- #


class TestTheAgentCannotReachThisSurface:
    def test_no_tool_constructs_the_mcp_store(self) -> None:
        """Structural. The claim "this fix removes no agent capability" is only
        true while the store is unreachable from tool code.

        Asserted over the AST of every module under ``tools/`` rather than by
        grep: an import inside a function body is exactly how such a reach
        would arrive, and it is invisible to a module-header check.

        If this fails, the finding has MOVED CATEGORY — from remote-chosen to
        agent-chosen-but-remotely-steerable — and the door built in
        web/server.py is no longer the whole perimeter. Re-decide it; do not
        widen this test.
        """
        import prometheus.tools as tools_pkg

        root = Path(tools_pkg.__file__).parent
        offenders: list[str] = []
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module and (
                    node.module.startswith("prometheus.mcp.store")
                ):
                    offenders.append(f"{path.name}:{node.lineno}")
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith("prometheus.mcp.store"):
                            offenders.append(f"{path.name}:{node.lineno}")
        assert offenders == [], (
            "a tool now reaches McpServerStore — the MCP definition surface is "
            f"agent-reachable and its perimeter must be re-decided: {offenders}")

    def test_the_store_is_constructed_only_on_boot_and_rest_paths(self) -> None:
        """The complement: name the readers, so a new one is a deliberate act."""
        import prometheus

        root = Path(prometheus.__file__).parent
        readers = set()
        for path in sorted(root.rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            if "McpServerStore" in text:
                readers.add(str(path.relative_to(root)))
        assert readers == {
            "daemon.py",                 # boot: does this daemon need a runtime
            "mcp/bootstrap.py",          # boot: build the runtime
            "mcp/store.py",              # the store itself
            "web/server.py",             # the REST surface, gated above
        }, sorted(readers)
