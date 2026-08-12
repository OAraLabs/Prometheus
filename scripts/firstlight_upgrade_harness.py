#!/usr/bin/env python3
"""FIRSTLIGHT upgrade harness — install old, live in it, upgrade in place.

Sibling of scripts/firstlight_harness.py (same contract: CLI + HTTP only,
no imports from src/prometheus, minimal non-inherited environment, loud
per-step failures). The fresh-install harness proves a stranger's first
ten minutes; THIS one proves their six-month-old install survives
`git pull`:

  U1  clone --source at the BASELINE TAG (v0.1.0, the only tag) + venv +
      pip install -e '.[full]'
  U2  POPULATE as a real user: setup --noninteractive (stub model on
      :8080 — v0.1.0 has no --probe-url, so the well-known port is the
      only probe path; the harness REFUSES to run if :8080 is taken),
      three --once turns (glob + a write_file into the workspace), daemon
      boot, REST turns on two sessions, a cron job via /api/cron
      (best-effort at the old tag), a scaffold brain vault pointed at by
      vault.root
  U3  SNAPSHOT: per-DB per-table row counts, sha256 of every non-DB
      file, the config text, the API token
  U4  UPGRADE IN PLACE: git checkout <HEAD sha> + pip install -e
      '.[full]' again. No wipe; the state dir is not touched.
  U5  ASSERT: daemon boots, doctor exits 0, one CLI + one REST turn, and
      EVERY snapshot surface survives — tables present, counts >=, rows
      still parse, user files hash-identical, cron intact, token still
      opens the API, the vault byte-identical (read-only invariant)
  U6  DOWNGRADE, reported not gated: checkout the baseline again, boot,
      doctor, one turn, per-store readability probe — the printed block
      is the source of truth for the docs' downgrade section
  U7  teardown, no residue

KNOWN-DEFECT RATCHET (same idea as FL-1's --strict-shutdown): findings on
the KNOWN list below are reported LOUDLY and tolerated by default so the
harness can gate everything else while a defect awaits its own round;
--strict-state turns them fatal. Everything NOT on the list is fatal
immediately.

  FL-3   the two-location lcm.db, REFINED BY THIS HARNESS'S FIRST RUN:
         no stranding happens on a v0.1.0 upgrade (conversation rows were
         already in <config>/data/lcm.db at the tag — the engine's move
         predates it). The real defect is DIVERGENT DEFAULTS: the
         conversation store's _default_db_path() still says
         <config>/lcm.db, and the coordinator's checkpoint store
         constructs against it — so "the shared lcm.db" is two files,
         conversations in data/, checkpoints in the root fossil, and any
         future bare LCMConversationStore() lands on the wrong one. The
         detector below stays armed for genuine stranding (pre-tag
         installs, regressions) and correctly stays silent for v0.1.0.
  FL-2u  FL-2's fix writes tools.deferred_loading at SETUP time, so an
         UPGRADED install keeps its old config and still advertises
         nothing (boot line "0 always_loaded of N").

Expected at the old tag, NOT findings: the v0.1.0 daemon hangs on
SIGTERM (FL-1, fixed at HEAD) — U2/U6 stop it with the SIGKILL
escalation and say so.

OUT OF SCOPE — documented, not silently skipped:
  * memory-tool row growth during populate (the stub scripts glob +
    write_file; memory.db is still snapshot-diffed read-side)
  * gateways, voice, GPU, Beacon, cloud providers (as round 1)
  * pip-release upgrades (this drives the git-checkout install path the
    README documents; `pip install -U` is a different lane)

SELF-TEST LEVERS: --mutate drop-table (drops one snapshotted table after
U3 -> U5 must name it) · --mutate delete-db (removes telemetry.db during
U4 -> U5 must name the file).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

BASELINE_TAG = "v0.1.0"
MARKER = "FIRSTLIGHT-COMPLETE"
NOTE_CONTENT = "FIRSTLIGHT-UPGRADE-NOTE"
SKIP_DIRS = {"logs", "cache", "__pycache__", ".git", "db-snapshots"}


class StepFailure(Exception):
    def __init__(self, why: str, log: Path | None = None) -> None:
        super().__init__(why)
        self.log = log


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def port_taken(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def http(method: str, url: str, payload: dict | None = None,
         token: str | None = None, timeout: float = 10.0) -> tuple[int, str]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(
        url, method=method, headers=headers,
        data=json.dumps(payload).encode() if payload is not None else None,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")


def db_census(root: Path) -> dict[str, dict[str, int]]:
    """{relative-db-path: {table: rowcount}} for every *.db under root."""
    census: dict[str, dict[str, int]] = {}
    for db in sorted(root.rglob("*.db")):
        if any(part in SKIP_DIRS for part in db.relative_to(root).parts):
            continue
        tables: dict[str, int] = {}
        try:
            conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            try:
                names = [r[0] for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name NOT LIKE 'sqlite_%'")]
                for t in names:
                    tables[t] = conn.execute(
                        f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
            finally:
                conn.close()
        except sqlite3.Error as exc:
            tables["<unreadable>"] = -1
            tables["<error>"] = 0
            print(f"[UPGRADE] census warning: {db}: {exc}")
        census[str(db.relative_to(root))] = tables
    return census


def file_census(root: Path) -> dict[str, str]:
    """{relative-path: sha256} for every non-DB file under root."""
    out: dict[str, str] = {}
    for f in sorted(root.rglob("*")):
        rel = f.relative_to(root)
        if not f.is_file() or f.suffix == ".db" or f.name == "daemon.lock":
            continue
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        out[str(rel)] = hashlib.sha256(f.read_bytes()).hexdigest()
    return out


def tree_hash(root: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(root.rglob("*")):
        if f.is_file():
            h.update(str(f.relative_to(root)).encode())
            h.update(f.read_bytes())
    return h.hexdigest()


class UpgradeHarness:
    def __init__(self, source: Path, keep: bool, strict_state: bool,
                 mutate: str) -> None:
        self.source = source
        self.keep = keep
        self.strict_state = strict_state
        self.mutate = mutate
        self.work = Path(tempfile.mkdtemp(prefix="flupgrade-"))
        self.clone = self.work / "clone"
        self.home = self.work / "home"
        self.logs = self.work / "logs"
        self.vault = self.work / "brain-vault"
        self.home.mkdir()
        self.logs.mkdir()
        # Container runs mount --source read-only under a different uid,
        # which trips git's dubious-ownership guard. safe.directory is
        # deliberately IGNORED from env/-c scopes; it only counts from a
        # global/system gitconfig — and since every harness subprocess runs
        # with HOME set to this isolated tree, THIS is its global config.
        (self.home / ".gitconfig").write_text(
            "[safe]\n\tdirectory = *\n", encoding="utf-8")
        self.venv = self.work / "venv"
        self.api_port = free_port()
        self.ws_port = free_port()
        self.head_sha = "?"
        self.token: str | None = None
        self.snapshot_dbs: dict[str, dict[str, int]] = {}
        self.snapshot_files: dict[str, str] = {}
        self.vault_hash = ""
        self.cron_created = False
        self.findings: list[str] = []
        self.stub: subprocess.Popen | None = None
        self.daemon: subprocess.Popen | None = None

    # -- plumbing (mirrors the fresh-install harness) -----------------
    def env(self) -> dict[str, str]:
        return {
            "HOME": str(self.home),
            "PATH": f"{self.venv}/bin:/usr/bin:/bin",
            "LANG": "C.UTF-8", "TERM": "dumb", "PYTHONUNBUFFERED": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        }

    def run(self, cmd: list[str], log_name: str, timeout: int,
            cwd: Path | None = None, expect_rc: int | None = 0,
            extra_env: dict[str, str] | None = None) -> Path:
        log = self.logs / f"{log_name}.log"
        env = self.env() | (extra_env or {})
        with log.open("a", encoding="utf-8") as fh:
            fh.write(f"$ {' '.join(cmd)}\n")
            fh.flush()
            try:
                proc = subprocess.run(cmd, cwd=cwd or self.clone, env=env,
                                      stdout=fh, stderr=subprocess.STDOUT,
                                      timeout=timeout)
            except subprocess.TimeoutExpired:
                raise StepFailure(f"`{' '.join(cmd)}` timed out ({timeout}s)",
                                  log)
        if expect_rc is not None and proc.returncode != expect_rc:
            raise StepFailure(f"`{' '.join(cmd)}` exited {proc.returncode} "
                              f"(expected {expect_rc})", log)
        return log

    def cfg_root(self) -> Path:
        return self.home / ".prometheus"

    def _read_token(self) -> str | None:
        env_file = self.home / ".config" / "prometheus" / "env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("PROMETHEUS_API_TOKEN="):
                    return line.split("=", 1)[1].strip() or None
        return None

    def _start_stub(self, port: int) -> None:
        # write_file resolves relative paths against the CLI's cwd (the
        # disposable clone) — the note must land in the STATE tree, so the
        # script carries the absolute workspace path, known from HOME.
        script = json.dumps([
            {"name": "glob", "arguments": {"pattern": "*"}},
            {"name": "write_file", "arguments": {
                "path": str(self.cfg_root() / "workspace"
                            / "firstlight-note.md"),
                "content": NOTE_CONTENT}},
        ])
        log = (self.logs / "stub.log").open("a", encoding="utf-8")
        self.stub = subprocess.Popen(
            [str(self.venv / "bin" / "python"),
             str(self.source / "scripts" / "firstlight_stub_model.py"),
             "--port", str(port), "--script", script],
            stdout=log, stderr=subprocess.STDOUT, env=self.env(),
        )
        deadline = time.time() + 10
        while time.time() < deadline:
            if port_taken(port):
                return
            time.sleep(0.2)
        raise StepFailure("stub model never came up")

    def _boot_daemon(self, log_name: str) -> Path:
        log_path = self.logs / f"{log_name}.log"
        log = log_path.open("a", encoding="utf-8")
        self.daemon = subprocess.Popen(
            [str(self.venv / "bin" / "prometheus"), "daemon"],
            cwd=self.clone, env=self.env(),
            stdout=log, stderr=subprocess.STDOUT,
        )
        deadline = time.time() + 120
        while time.time() < deadline:
            if self.daemon.poll() is not None:
                raise StepFailure(f"daemon exited rc={self.daemon.returncode} "
                                  f"before the API answered", log_path)
            try:
                code, _ = http("GET",
                               f"http://127.0.0.1:{self.api_port}/api/status",
                               timeout=2)
                if code in (200, 401):
                    self.token = self._read_token()
                    return log_path
            except Exception:
                pass
            time.sleep(1)
        raise StepFailure("/api/status never answered within 120s", log_path)

    def _stop_daemon(self, tolerate_hang: bool) -> None:
        if not self.daemon or self.daemon.poll() is not None:
            self.daemon = None
            return
        self.daemon.send_signal(signal.SIGTERM)
        try:
            self.daemon.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.daemon.kill()
            self.daemon.wait(timeout=15)
            if tolerate_hang:
                print("[UPGRADE] note: old-tag daemon needed SIGKILL "
                      "(FL-1 exists at the baseline; fixed at HEAD)")
            else:
                raise StepFailure("HEAD daemon ignored SIGTERM for 15s — "
                                  "FL-1 regression")
        self.daemon = None

    def _rest_turn(self, session: str) -> None:
        base = f"http://127.0.0.1:{self.api_port}"
        code, body = http("POST", f"{base}/api/chat/send",
                          {"session_id": session, "message": "walk the script"},
                          token=self.token)
        if code != 200:
            raise StepFailure(f"chat/send({session}) -> {code}: {body[:160]}")
        deadline = time.time() + 240
        while time.time() < deadline:
            code, body = http("GET", f"{base}/api/sessions/{session}/messages",
                              token=self.token, timeout=5)
            if code == 200 and MARKER in body:
                return
            time.sleep(2)
        raise StepFailure(f"REST turn on {session} never completed")

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def u1_install_baseline(self) -> str:
        if port_taken(8080):
            raise StepFailure(
                "port 8080 is already in use — v0.1.0's setup probes ONLY "
                "the well-known ports, so this harness needs :8080 free "
                "(run in a container; the fresh-install harness's probe-url "
                "escape does not exist at the baseline tag)")
        out = subprocess.run(["git", "-C", str(self.source), "rev-parse",
                              "HEAD"], capture_output=True, text=True,
                             env=self.env())
        if out.returncode != 0:
            raise StepFailure(f"--source is not a git repo: {out.stderr}")
        self.head_sha = out.stdout.strip()
        self.run(["git", "clone", "--quiet", str(self.source),
                  str(self.clone)], "u1-install", 300, cwd=self.work)
        self.run(["git", "-C", str(self.clone), "checkout", "--quiet",
                  BASELINE_TAG], "u1-install", 60, cwd=self.work)
        self.run([sys.executable, "-m", "venv", str(self.venv)],
                 "u1-install", 120, cwd=self.work)
        self.run([str(self.venv / "bin" / "pip"), "install", "--quiet",
                  "-e", ".[full]"], "u1-install", 1500)
        return f"baseline {BASELINE_TAG}; HEAD target {self.head_sha[:12]}"

    def u2_populate(self) -> str:
        self._start_stub(8080)
        self.run([str(self.venv / "bin" / "prometheus"), "setup",
                  "--noninteractive", "--timeout", "3"], "u2-populate", 120)
        cfg = self.cfg_root() / "prometheus.yaml"
        if not cfg.exists():
            raise StepFailure(f"baseline setup wrote no config at {cfg}",
                              self.logs / "u2-populate.log")
        text = cfg.read_text(encoding="utf-8")
        for key, port in (("api_port", self.api_port), ("ws_port", self.ws_port)):
            old = f"{key}: {8005 if key == 'api_port' else 8010}"
            if text.count(old) != 1:
                raise StepFailure(f"expected one `{old}` in the baseline "
                                  f"config, found {text.count(old)} — the "
                                  f"v0.1.0 template shape changed")
            text = text.replace(old, f"{key}: {port}")
        # A scaffold brain vault, pointed at by config — v0.1.0 ignores the
        # unknown key; HEAD's vault tools must read it and never write it.
        (self.vault / "wiki").mkdir(parents=True)
        (self.vault / "BRAIN.md").write_text("# Vault Router\n")
        (self.vault / "CLAUDE.md").write_text(
            "This vault's operating instructions are in BRAIN.md.\n")
        (self.vault / "wiki" / "index.md").write_text(
            "# Index\n\n- [Note](sources/Note.md) — upgrade-harness seed\n")
        (self.vault / "wiki" / "sources").mkdir()
        (self.vault / "wiki" / "sources" / "Note.md").write_text(
            "---\ntype: concept\n---\n\n# Note\n\nupgrade harness seed page\n")
        text += f"\nvault:\n  root: {self.vault}\n"
        cfg.write_text(text, encoding="utf-8")

        for i in range(3):
            log = self.run([str(self.venv / "bin" / "prometheus"), "--once",
                            f"populate turn {i}"], "u2-populate", 300)
        if MARKER not in log.read_text(encoding="utf-8", errors="replace"):
            raise StepFailure("baseline --once turns never completed the "
                              "script", log)
        note = self.cfg_root() / "workspace" / "firstlight-note.md"
        if not note.exists() or NOTE_CONTENT not in note.read_text():
            raise StepFailure(
                f"populate's write_file never landed at {note} — the "
                f"workspace write is the state the upgrade must preserve")

        self._boot_daemon("u2-daemon")
        for session in ("flu-alpha", "flu-beta"):
            self._rest_turn(session)
        code, body = http("POST",
                          f"http://127.0.0.1:{self.api_port}/api/cron",
                          {"name": "flu_probe", "schedule": "0 6 * * *",
                           "command": "echo firstlight"}, token=self.token)
        self.cron_created = code in (200, 201)
        if not self.cron_created:
            print(f"[UPGRADE] note: cron create unavailable at baseline "
                  f"({code}) — recorded, not fatal")
        self._stop_daemon(tolerate_hang=True)
        return ("3 CLI turns + 2 REST sessions + workspace note"
                + (" + cron job" if self.cron_created else ""))

    def u3_snapshot(self) -> str:
        self.snapshot_dbs = db_census(self.cfg_root())
        self.snapshot_files = file_census(self.cfg_root())
        self.vault_hash = tree_hash(self.vault)
        (self.work / "snapshot.json").write_text(json.dumps(
            {"dbs": self.snapshot_dbs, "files": self.snapshot_files},
            indent=1))
        n_tables = sum(len(t) for t in self.snapshot_dbs.values())
        if not self.snapshot_dbs:
            raise StepFailure("snapshot found ZERO databases — populate did "
                              "not accumulate state; the harness would be "
                              "testing nothing")
        if self.mutate == "drop-table":
            db_rel, tables = next(iter(self.snapshot_dbs.items()))
            table = next(iter(tables))
            conn = sqlite3.connect(self.cfg_root() / db_rel)
            conn.execute(f'DROP TABLE "{table}"')
            conn.commit()
            conn.close()
            print(f"[UPGRADE] MUTATION drop-table: dropped {table} "
                  f"from {db_rel}")
        return (f"{len(self.snapshot_dbs)} DBs / {n_tables} tables, "
                f"{len(self.snapshot_files)} files hashed")

    def u4_upgrade_in_place(self) -> str:
        self.run(["git", "-C", str(self.clone), "checkout", "--quiet",
                  self.head_sha], "u4-upgrade", 60, cwd=self.work)
        self.run([str(self.venv / "bin" / "pip"), "install", "--quiet",
                  "-e", ".[full]"], "u4-upgrade", 1500)
        if self.mutate == "delete-db":
            victim = self.cfg_root() / "telemetry.db"
            if victim.exists():
                victim.unlink()
                print("[UPGRADE] MUTATION delete-db: removed telemetry.db")
        return f"clone now at {self.head_sha[:12]}, state dir untouched"

    def _state_diff(self) -> list[str]:
        fatal: list[str] = []
        now = db_census(self.cfg_root())
        for db_rel, tables in self.snapshot_dbs.items():
            if db_rel not in now:
                fatal.append(f"database GONE after upgrade: {db_rel}")
                continue
            for table, count in tables.items():
                if table.startswith("<"):
                    continue
                if table not in now[db_rel]:
                    fatal.append(f"table GONE after upgrade: {db_rel}:{table}")
                elif now[db_rel][table] < count:
                    fatal.append(
                        f"ROWS LOST: {db_rel}:{table} {count} -> "
                        f"{now[db_rel][table]}")
                else:
                    conn = sqlite3.connect(
                        f"file:{self.cfg_root() / db_rel}?mode=ro", uri=True)
                    try:
                        conn.execute(f'SELECT * FROM "{table}" LIMIT 1'
                                     ).fetchone()
                    except sqlite3.Error as exc:
                        fatal.append(f"table UNREADABLE after upgrade: "
                                     f"{db_rel}:{table}: {exc}")
                    finally:
                        conn.close()
        files_now = file_census(self.cfg_root())
        for rel, digest in self.snapshot_files.items():
            if any(rel.startswith(p) for p in ("sessions/", "data/")):
                continue  # stores legitimately append; DB census covers them
            if rel.startswith("anatomy") or rel == "ANATOMY.md":
                # Machine-owned projection: AnatomyWriter regenerates it
                # from infrastructure state on boot (infra/anatomy_writer).
                # A rewrite here is the system working, not state loss.
                continue
            if rel not in files_now:
                fatal.append(f"file GONE after upgrade: {rel}")
            elif files_now[rel] != digest and rel.endswith(
                    ("prometheus.yaml", ".md", "firstlight-note.md")):
                fatal.append(f"user file REWRITTEN by upgrade: {rel}")
        if tree_hash(self.vault) != self.vault_hash:
            fatal.append("the brain vault was MODIFIED — the read-only "
                         "invariant broke")
        return fatal

    def u5_assert_upgrade(self) -> str:
        log = self._boot_daemon("u5-daemon")
        code, _ = http("GET", f"http://127.0.0.1:{self.api_port}/api/status",
                       token=self.token)
        if code != 200:
            raise StepFailure(f"authed /api/status -> {code} after upgrade "
                              f"(the pre-upgrade token must keep working)", log)
        self.run([str(self.venv / "bin" / "prometheus"), "doctor"],
                 "u5-doctor", 120)
        once = self.run([str(self.venv / "bin" / "prometheus"), "--once",
                         "post-upgrade turn"], "u5-once", 300)
        if MARKER not in once.read_text(encoding="utf-8", errors="replace"):
            raise StepFailure("post-upgrade --once turn did not complete",
                              once)
        self._rest_turn("flu-alpha")

        fatal = self._state_diff()

        # FL-3: where did HEAD write the LCM rows the turns above produced?
        old_lcm = self.cfg_root() / "lcm.db"
        new_lcm = self.cfg_root() / "data" / "lcm.db"
        old_count = sum(self.snapshot_dbs.get("lcm.db", {}).values())
        if new_lcm.exists() and old_lcm.exists():
            now_old = sum(db_census(self.cfg_root()).get("lcm.db", {}).values())
            if now_old == old_count and old_count > 0:
                self.findings.append(
                    f"FL-3 CONFIRMED: {old_count} LCM rows written at "
                    f"{BASELINE_TAG} sit stranded in <config>/lcm.db; HEAD "
                    f"reads and writes <config>/data/lcm.db and never looks "
                    f"back. An in-place upgrade silently orphans the user's "
                    f"conversation history.")
        baseline_line = ""
        for line in (self.logs / "u5-daemon.log").read_text(
                encoding="utf-8", errors="replace").splitlines():
            if "advertisement baseline" in line:
                baseline_line = line
                break
        if "0 always_loaded" in baseline_line:
            self.findings.append(
                "FL-2u CONFIRMED: the upgraded install advertises ZERO tools "
                "— FL-2's fix runs at setup time, and an upgrade keeps the "
                "old config (boot line: ..." + baseline_line[-60:] + ")")

        self._stop_daemon(tolerate_hang=False)
        if fatal:
            raise StepFailure("STATE LOST IN UPGRADE:\n  " +
                              "\n  ".join(fatal))
        for f in self.findings:
            print(f"[UPGRADE-FINDING] {f}")
        if self.findings and self.strict_state:
            raise StepFailure(f"{len(self.findings)} known-defect finding(s) "
                              f"and --strict-state is set")
        return (f"state intact ({len(self.snapshot_dbs)} DBs verified); "
                f"{len(self.findings)} known-defect finding(s) reported")

    def u6_downgrade_report(self) -> str:
        self.run(["git", "-C", str(self.clone), "checkout", "--quiet",
                  BASELINE_TAG], "u6-downgrade", 60, cwd=self.work)
        self.run([str(self.venv / "bin" / "pip"), "install", "--quiet",
                  "-e", ".[full]"], "u6-downgrade", 1500)
        report: list[str] = ["DOWNGRADE REPORT (HEAD state, baseline code):"]
        boot_ok = True
        try:
            self._boot_daemon("u6-daemon")
            report.append("  daemon: BOOTS")
            once = self.run([str(self.venv / "bin" / "prometheus"), "--once",
                             "downgrade probe"], "u6-once", 300,
                            expect_rc=None)
            text = once.read_text(encoding="utf-8", errors="replace")
            report.append("  --once turn: " +
                          ("completes" if MARKER in text else
                           "DOES NOT complete"))
        except StepFailure as exc:
            boot_ok = False
            report.append(f"  daemon: DOES NOT BOOT ({exc})")
        finally:
            self._stop_daemon(tolerate_hang=True)
        for db_rel, tables in db_census(self.cfg_root()).items():
            unreadable = [t for t, c in tables.items() if c < 0]
            report.append(f"  {db_rel}: {len(tables)} tables, " +
                          ("ALL READABLE" if not unreadable else
                           f"UNREADABLE: {unreadable}"))
        report.append(
            "  verdict: " +
            ("probably readable but lossy — new-column semantics and any "
             "post-upgrade state in new locations are invisible to the old "
             "code; not supported, and the docs must say so in those words"
             if boot_ok else
             "NOT EVEN BOOTABLE — downgrade is corrupting; docs must say "
             "so plainly"))
        block = "\n".join(report)
        print(block)
        (self.work / "downgrade-report.txt").write_text(block)
        return f"reported ({'boots' if boot_ok else 'DOES NOT BOOT'})"

    def u7_teardown(self) -> str:
        for proc in (self.daemon, self.stub):
            if proc and proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
        self.daemon = self.stub = None
        if self.keep:
            return f"--keep set, tree kept at {self.work}"
        shutil.rmtree(self.work, ignore_errors=False)
        return "temp tree removed, no residue"

    def main(self) -> int:
        steps = [
            ("U1", f"install baseline {BASELINE_TAG}", self.u1_install_baseline),
            ("U2", "populate like a user", self.u2_populate),
            ("U3", "snapshot state", self.u3_snapshot),
            ("U4", "upgrade in place to HEAD", self.u4_upgrade_in_place),
            ("U5", "assert nothing was lost", self.u5_assert_upgrade),
            ("U6", "downgrade, reported not gated", self.u6_downgrade_report),
            ("U7", "teardown, no residue", self.u7_teardown),
        ]
        t0 = time.time()
        print(f"[UPGRADE] source={self.source} work={self.work}")
        for sid, name, fn in steps:
            started = time.time()
            try:
                detail = fn()
            except StepFailure as exc:
                for proc in (self.daemon, self.stub):
                    if proc and proc.poll() is None:
                        proc.kill()
                print(f"\n[UPGRADE] FAILED at {sid} — {name}")
                print(f"[UPGRADE] why: {exc}")
                if exc.log and exc.log.exists():
                    tail = exc.log.read_text(
                        encoding="utf-8", errors="replace").splitlines()[-30:]
                    print("[UPGRADE] log tail:\n" + "\n".join(tail))
                print(f"[UPGRADE] logs kept at: {self.logs}")
                return 1
            print(f"[UPGRADE] {sid} ok ({time.time() - started:6.1f}s) — "
                  f"{name}: {detail}")
        print(f"\n[UPGRADE] PASS — {time.time() - t0:.1f}s, "
              f"{BASELINE_TAG} -> {self.head_sha[:12]}, "
              f"{len(self.findings)} known-defect finding(s)")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=".")
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--strict-state", action="store_true",
                        help="known-defect findings (FL-3, FL-2u) become "
                             "failures — flip on as each defect's fix lands")
    parser.add_argument("--mutate", default="none",
                        choices=["none", "drop-table", "delete-db"],
                        help="self-test levers: tamper with state so U5's "
                             "reporting can be verified")
    args = parser.parse_args()
    return UpgradeHarness(Path(args.source).resolve(), args.keep,
                          args.strict_state, args.mutate).main()


if __name__ == "__main__":
    sys.exit(main())
