"""scripts/distill/setups.py: the pure logic of GPU setups, with SSH mocked.

Nothing here reaches a host: the probe's parsers run on canned text, and the
apply loop runs against a fake runner that models systemctl and nvidia-smi.
"""

import json
import os
import shlex
import stat
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "distill"))

import setups as S  # noqa: E402

UNIT_DIR = "/srv/svc/.config/systemd/user"

# ── canned live state ───────────────────────────────────────────────────────

LLAMA_EXEC = ["/opt/llama/llama-server -m /models/big.gguf --port 8080 --host 127.0.0.1 -ngl 99"]


@pytest.fixture(autouse=True)
def clock(monkeypatch, tmp_path):
    """A fake clock that only moves when the code sleeps, and a private $HOME."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    c = {"t": 0.0, "sleeps": []}

    def sleep(s):
        c["sleeps"].append(s)
        c["t"] += s

    monkeypatch.setattr(S, "_sleep", sleep)
    monkeypatch.setattr(S, "_now", lambda: c["t"])
    return c


def proc(pid, unit, vram, scope="user", process="/opt/bin/x"):
    return {"gpu_uuid": "GPU-0", "pid": pid, "process": process, "vram_mib": vram,
            "scope": scope, "unit": unit}


def unit_info(active="active", ok=True, exec_start=None, ports=None, spec=None, load="loaded"):
    return {"active": active, "load": load, "files": [f"{UNIT_DIR}/x.service"],
            "exec_start": exec_start or [], "ports": ports or [],
            "health_spec": spec or {"type": "systemd", "unit": "x"},
            "health": {"ok": ok, "detail": "active" if ok else "down"}}


def live_host(procs, units, used=20000):
    return {"reachable": True, "ok": True,
            "gpus": [{"index": 0, "uuid": "GPU-0", "name": "Test GPU",
                      "vram_total_mib": 24564, "vram_used_mib": used}],
            "procs": procs, "units": units, "deadman_timers": []}


def business_live():
    return {
        "alpha": live_host(
            [proc(10, "llm.service", 20000, process="/opt/llama/llama-server")],
            {"llm.service": unit_info(exec_start=LLAMA_EXEC, ports=["127.0.0.1:8080"],
                                      spec={"type": "http", "url": "http://127.0.0.1:8080/health"})}),
        "beta": live_host(
            [proc(20, "chat.service", 11000), proc(21, "tts.service", 3000),
             proc(22, "ollama.service", 5000, process="/usr/bin/ollama")],
            {"chat.service": unit_info(), "tts.service": unit_info(),
             "ollama.service": unit_info(exec_start=["/usr/bin/ollama serve"])}),
    }


HOSTS = S.Hosts({"alpha": S.Host("alpha", "user@alpha-host"), "beta": S.Host("beta", "beta-host")})


def business_setup():
    return S.validate_setup(S.build_setup("business", business_live(), HOSTS, "2026-10-30T00:00:00+00:00"))


def flywheel_setup():
    b = business_setup()
    beta = b["hosts"]["beta"]
    return S.validate_setup({
        "name": "flywheel", "version": 1,
        "hosts": {
            "alpha": {"services": [], "allow_extra": ["distill-*"], "min_free_mib": 22000},
            "beta": {"services": [s for s in beta["services"] if s["unit"] == "tts.service"],
                     "allow_extra": ["distill-*"]},
        },
    })


# ── probe parsers ───────────────────────────────────────────────────────────


def test_unit_from_cgroup_user_and_system():
    user = "0::/user.slice/user-1000.slice/user@1000.service/app.slice/llm.service\n"
    assert S.unit_from_cgroup(user, 1000) == ("user", "llm.service")
    system = "0::/system.slice/ollama.service\n"
    assert S.unit_from_cgroup(system, 1000) == ("system", "ollama.service")
    bare = "0::/user.slice/user-1000.slice/session-3.scope\n"
    assert S.unit_from_cgroup(bare, 1000) == ("system", "session-3.scope")
    assert S.unit_from_cgroup("0::/\n", 1000) == ("system", None)


def test_unit_from_cgroup_other_account_and_user_scopes():
    other = "0::/user.slice/user-1001.slice/user@1001.service/app.slice/ollama.service\n"
    assert S.unit_from_cgroup(other, 1000) == ("other-user", "ollama.service")
    tmux = "0::/user.slice/user-1000.slice/user@1000.service/app.slice/tmux-spawn-1234.scope\n"
    assert S.unit_from_cgroup(tmux, 1000) == ("user", "tmux-spawn-1234.scope")


def test_parse_ss_listen():
    text = ("LISTEN 0 512 127.0.0.1:8080 0.0.0.0:* users:((\"llama-server\",pid=3732,fd=37))\n"
            "LISTEN 0 4096 0.0.0.0:22 0.0.0.0:*\n"
            "LISTEN 0 2048 [::]:9000 [::]:* users:((\"py\",pid=5,fd=3),(\"py\",pid=6,fd=3))\n")
    assert S.parse_ss_listen(text) == [(3732, "127.0.0.1:8080"), (5, "[::]:9000"), (6, "[::]:9000")]


def test_parse_list_units():
    text = ("distill-train.service loaded activating auto-restart distill train\n"
            "● distill-deadman-alpha.service loaded failed failed distill dead-man\n"
            "distill-quant.scope loaded active running quant\n"
            "distill-deadman-alpha.timer loaded active waiting timer\n")
    assert S.parse_list_units(text) == {"distill-train.service": "activating",
                                        "distill-deadman-alpha.service": "failed",
                                        "distill-quant.scope": "active"}


def test_parse_unit_cat_continuations_and_dropin_reset():
    text = (
        f"# {UNIT_DIR}/llm.service\n"
        "[Service]\n"
        "ExecStart=/opt/llama-server \\\n"
        "  -m /models/a.gguf \\\n"
        "  --port 8080\n"
        "\n"
        f"# {UNIT_DIR}/llm.service.d/override.conf\n"
        "[Service]\n"
        "ExecStart=\n"
        "ExecStart=/opt/llama-server -m /models/b.gguf --port 8081\n"
    )
    parsed = S.parse_unit_cat(text)
    assert parsed["files"] == [f"{UNIT_DIR}/llm.service", f"{UNIT_DIR}/llm.service.d/override.conf"]
    assert parsed["exec_start"] == ["/opt/llama-server -m /models/b.gguf --port 8081"]


def test_redact_unit_text():
    text = ('Environment=HF_TOKEN=hf_abc123 PLAIN=keep\n'
            'Environment="OPENAI_API_KEY=sk-xyz"\n'
            'ExecStart=/opt/srv --api-key s3cret --port 1\n')
    out = S.redact_unit_text(text)
    assert "hf_abc123" not in out and "sk-xyz" not in out and "s3cret" not in out
    assert "PLAIN=keep" in out and "--port 1" in out


@pytest.mark.parametrize("line, secret", [
    ("ExecStart=/opt/llama-server -hf org/m --hf-token hf_SPACE1 --port 8080", "hf_SPACE1"),
    ("ExecStart=/opt/llama-server -hf org/m -hft hf_SHORT2 --port 8080", "hf_SHORT2"),
    ("ExecStart=/opt/llama-server --hf-token=hf_EQ3 --port 8080", "hf_EQ3"),
    ("ExecStart=/opt/srv --token tok4 --port 8080", "tok4"),
    ("ExecStart=/opt/srv --password pw5 --port 8080", "pw5"),
    ("ExecStart=/opt/srv --auth-token at6 --port 8080", "at6"),
    ("ExecStart=/opt/srv --API-KEY UPPER7 --port 8080", "UPPER7"),
    ('ExecStart=/opt/srv --api-key "two words8" --port 8080', "words8"),
    ("ExecStart=/opt/srv -k shortk9 --port 8080", "shortk9"),
    ("ExecStart=/opt/srv --upstream https://user:pass10@api.example.com/v1 --port 8080", "pass10"),
    ("ExecStart=/opt/srv --upstream https://tok11@api.example.com/v1 --port 8080", "tok11"),
    ("ExecStart=/usr/bin/curl -H 'Authorization: Bearer bear12' http://x --port 8080", "bear12"),
    ("ExecStart=/usr/bin/curl -H 'X-Api-Key: hdr13' http://x --port 8080", "hdr13"),
    ('Environment="SERVICE_PASSWORD=two words14"', "words14"),
    ("Environment=DB_PASSWORD=pw15 MY_SECRET=s16", "pw15"),
    ("Environment=DB_PASSWORD=pw15 MY_SECRET=s16", "s16"),
    ("Environment=GH_AUTH=ga17", "ga17"),
    ("ExecStart=/usr/bin/curl -H 'X-Api-Key:nospace18' http://x --port 8080", "nospace18"),
    ('ExecStart=/usr/bin/curl -H "Authorization:Bearer bear19" http://x --port 8080', "bear19"),
    ("ExecStart=/usr/bin/curl --header=X-Auth-Token:hdr20 http://x --port 8080", "hdr20"),
    ("ExecStart=/opt/srv --pass pw21 --port 8080", "pw21"),
    ("ExecStart=/opt/srv --passphrase pp22 --port 8080", "pp22"),
    ("Environment=DB_PASS=pw23", "pw23"),
])
def test_redact_unit_text_secret_forms(line, secret):
    out = S.redact_unit_text(line)
    assert secret not in out
    assert "<redacted>" in out
    if "--port 8080" in line:
        assert "--port 8080" in out  # the health check still finds the port


def test_redact_keeps_a_flag_that_takes_no_value():
    out = S.redact_unit_text("ExecStart=/opt/srv --no-auth --port 8080 -m /models/a.gguf")
    assert out == "ExecStart=/opt/srv --no-auth --port 8080 -m /models/a.gguf"


@pytest.mark.parametrize("line", [
    "ExecStart=/opt/llama-server -hf org/tokenizer-model:Q4_K_M --port 8080",  # a repo tag, not a header
    "ExecStart=/opt/srv --pass-through upstream --port 8080",
    "Environment=PASS_COUNT=3",
])
def test_redact_leaves_lookalikes_alone(line):
    assert S.redact_unit_text(line) == line


@pytest.mark.parametrize("assign", ["HF_TOKEN=$(cat ~/.hf)", "HF_TOKEN=hf_INLINE24"])
def test_redact_sh_c_assignment_keeps_the_command(assign):
    # The quotes wrap a whole shell command here, not one assignment: only the value goes.
    text = (f"# {UNIT_DIR}/llm.service\n"
            f'ExecStart=/bin/sh -c "{assign} exec /opt/llama-server -hf org/m --port 8080"\n')
    out = S.redact_unit_text(text)
    assert "hf_INLINE24" not in out and "$(cat" not in out and "HF_TOKEN=<redacted>" in out
    execs = S.parse_unit_cat(out)["exec_start"]
    assert "llama-server" in execs[0] and "--port 8080" in execs[0]
    assert S.guess_health("llm.service", execs, []) == {"type": "http",
                                                        "url": "http://127.0.0.1:8080/health"}
    assert S.guess_model(execs) == "org/m"


def test_redact_environment_quotes_still_wrap_the_whole_value():
    out = S.redact_unit_text('Environment="A=1" "API_TOKEN=two words25" "B=2"\n')
    assert "words25" not in out and '"A=1"' in out and '"B=2"' in out


def test_guess_health():
    spec = S.guess_health("llm.service", LLAMA_EXEC, ["127.0.0.1:8080"])
    assert spec == {"type": "http", "url": "http://127.0.0.1:8080/health"}
    wildcard = S.guess_health("llm.service", ["/opt/llama-server --port 9"], ["0.0.0.0:9"])
    assert wildcard["url"] == "http://127.0.0.1:9/health"
    assert S.guess_health("tts.service", ["/opt/tts"], ["0.0.0.0:8001"]) == {
        "type": "systemd", "unit": "tts.service"}


def test_probe_source_compiles_standalone():
    compile(S.probe_source(), "<probe>", "exec")


def test_probe_main_lists_jobs_classifies_by_uid_and_reads_the_last_apply(tmp_path, monkeypatch):
    uid = 1000
    cgroups = {
        10: f"0::/user.slice/user-{uid}.slice/user@{uid}.service/app.slice/llm.service\n",
        11: "0::/user.slice/user-1001.slice/user@1001.service/app.slice/theirs.service\n",
        12: f"0::/user.slice/user-{uid}.slice/user@{uid}.service/app.slice/tmux-spawn-1.scope\n",
    }
    outputs = {
        "--query-gpu": "0, GPU-0, Test GPU, 24564, 21000\n",
        "--query-compute-apps": "GPU-0, 10, /opt/llama-server, 20000\nGPU-0, 11, /opt/x, 500\n"
                                "GPU-0, 12, /opt/y, 500\n",
        "list-units": "distill-pipeline.service loaded activating auto-restart pipeline\n"
                      "distill-old.service loaded inactive dead old\n"
                      "distill-deadman-alpha.service loaded failed failed dead-man\n",
        "cat": f"# {UNIT_DIR}/u.service\nExecStart=/opt/srv --hf-token hf_SECRET --port 9\n",
    }

    def fake_run(argv, timeout=20):
        if argv[0] == "nvidia-smi":
            return {"rc": 0, "out": outputs[argv[1].split("=")[0]], "err": ""}
        if argv[:3] == ["systemctl", "--user", "is-active"]:
            return {"rc": 0, "out": "active\n", "err": ""}
        if argv[:5] == ["systemctl", "--user", "show", "-p", "LoadState"]:
            return {"rc": 0, "out": "loaded\n", "err": ""}
        for key in ("list-units", "cat"):
            if argv[:3] == ["systemctl", "--user", key]:
                return {"rc": 0, "out": outputs[key], "err": ""}
        return {"rc": 0, "out": "", "err": ""}

    real_open = open

    def fake_open(path, *a, **kw):
        if str(path).startswith("/proc/"):
            from io import StringIO
            return StringIO(cgroups[int(str(path).split("/")[2])])
        return real_open(path, *a, **kw)

    rel = S.REMOTE_DIR + "/" + S.LAST_APPLY
    (tmp_path / "home" / S.REMOTE_DIR).mkdir(parents=True)
    (tmp_path / "home" / rel).write_text(json.dumps({"setup": "business", "result": "failed"}))
    monkeypatch.setattr(S, "_run", fake_run)
    monkeypatch.setattr(S, "open", fake_open, raising=False)
    monkeypatch.setattr(S, "check_health", lambda spec: {"ok": True, "detail": "active"})
    monkeypatch.setattr(S.os, "getuid", lambda: uid)
    out = S.probe_main({"units": [], "health": {}, "last_apply": rel})
    scopes = {p["pid"]: (p["scope"], p["unit"]) for p in out["procs"]}
    assert scopes == {10: ("user", "llm.service"), 11: ("other-user", "theirs.service"),
                      12: ("user", "tmux-spawn-1.scope")}
    assert set(out["units"]) == {"llm.service", "tmux-spawn-1.scope", "distill-pipeline.service"}
    assert out["deadman_units"] == {"distill-deadman-alpha.service": "failed"}
    assert out["units"]["llm.service"]["load"] == "loaded"
    assert out["last_apply"]["result"] == "failed"
    assert "hf_SECRET" not in json.dumps(out)


# ── setups: build + YAML round trip ─────────────────────────────────────────


def test_build_setup_shape():
    s = business_setup()
    alpha = s["hosts"]["alpha"]
    assert [x["unit"] for x in alpha["services"]] == ["llm.service"]
    llm = alpha["services"][0]
    assert llm["model"] == "/models/big.gguf"
    assert llm["health"]["type"] == "http"
    assert llm["gpu_on_demand"] is False
    beta = {x["unit"]: x for x in s["hosts"]["beta"]["services"]}
    assert list(beta) == ["chat.service", "ollama.service", "tts.service"]  # by VRAM, largest first
    assert beta["ollama.service"]["gpu_on_demand"] is True


def test_build_setup_records_unmanaged_processes():
    live = {"alpha": live_host([proc(9, "something.service", 100, scope="system")], {})}
    s = S.build_setup("x", live, HOSTS)
    assert s["hosts"]["alpha"]["services"] == []
    assert s["hosts"]["alpha"]["unmanaged_gpu_processes"][0]["pid"] == 9


def test_build_setup_puts_user_scopes_and_other_accounts_in_unmanaged():
    live = {"alpha": live_host(
        [proc(1, "llm.service", 20000), proc(2, "tmux-spawn-1.scope", 900),
         proc(3, "ollama.service", 500, scope="other-user")],
        {"llm.service": unit_info(), "tmux-spawn-1.scope": unit_info()})}
    s = S.validate_setup(S.build_setup("x", live, HOSTS))  # used to raise ValueError on the scope
    assert [x["unit"] for x in s["hosts"]["alpha"]["services"]] == ["llm.service"]
    assert {u["pid"] for u in s["hosts"]["alpha"]["unmanaged_gpu_processes"]} == {2, 3}


def test_yaml_round_trip(tmp_path):
    s = business_setup()
    S.write_private(tmp_path / "business.yaml", S.dump_setup(s))
    assert (tmp_path / "business.yaml").stat().st_mode & 0o077 == 0
    assert S.load_setup(tmp_path, "business") == s
    assert set(S.list_setups(tmp_path)) == {"business"}


def test_write_private_tightens_an_existing_loose_file(tmp_path):
    p = tmp_path / "flywheel.yaml"
    p.write_text("hand-written\n")
    os.chmod(p, 0o644)
    S.write_private(p, "saved\n")
    assert stat.S_IMODE(p.stat().st_mode) == 0o600
    assert p.read_text() == "saved\n"


def test_list_setups_skips_hosts_file(tmp_path):
    (tmp_path / "hosts.yaml").write_text("hosts: {alpha: {ssh: a}}\n")
    assert S.list_setups(tmp_path) == {}


@pytest.mark.parametrize("bad", [
    {"name": "x", "hosts": {"alpha": {"services": [{"unit": "rm -rf /.service"}]}}},
    {"name": "x", "hosts": {"alpha": {"services": [{"unit": "a.service"}, {"unit": "a.service"}]}}},
    {"name": "x", "hosts": {"alpha": {"services": [{"unit": "distill-x.scope"}]}}},
    {"name": "x", "hosts": {"alpha": {"stop": ["x; reboot"]}}},
    {"name": "../x", "hosts": {}},
    {"name": "x", "hosts": {"Bad Host": {}}},
    {"name": "x"},
])
def test_validate_rejects(bad):
    with pytest.raises(ValueError):
        S.validate_setup(bad)


def test_load_hosts(tmp_path):
    p = tmp_path / "hosts.yaml"
    p.write_text("hosts:\n  alpha: {ssh: a, never_stop: [gpu-web.service], python: /opt/py/bin/python3,"
                 " notify_cmd: /opt/notify}\n  beta: {ssh: b}\n"
                 "never_stop: [site.service]\n"
                 "business_units: {beta: [model-server.service]}\n")
    h = S.load_hosts(p)
    assert h.hosts["alpha"].ssh == "a"
    guard = h.guard("alpha")
    assert guard[: len(S.NEVER_STOP)] == S.NEVER_STOP
    assert "site.service" in guard and "gpu-web.service" in guard
    assert h.hosts["alpha"].python == "/opt/py/bin/python3" and h.hosts["beta"].python is None
    assert h.hosts["alpha"].notify_cmd == "/opt/notify"
    assert h.business_units == {"beta": ("model-server.service",)}


def test_load_hosts_rejects_bad_business_unit(tmp_path):
    p = tmp_path / "hosts.yaml"
    p.write_text("hosts: {beta: {ssh: b}}\nbusiness_units: {beta: ['x; reboot']}\n")
    with pytest.raises(SystemExit):
        S.load_hosts(p)


# ── status matching ─────────────────────────────────────────────────────────


def test_status_matches_business_only():
    live = business_live()
    assert S.match_setup(business_setup(), live) == []
    reasons = S.match_setup(flywheel_setup(), live)
    assert any("extra GPU unit llm.service" in r for r in reasons)
    assert any("extra GPU unit chat.service" in r for r in reasons)


def test_status_on_demand_unit_without_gpu_still_matches():
    live = business_live()
    live["beta"]["procs"] = [p for p in live["beta"]["procs"] if p["unit"] != "ollama.service"]
    assert S.match_setup(business_setup(), live) == []


def test_status_inactive_unit_and_unreachable_host():
    live = business_live()
    live["beta"]["units"]["tts.service"]["active"] = "failed"
    live["beta"]["procs"] = [p for p in live["beta"]["procs"] if p["unit"] != "tts.service"]
    live["alpha"] = {"reachable": False, "ok": False, "error": "timeout"}
    reasons = S.match_setup(business_setup(), live)
    assert "alpha: not reachable" in reasons
    assert "beta: tts.service is failed" in reasons


def test_status_flywheel_with_training_job_matches():
    live = {
        "alpha": live_host([proc(30, "distill-train.service", 21000)],
                           {"distill-train.service": unit_info()}),
        "beta": live_host([proc(21, "tts.service", 3000), proc(40, "distill-eval.service", 9000)],
                          {"tts.service": unit_info(), "distill-eval.service": unit_info()}),
    }
    assert S.match_setup(flywheel_setup(), live) == []
    assert any("distill-train" in r for r in S.match_setup(business_setup(), live))


# ── plans + never-stop ──────────────────────────────────────────────────────


def test_plan_business_to_flywheel():
    plans = S.compute_plan(flywheel_setup(), business_live())
    assert plans["alpha"].stop == ["llm.service"] and plans["alpha"].start == []
    assert plans["beta"].stop == ["chat.service", "ollama.service"]
    assert plans["beta"].keep == ["tts.service"] and plans["beta"].start == []
    assert not any(p.refused or p.blockers for p in plans.values())


def test_plan_flywheel_to_business_stops_training_and_starts_services():
    live = {
        "alpha": live_host([proc(30, "distill-train.service", 21000)],
                           {"distill-train.service": unit_info(),
                            "llm.service": unit_info(active="inactive")}),
        "beta": live_host([proc(21, "tts.service", 3000)],
                          {"tts.service": unit_info(),
                           "chat.service": unit_info(active="inactive"),
                           "ollama.service": unit_info(active="inactive")}),
    }
    plans = S.compute_plan(business_setup(), live)
    assert plans["alpha"].stop == ["distill-train.service"]
    assert plans["alpha"].start == ["llm.service"]
    assert plans["beta"].start == ["chat.service", "ollama.service"]
    assert plans["beta"].keep == ["tts.service"]


def test_plan_is_noop_when_already_there():
    plans = S.compute_plan(business_setup(), business_live())
    assert all(not p.stop and not p.start for p in plans.values())


def test_plan_refuses_never_stop_units():
    live = business_live()
    live["beta"]["units"]["prometheus.service"] = unit_info()
    live["beta"]["units"]["cloudflared-tunnel.service"] = unit_info()
    plans = S.compute_plan(flywheel_setup(), live)
    assert plans["beta"].refused == ["cloudflared-tunnel.service", "prometheus.service"]
    assert "prometheus.service" not in plans["beta"].stop


def test_plan_refuses_site_guard_from_hosts_file():
    guards = {"alpha": S.NEVER_STOP + ("llm.service",), "beta": S.NEVER_STOP}
    plans = S.compute_plan(flywheel_setup(), business_live(), guards)
    assert plans["alpha"].refused == ["llm.service"] and plans["alpha"].stop == []


def test_plan_blocks_on_unreachable_and_unmanaged():
    live = business_live()
    live["alpha"] = {"reachable": False, "ok": False, "error": "timeout"}
    live["beta"]["procs"].append(proc(99, None, 500, scope="system"))
    plans = S.compute_plan(flywheel_setup(), live)
    assert plans["alpha"].blockers and "not reachable" in plans["alpha"].blockers[0]
    assert any("pid 99" in b for b in plans["beta"].blockers)


@pytest.mark.parametrize("unit, scope", [
    ("tmux-spawn-1234.scope", "user"),        # tmux pane under the user manager
    ("run-u42.scope", "user"),                # systemd-run --user --scope
    ("distill-a\\x2db.scope", "user"),        # escaped name: not a plain name
    ("ollama.service", "other-user"),         # another account's manager
    ("session-3.scope", "system"),            # nohup / tmux from an SSH login
])
def test_plan_blocks_gpu_holders_it_cannot_stop_before_anything_runs(unit, scope):
    live = business_live()
    live["alpha"]["procs"].append(proc(77, unit, 900, scope=scope))
    if scope == "user":
        live["alpha"]["units"][unit] = unit_info()
    plans = S.compute_plan(business_setup(), live)
    assert any("pid 77" in b for b in plans["alpha"].blockers)
    assert unit not in plans["alpha"].stop


def test_plan_stops_a_distill_scope():
    live = business_live()
    live["alpha"]["procs"].append(proc(78, "distill-eval.scope", 900))
    live["alpha"]["units"]["distill-eval.scope"] = unit_info()
    plans = S.compute_plan(business_setup(), live)
    assert plans["alpha"].stop == ["distill-eval.scope"] and not plans["alpha"].blockers
    # ...and the flywheel allows it to keep running
    assert S.compute_plan(flywheel_setup(), live)["alpha"].stop == ["llm.service"]


def test_plan_non_restore_refuses_units_business_did_not_capture():
    captured = {"alpha": {"llm.service"}, "beta": {"chat.service", "tts.service"}}
    plans = S.compute_plan(flywheel_setup(), business_live(), captured=captured)
    assert plans["beta"].uncaptured == ["ollama.service"]
    assert plans["beta"].stop == ["chat.service"]
    fw = flywheel_setup()
    fw["hosts"]["beta"]["stop"] = ["ollama.service"]
    plans = S.compute_plan(fw, business_live(), captured=captured)
    assert plans["beta"].uncaptured == [] and plans["beta"].stop == ["chat.service", "ollama.service"]


def test_plan_retry_restarts_unhealthy_or_off_gpu_targets():
    live = business_live()
    live["alpha"]["units"]["llm.service"]["health"] = {"ok": False, "detail": "HTTP 503"}
    live["beta"]["procs"] = [p for p in live["beta"]["procs"] if p["unit"] != "tts.service"]
    plans = S.compute_plan(business_setup(), live, restart_unhealthy=True)
    assert plans["alpha"].restart == ["llm.service"]
    assert plans["beta"].restart == ["tts.service"]
    assert "ollama.service" in plans["beta"].keep
    assert S.compute_plan(business_setup(), live)["alpha"].restart == []  # first attempt: no restarts
    guards = {"alpha": S.NEVER_STOP + ("llm.service",), "beta": S.NEVER_STOP}
    plans = S.compute_plan(business_setup(), live, guards, restart_unhealthy=True)
    assert plans["alpha"].restart == [] and plans["alpha"].notes


@pytest.mark.parametrize("unit", ["prometheus.service", "ssh.service", "sshd.service",
                                  "tailscaled.service", "cloudflared.service",
                                  "distill-deadman-alpha.service"])
def test_never_stop_constant(unit):
    assert S.is_never_stop(unit)


def test_the_constant_holds_only_generic_names():
    # Site-specific units belong in the gitignored hosts.yaml never_stop, not here.
    generic = {"prometheus.service", "prometheus-*.service", "cloudflared*", "tailscale*", "ssh*",
               "sshd*", "distill-deadman-*", "dbus*", "systemd-*", "pipewire*", "wireplumber*"}
    assert set(S.NEVER_STOP) <= generic


def test_systemctl_guard_refuses_at_execute_layer():
    runner = FakeRunner({})
    with pytest.raises(RuntimeError):
        S.systemctl_user(runner, "beta", "stop", "prometheus.service", S.NEVER_STOP)
    with pytest.raises(RuntimeError):
        S.systemctl_user(runner, "beta", "restart", "prometheus.service", S.NEVER_STOP)
    with pytest.raises(ValueError):
        S.systemctl_user(runner, "beta", "stop", "x; reboot", S.NEVER_STOP)
    with pytest.raises(ValueError):
        S.systemctl_user(runner, "beta", "stop", "tmux-spawn-1.scope", S.NEVER_STOP)
    with pytest.raises(ValueError):
        S.systemctl_user(runner, "beta", "start", "distill-x.scope", S.NEVER_STOP)
    assert runner.calls == []


def test_systemctl_stops_a_distill_scope():
    runner = FakeRunner({"beta": {"units": {"distill-x.scope": ("active", 0)}}})
    S.systemctl_user(runner, "beta", "stop", "distill-x.scope", S.NEVER_STOP)
    assert runner.calls == [("beta", ["systemctl", "--user", "stop", "distill-x.scope"])]


# ── execute loop against a fake host ────────────────────────────────────────


class FakeRunner:
    """Models systemctl start/stop/restart + nvidia-smi for the probe's JSON.

    state: host -> {"units": {unit: (active, vram) or (active, vram, healthy)}}.
    """

    def __init__(self, state, extra_procs=None):
        self.state = state
        self.extra_procs = extra_procs or {}
        self.calls = []
        self.shell = []

    def python(self, host):
        return "python3"

    def command(self, host, argv):
        return argv

    def _unit(self, host, u):
        st = self.state[host]["units"].get(u, ("inactive", 0))
        return (st + (True,))[:3]

    def run(self, host, argv, input=None, timeout=90):
        if argv[:2] == ["python3", "-"]:
            req = json.loads(argv[2])
            units = self.state[host]["units"]
            procs = [proc(i, u, v) for i, (u, st) in enumerate(units.items())
                     for (a, v, _) in [self._unit(host, u)] if a == "active" and v]
            procs += self.extra_procs.get(host, [])
            jobs = {u for u in units if u.startswith("distill-") and not u.startswith("distill-deadman-")}
            names = set(req["units"]) | {p["unit"] for p in procs if p["scope"] == "user"} | jobs
            info = {}
            for u in names:
                a, _, healthy = self._unit(host, u)
                info[u] = unit_info(active=a, ok=a == "active" and healthy)
            out = json.dumps(live_host(procs, info, used=sum(p["vram_mib"] for p in procs)))
            return subprocess.CompletedProcess(argv, 0, out + "\n", "")
        if argv[0] == "sh":
            self.shell.append((host, argv, input))
            return subprocess.CompletedProcess(argv, 0, "", "")
        self.calls.append((host, argv))
        verb, unit = argv[2], argv[3]
        _, vram, healthy = self._unit(host, unit)
        self.state[host]["units"][unit] = ("inactive" if verb == "stop" else "active", vram, healthy)
        return subprocess.CompletedProcess(argv, 0, "", "")


def write_setups(tmp_path):
    for s in (business_setup(), flywheel_setup()):
        S.write_private(tmp_path / f"{s['name']}.yaml", S.dump_setup(s))


def args_for(tmp_path, name, execute):
    return type("A", (), {"setups_dir": tmp_path, "name": name, "only_host": None,
                          "execute": execute})()


def business_state():
    return {"alpha": {"units": {"llm.service": ("active", 20000)}},
            "beta": {"units": {"chat.service": ("active", 11000), "tts.service": ("active", 3000),
                               "ollama.service": ("active", 5000)}}}


def test_apply_execute_reaches_flywheel_and_back(tmp_path):
    write_setups(tmp_path)
    state = business_state()
    runner = FakeRunner(state)
    assert S.cmd_apply(args_for(tmp_path, "flywheel", True), HOSTS, runner) == 0
    assert ("alpha", ["systemctl", "--user", "stop", "llm.service"]) in runner.calls
    assert state["beta"]["units"]["tts.service"][0] == "active"
    runner.calls.clear()
    assert S.cmd_apply(args_for(tmp_path, "business", True), HOSTS, runner) == 0
    assert all(state["beta"]["units"][u][0] == "active" for u in ("chat.service", "ollama.service"))
    assert state["alpha"]["units"]["llm.service"][0] == "active"
    assert not any(c[1][2] == "stop" for c in runner.calls)
    record = json.loads(S.last_apply_path().read_text())
    assert record["setup"] == "business" and record["result"] == "ok"


def test_apply_never_stops_the_daemon_even_when_it_holds_vram(tmp_path):
    # The daemon holds VRAM and business (hand-edited) lists it, so nothing but
    # the never-stop guard keeps the flywheel from stopping it.
    write_setups(tmp_path)
    s = business_setup()
    s["hosts"]["beta"]["services"].append({"unit": "prometheus.service"})
    S.write_private(tmp_path / "business.yaml", S.dump_setup(S.validate_setup(s)))
    state = business_state()
    state["beta"]["units"]["prometheus.service"] = ("active", 700)
    runner = FakeRunner(state)
    assert S.cmd_apply(args_for(tmp_path, "flywheel", True), HOSTS, runner) == 2
    assert runner.calls == []
    assert state["beta"]["units"]["prometheus.service"][0] == "active"


def test_execute_layer_guard_holds_when_a_plan_names_the_daemon(tmp_path, monkeypatch):
    write_setups(tmp_path)
    state = business_state()
    state["alpha"]["units"]["llm.service"] = ("inactive", 20000)
    state["beta"]["units"]["prometheus.service"] = ("active", 0)  # not a candidate: only the bug adds it
    runner = FakeRunner(state)
    real = S.compute_plan

    def bad_plan(*a, **kw):  # a planning bug that puts the daemon in "stop"
        plans = real(*a, **kw)
        plans["beta"].stop.append("prometheus.service")
        return plans

    monkeypatch.setattr(S, "compute_plan", bad_plan)
    assert S.cmd_apply(args_for(tmp_path, "business", True), HOSTS, runner) == 1
    assert ("beta", ["systemctl", "--user", "stop", "prometheus.service"]) not in runner.calls
    assert state["beta"]["units"]["prometheus.service"][0] == "active"
    # the refusal is a failed attempt, not a crash: the start still ran
    assert ("alpha", ["systemctl", "--user", "start", "llm.service"]) in runner.calls


def test_apply_dry_run_changes_nothing(tmp_path):
    write_setups(tmp_path)
    runner = FakeRunner(business_state())
    assert S.cmd_apply(args_for(tmp_path, "flywheel", False), HOSTS, runner) == 0
    assert runner.calls == []
    assert not S.last_apply_path().exists()


def test_apply_retries_then_fails(tmp_path, clock):
    write_setups(tmp_path)

    class Broken(FakeRunner):
        def run(self, host, argv, input=None, timeout=90):
            cp = super().run(host, argv, input, timeout)
            if argv[:3] == ["systemctl", "--user", "start"] and argv[3] == "llm.service":
                self.state[host]["units"]["llm.service"] = ("failed", 0)
            return cp

    state = business_state()
    state["alpha"]["units"]["llm.service"] = ("inactive", 20000)
    runner = Broken(state)
    assert S.cmd_apply(args_for(tmp_path, "business", True), HOSTS, runner) == 1
    starts = [c for c in runner.calls if c[1][2:] == ["start", "llm.service"]]
    assert len(starts) == 1 + S.RESTORE_RETRIES
    assert S.RETRY_DELAY_S >= 30 and clock["sleeps"].count(S.RETRY_DELAY_S) == S.RESTORE_RETRIES


def test_apply_retry_restarts_an_active_but_unhealthy_unit(tmp_path):
    write_setups(tmp_path)

    class Sick(FakeRunner):  # llm comes up unhealthy; only a restart fixes it
        def run(self, host, argv, input=None, timeout=90):
            cp = super().run(host, argv, input, timeout)
            if argv[:2] == ["systemctl", "--user"] and argv[3] == "llm.service":
                self.state[host]["units"]["llm.service"] = ("active", 20000, argv[2] == "restart")
            return cp

    state = business_state()
    state["alpha"]["units"]["llm.service"] = ("inactive", 20000)
    runner = Sick(state)
    assert S.cmd_apply(args_for(tmp_path, "business", True), HOSTS, runner) == 0
    verbs = [c[1][2] for c in runner.calls if c[1][3] == "llm.service"]
    assert verbs == ["start", "restart"]


def test_apply_systemctl_error_still_starts_and_retries(tmp_path):
    write_setups(tmp_path)

    class Flaky(FakeRunner):  # the first stop of the job raises (ssh died, say)
        raised = False

        def run(self, host, argv, input=None, timeout=90):
            if argv[2:] == ["stop", "distill-train.service"] and not self.raised:
                Flaky.raised = True
                raise OSError("connection reset")
            return super().run(host, argv, input, timeout)

    state = business_state()
    state["alpha"]["units"] = {"llm.service": ("inactive", 20000), "distill-train.service": ("active", 21000)}
    runner = Flaky(state)
    assert S.cmd_apply(args_for(tmp_path, "business", True), HOSTS, runner) == 0
    start = ("alpha", ["systemctl", "--user", "start", "llm.service"])
    stop = ("alpha", ["systemctl", "--user", "stop", "distill-train.service"])
    # the start ran in the same attempt as the failed stop; the retry then stopped the job
    assert runner.calls.index(start) < runner.calls.index(stop)
    assert state["alpha"]["units"]["distill-train.service"][0] == "inactive"


class ProbeRaises(FakeRunner):
    """The probe's own transport raises (ssh missing, say), outside any systemctl call."""

    def __init__(self, state, times):
        super().__init__(state)
        self.left = times

    def run(self, host, argv, input=None, timeout=90):
        if argv[:2] == ["python3", "-"] and self.left:
            self.left -= 1
            raise OSError("ssh: No such file or directory")
        return super().run(host, argv, input, timeout)


def test_apply_attempt_that_raises_outside_systemctl_is_retried_and_recorded(tmp_path, clock):
    write_setups(tmp_path)
    state = business_state()
    state["alpha"]["units"]["llm.service"] = ("inactive", 20000)
    runner = ProbeRaises(state, times=1)
    assert S.cmd_apply(args_for(tmp_path, "business", True), HOSTS, runner) == 0
    assert clock["sleeps"][0] == S.RETRY_DELAY_S  # the retry waited, then ran
    assert ("alpha", ["systemctl", "--user", "start", "llm.service"]) in runner.calls
    record = json.loads(S.last_apply_path().read_text())
    assert record["result"] == "ok" and record["attempts"] == 2


def test_apply_whose_every_attempt_raises_fails_records_and_notifies(tmp_path, clock):
    write_setups(tmp_path)
    hosts = S.Hosts({"alpha": S.Host("alpha", "a", notify_cmd="/opt/notify"), "beta": S.Host("beta", "b")})
    runner = ProbeRaises(business_state(), times=99)
    assert S.cmd_apply(args_for(tmp_path, "business", True), hosts, runner) == 1
    assert clock["sleeps"] == [S.RETRY_DELAY_S] * S.RESTORE_RETRIES
    record = json.loads(S.last_apply_path().read_text())
    assert record["result"] == "failed" and record["attempts"] == 1 + S.RESTORE_RETRIES
    assert any("OSError" in p for p in record["problems"])
    assert [h for h, _, _ in runner.shell] == ["alpha"] and "OSError" in runner.shell[0][2]


def test_restore_stops_distill_jobs_that_hold_no_vram(tmp_path):
    write_setups(tmp_path)
    state = business_state()
    state["alpha"]["units"]["distill-pipeline.service"] = ("activating", 0)  # CPU phase / auto-restart
    state["beta"]["units"]["distill-quant.scope"] = ("active", 0)
    runner = FakeRunner(state)
    assert S.cmd_apply(args_for(tmp_path, "business", True), HOSTS, runner) == 0
    assert ("alpha", ["systemctl", "--user", "stop", "distill-pipeline.service"]) in runner.calls
    assert ("beta", ["systemctl", "--user", "stop", "distill-quant.scope"]) in runner.calls


def test_flywheel_refuses_to_stop_what_business_did_not_capture(tmp_path):
    # business was saved while ollama was idle, so it is not in business.
    s = business_setup()
    s["hosts"]["beta"]["services"] = [x for x in s["hosts"]["beta"]["services"]
                                      if x["unit"] != "ollama.service"]
    S.write_private(tmp_path / "business.yaml", S.dump_setup(S.validate_setup(s)))
    S.write_private(tmp_path / "flywheel.yaml", S.dump_setup(flywheel_setup()))
    runner = FakeRunner(business_state())
    assert S.cmd_apply(args_for(tmp_path, "flywheel", True), HOSTS, runner) == 2
    assert runner.calls == []
    fw = flywheel_setup()
    fw["hosts"]["beta"]["stop"] = ["ollama.service"]
    S.write_private(tmp_path / "flywheel.yaml", S.dump_setup(S.validate_setup(fw)))
    assert S.cmd_apply(args_for(tmp_path, "flywheel", True), HOSTS, runner) == 0
    assert ("beta", ["systemctl", "--user", "stop", "ollama.service"]) in runner.calls


def test_blocked_restore_waits_between_attempts_then_records_and_notifies(tmp_path, clock, capsys):
    write_setups(tmp_path)
    hosts = S.Hosts({"alpha": S.Host("alpha", "a", notify_cmd="/opt/notify"),
                     "beta": S.Host("beta", "b")})
    state = business_state()
    state["alpha"]["units"] = {"llm.service": ("inactive", 20000), "distill-train.service": ("active", 21000)}
    runner = FakeRunner(state, extra_procs={"alpha": [proc(99, "session-4.scope", 8000, scope="system")]})
    assert S.cmd_apply(args_for(tmp_path, "business", True), hosts, runner) == 1
    assert runner.calls == []  # blocked before any stop
    assert clock["sleeps"] == [S.RETRY_DELAY_S] * S.RESTORE_RETRIES
    record = json.loads(S.last_apply_path().read_text())
    assert record["result"] == "failed" and record["attempts"] == 1 + S.RESTORE_RETRIES
    assert any("pid 99" in p for p in record["problems"])
    assert stat.S_IMODE(S.last_apply_path().stat().st_mode) == 0o600
    [(host, argv, msg)] = runner.shell
    assert host == "alpha" and argv == ["sh", "-c", "/opt/notify"]
    assert "business FAILED on alpha" in msg and "pid 99" in msg
    assert len(msg.splitlines()) <= 5
    assert "beta: no notify_cmd in hosts.yaml: nobody was notified" in capsys.readouterr().out


def test_failure_message_is_at_most_five_lines():
    msg = S.failure_message("business", "alpha", [f"alpha: problem {i}\nsecond line" for i in range(9)],
                            3, "failed")
    lines = msg.splitlines()
    assert len(lines) == 5 and "6 more" in lines[-1]


def test_apply_refused_plan_executes_nothing(tmp_path):
    write_setups(tmp_path)
    runner = FakeRunner({"alpha": {"units": {"llm.service": ("active", 20000)}},
                         "beta": {"units": {"chat.service": ("active", 11000),
                                            "tts.service": ("active", 3000)}}})
    # A (hand-edited) setup naming the daemon makes every other setup a candidate to stop it.
    s = business_setup()
    s["hosts"]["beta"]["services"].append({"unit": "prometheus.service"})
    S.write_private(tmp_path / "business.yaml", S.dump_setup(S.validate_setup(s)))
    runner.state["beta"]["units"]["prometheus.service"] = ("active", 0)
    assert S.cmd_apply(args_for(tmp_path, "flywheel", True), HOSTS, runner) == 2
    assert runner.calls == []
    assert json.loads(S.last_apply_path().read_text())["result"] == "refused"


# ── save + status commands ──────────────────────────────────────────────────


class StaticRunner:
    """Answers every probe with a fixed live state per host."""

    def __init__(self, live):
        self.live = live
        self.reqs = []

    def python(self, host):
        return "python3"

    def run(self, host, argv, input=None, timeout=90):
        self.reqs.append((host, json.loads(argv[2])))
        return subprocess.CompletedProcess(argv, 0, json.dumps(self.live[host]) + "\n", "")


def save_args(tmp_path, name):
    return type("A", (), {"setups_dir": tmp_path, "name": name, "only_host": None, "force": True})()


def test_save_with_a_user_scope_on_the_gpu(tmp_path):
    live = business_live()
    live["alpha"]["procs"].append(proc(5, "tmux-spawn-9.scope", 900))
    live["alpha"]["units"]["tmux-spawn-9.scope"] = unit_info()
    assert S.cmd_save(save_args(tmp_path, "business"), HOSTS, StaticRunner(live)) == 0
    saved = S.load_setup(tmp_path, "business")
    assert [s["unit"] for s in saved["hosts"]["alpha"]["services"]] == ["llm.service"]
    assert saved["hosts"]["alpha"]["unmanaged_gpu_processes"][0]["unit"] == "tmux-spawn-9.scope"


def test_save_business_captures_idle_business_units(tmp_path):
    live = business_live()
    live["beta"]["procs"] = [p for p in live["beta"]["procs"] if p["unit"] != "ollama.service"]
    hosts = S.Hosts(dict(HOSTS.hosts), business_units={"beta": ("ollama.service",)})
    runner = StaticRunner(live)
    assert S.cmd_save(save_args(tmp_path, "business"), hosts, runner) == 0
    assert ("beta", ["ollama.service"]) in [(h, r["units"]) for h, r in runner.reqs]
    beta = {s["unit"]: s for s in S.load_setup(tmp_path, "business")["hosts"]["beta"]["services"]}
    assert beta["ollama.service"]["vram_mib"] == 0 and beta["ollama.service"]["gpu_on_demand"] is True
    # other setups do not pick them up
    assert S.cmd_save(save_args(tmp_path, "other"), hosts, StaticRunner(live)) == 0
    other = S.load_setup(tmp_path, "other")["hosts"]["beta"]["services"]
    assert "ollama.service" not in {s["unit"] for s in other}


@pytest.mark.parametrize("load", ["not-found", "masked"])
def test_save_business_refuses_a_business_unit_the_user_manager_has_not_loaded(tmp_path, capsys, load):
    # A typo or a system unit: is-active says "inactive", and a restore could never start it.
    live = business_live()
    live["beta"]["units"]["model-server.service"] = unit_info(active="inactive", ok=False, load=load)
    hosts = S.Hosts(dict(HOSTS.hosts), business_units={"beta": ("model-server.service",)})
    assert S.cmd_save(save_args(tmp_path, "business"), hosts, StaticRunner(live)) == 1
    assert not (tmp_path / "business.yaml").exists()
    err = capsys.readouterr().err
    assert "REFUSED" in err and "model-server.service" in err and load in err


def test_save_business_skips_a_stopped_business_unit_so_status_matches(tmp_path, capsys):
    live = business_live()
    live["beta"]["units"]["model-server.service"] = unit_info(active="inactive", ok=False)
    hosts = S.Hosts(dict(HOSTS.hosts), business_units={"beta": ("ollama.service", "model-server.service")})
    assert S.cmd_save(save_args(tmp_path, "business"), hosts, StaticRunner(live)) == 0
    assert "WARNING" in capsys.readouterr().err
    saved = S.load_setup(tmp_path, "business")
    units = {s["unit"] for s in saved["hosts"]["beta"]["services"]}
    assert "model-server.service" not in units and "ollama.service" in units
    assert S.match_setup(saved, live) == []  # status shows business as a MATCH right after saving
    assert all(not p.start for p in S.compute_plan(saved, live).values())  # a restore starts nothing


def test_status_shows_unhealthy_units_and_the_last_apply(tmp_path, capsys):
    write_setups(tmp_path)
    live = business_live()
    live["alpha"]["units"]["llm.service"]["health"] = {"ok": False, "detail": "HTTP 503"}
    live["alpha"]["last_apply"] = {"setup": "business", "result": "failed", "attempts": 3,
                                   "finished_at": "2026-10-31T11:20:00+00:00", "where": "on the host",
                                   "problems": ["alpha: llm.service unhealthy (HTTP 503)"]}
    live["alpha"]["deadman_units"] = {"distill-deadman-alpha.service": "failed"}
    S.write_private(S.last_apply_path(), json.dumps({"setup": "flywheel", "result": "ok"}))
    runner = StaticRunner(live)
    runner.local = False
    assert S.cmd_status(type("A", (), {"setups_dir": tmp_path, "only_host": None})(), HOSTS, runner) == 0
    out = capsys.readouterr().out
    assert "UNHEALTHY (HTTP 503)" in out
    assert "last apply run on alpha: business FAILED" in out and "HTTP 503" in out
    assert "dead-man unit: distill-deadman-alpha.service failed" in out
    assert "last apply run from this machine: flywheel OK" in out
    assert "setup business: MATCH, but UNHEALTHY: alpha llm.service (HTTP 503)" in out
    assert "active setup: business (UNHEALTHY)" in out


# ── dead-man ────────────────────────────────────────────────────────────────

WHEN = datetime(2026, 10, 31, 11, 15, tzinfo=timezone.utc)


def test_deadman_commands():
    steps = S.deadman_commands("business", "alpha", WHEN)
    run = [c for k, c in steps if k == "ssh" and "systemd-run" in c][0]
    assert "--unit=distill-deadman-alpha" in run
    assert "--on-calendar='2026-10-31 11:15:00 UTC'" in run
    assert "--user" in run
    assert shlex.split(run)[shlex.split(run).index("--description=distill dead-man: apply business") + 1] \
        == S.DEADMAN_PYTHON
    kinds = [k for k, _ in steps]
    # old failed units are reset before scheduling; the host is test-run before arming
    assert "reset-failed" in steps[0][1] and kinds.index("check") < kinds.index("ssh", 2)
    assert [s for k, s in steps if k == "scp"] == [("script", "setups.py"), ("setup", "business.yaml"),
                                                   ("hosts", "hosts.yaml")]
    assert "is-active --quiet distill-deadman-alpha.timer" in steps[-1][1]


def _main_argv(command, home):
    """What the remote shell hands the interpreter: tokens after the script, $HOME expanded."""
    toks = shlex.split(command)
    i = next(i for i, t in enumerate(toks) if t.endswith("/setups.py"))
    return toks[i - 1], [t.replace("$HOME", str(home)) for t in toks[i + 1:]]


def test_deadman_timer_command_parses_through_main(tmp_path, monkeypatch):
    steps = S.deadman_commands("business", "alpha", WHEN, python="/opt/py/bin/python3")
    run = [c for k, c in steps if k == "ssh" and "systemd-run" in c][0]
    python, argv = _main_argv(run, tmp_path)
    assert python == "/opt/py/bin/python3"
    rdir = tmp_path / S.REMOTE_DIR
    rdir.mkdir(parents=True)
    (rdir / "hosts.yaml").write_text("hosts: {alpha: {ssh: a}}\nnever_stop: [site.service]\n")
    seen = {}
    monkeypatch.setattr(S, "cmd_apply", lambda args, hosts, runner: seen.update(
        args=args, hosts=hosts, runner=runner) or 0)
    assert S.main(argv) == 0
    a = seen["args"]
    assert a.cmd == "apply" and a.name == "business" and a.execute is True
    assert a.local is True and a.only_host == "alpha"
    assert a.hosts_file == rdir / "hosts.yaml" and a.setups_dir == rdir
    assert "site.service" in seen["hosts"].guard("alpha") and seen["runner"].local


def test_deadman_check_command_parses_through_main(tmp_path, monkeypatch):
    steps = S.deadman_commands("business", "alpha", WHEN)
    check = [c for k, c in steps if k == "check"][0]
    python, argv = _main_argv(check, tmp_path)
    assert python == S.DEADMAN_PYTHON
    rdir = tmp_path / S.REMOTE_DIR
    rdir.mkdir(parents=True)
    (rdir / "hosts.yaml").write_text("hosts: {alpha: {ssh: a}}\n")
    seen = {}
    monkeypatch.setattr(S, "cmd_status", lambda args, hosts, runner: seen.update(args=args) or 0)
    assert S.main(argv) == 0
    assert seen["args"].cmd == "status" and seen["args"].local and seen["args"].only_host == "alpha"


def test_local_execute_without_its_hosts_file_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "cmd_apply", lambda *a: 0)
    with pytest.raises(SystemExit):
        S.main(["--setups-dir", str(tmp_path), "--local", "--only-host", "alpha", "apply", "x", "--execute"])
    with pytest.raises(SystemExit):
        S.main(["--hosts-file", str(tmp_path / "gone.yaml"), "--local", "--only-host", "alpha", "status"])


def test_main_dry_run_is_default(tmp_path, monkeypatch):
    hosts = tmp_path / "hosts.yaml"
    hosts.write_text("hosts: {alpha: {ssh: a}}\n")
    seen = {}
    monkeypatch.setattr(S, "cmd_apply", lambda args, hosts, runner: seen.setdefault("a", args) and 0)
    S.main(["--hosts-file", str(hosts), "--local", "--only-host", "alpha", "apply", "x"])
    assert seen["a"].execute is False


def _deadman_args(tmp_path, hosts_file, execute=True, **kw):
    base = {"setups_dir": tmp_path, "hosts_file": hosts_file, "name": "business", "only_host": None,
            "at": (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat(), "cancel": False,
            "execute": execute}
    base.update(kw)
    return type("A", (), base)()


def _deadman_fixture(tmp_path, beta_notify=True):
    write_setups(tmp_path)
    hosts_file = tmp_path / "gpu-hosts.yaml"  # not named hosts.yaml
    beta = "{ssh: beta-host, notify_cmd: /opt/notify-b}" if beta_notify else "{ssh: beta-host}"
    hosts_file.write_text("hosts:\n  alpha: {ssh: user@alpha-host, python: /opt/py/bin/python3,"
                          f" notify_cmd: /opt/notify}}\n  beta: {beta}\n"
                          "never_stop: [site.service]\nbusiness_units: {beta: [ollama.service]}\n")
    return hosts_file, S.load_hosts(hosts_file)


def test_deadman_refuses_to_arm_a_host_that_cannot_notify(tmp_path, monkeypatch, capsys):
    hosts_file, hosts = _deadman_fixture(tmp_path, beta_notify=False)
    calls = []
    monkeypatch.setattr(S, "_call", lambda argv, timeout=120: calls.append(argv)
                        or subprocess.CompletedProcess(argv, 0, "", ""))
    with pytest.raises(SystemExit, match="beta.*notify nobody.*--allow-no-notify"):
        S.cmd_deadman(_deadman_args(tmp_path, hosts_file), hosts, S.Runner(hosts))
    assert calls == []  # refused before any host was touched
    # alpha alone has a notifier, so arming only alpha goes ahead
    assert S.cmd_deadman(_deadman_args(tmp_path, hosts_file, only_host="alpha"), hosts, S.Runner(hosts)) == 0
    # the dry run says so; --allow-no-notify arms with the warning
    calls.clear()
    assert S.cmd_deadman(_deadman_args(tmp_path, hosts_file, execute=False), hosts, S.Runner(hosts)) == 0
    assert calls == [] and "WARNING: no notify_cmd in hosts.yaml for beta" in capsys.readouterr().out
    args = _deadman_args(tmp_path, hosts_file, allow_no_notify=True)
    assert S.cmd_deadman(args, hosts, S.Runner(hosts)) == 0
    assert any("systemd-run" in c[-1] for c in calls if c[0] == "ssh" and c[-2] == "beta-host")
    assert "WARNING: no notify_cmd" in capsys.readouterr().out


def test_main_parses_allow_no_notify(tmp_path, monkeypatch):
    hosts_file = tmp_path / "hosts.yaml"
    hosts_file.write_text("hosts: {alpha: {ssh: a}}\n")
    seen = {}
    monkeypatch.setattr(S, "cmd_deadman", lambda args, hosts, runner: seen.setdefault("a", args) and 0)
    S.main(["--hosts-file", str(hosts_file), "deadman", "business", "--at", "2030-01-01T07:15",
            "--allow-no-notify"])
    assert seen["a"].allow_no_notify is True and seen["a"].execute is False


def test_deadman_arms_with_one_host_copies_and_the_host_python(tmp_path, monkeypatch):
    hosts_file, hosts = _deadman_fixture(tmp_path)
    calls, copied = [], {}

    def fake_call(argv, timeout=120):
        calls.append(argv)
        if argv[0] == "scp":
            copied[(argv[-1].split(":")[0], argv[-1].rsplit("/", 1)[1])] = Path(argv[-2]).read_text()
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(S, "_call", fake_call)
    assert S.cmd_deadman(_deadman_args(tmp_path, hosts_file), hosts, S.Runner(hosts)) == 0
    alpha_hosts = yaml.safe_load(copied[("user@alpha-host", "hosts.yaml")])
    assert set(alpha_hosts["hosts"]) == {"alpha"} and alpha_hosts["never_stop"] == ["site.service"]
    assert "business_units" not in alpha_hosts
    beta_hosts = yaml.safe_load(copied[("beta-host", "hosts.yaml")])
    assert beta_hosts["business_units"] == {"beta": ["ollama.service"]}
    assert set(yaml.safe_load(copied[("user@alpha-host", "business.yaml")])["hosts"]) == {"alpha"}
    ssh = [c[-1] for c in calls if c[0] == "ssh" and c[-2] == "user@alpha-host"]
    assert ssh[0].startswith("systemctl --user reset-failed distill-deadman-alpha")
    run = [c for c in ssh if "systemd-run" in c][0]
    assert "/opt/py/bin/python3" in run and S.DEADMAN_PYTHON not in run
    beta_run = [c[-1] for c in calls if c[0] == "ssh" and c[-2] == "beta-host" and "systemd-run" in c[-1]][0]
    assert S.DEADMAN_PYTHON in beta_run


def test_deadman_refuses_to_arm_when_the_host_cannot_run_the_script(tmp_path, monkeypatch, capsys):
    hosts_file, hosts = _deadman_fixture(tmp_path)
    calls = []

    def fake_call(argv, timeout=120):
        calls.append(argv)
        if argv[0] == "ssh" and argv[-1].endswith(" status"):
            return subprocess.CompletedProcess(argv, 1, "", "ModuleNotFoundError: No module named 'yaml'")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(S, "_call", fake_call)
    assert S.cmd_deadman(_deadman_args(tmp_path, hosts_file, only_host="alpha"), hosts, S.Runner(hosts)) == 1
    assert not any("systemd-run" in c[-1] for c in calls)
    assert "not arming" in capsys.readouterr().out


def test_deadman_cancel_reports_failure(tmp_path, monkeypatch, capsys):
    hosts_file, hosts = _deadman_fixture(tmp_path)
    cmd = S.cancel_command("alpha")
    assert not cmd.rstrip().endswith("true") and "show -p ActiveState" in cmd

    def fake_call(argv, timeout=120):
        if "alpha-host" in " ".join(argv):
            return subprocess.CompletedProcess(argv, 1, "timer distill-deadman-alpha.timer is still active\n", "")
        return subprocess.CompletedProcess(argv, 0, "no dead-man timer\n", "")

    monkeypatch.setattr(S, "_call", fake_call)
    args = _deadman_args(tmp_path, hosts_file, cancel=True, name=None, at=None)
    assert S.cmd_deadman(args, hosts, S.Runner(hosts)) == 1
    out = capsys.readouterr().out
    assert "alpha: cancel FAILED" in out and "beta: cancel ok" in out


FAKE_SYSTEMCTL = """#!/bin/sh
[ "$FAKE_BUS" = down ] && { echo "Failed to connect to bus" >&2; exit 1; }
case "$*" in
  *"show -p ActiveState --value "*.timer) echo "$FAKE_TIMER";;
  *"show -p ActiveState --value "*.service) echo "$FAKE_SERVICE";;
esac
exit 0
"""


@pytest.mark.parametrize("timer, service, bus, rc, says", [
    ("inactive", "inactive", "up", 0, "no dead-man timer"),
    ("active", "inactive", "up", 1, "still active"),
    ("inactive", "activating", "up", 1, "running; not stopped"),
    ("inactive", "inactive", "down", 1, "cannot query"),
])
def test_cancel_command_proves_the_timer_is_gone(tmp_path, timer, service, bus, rc, says):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "systemctl"
    fake.write_text(FAKE_SYSTEMCTL)
    fake.chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "FAKE_TIMER": timer,
           "FAKE_SERVICE": service, "FAKE_BUS": bus}
    cp = subprocess.run(["sh", "-c", S.cancel_command("alpha")], capture_output=True, text=True,
                        env=env, timeout=10)
    assert cp.returncode == rc and says in cp.stdout


def test_parse_at_naive_is_eastern():
    dt = S.parse_at("2026-10-31T07:15")
    assert S.on_calendar(dt) == "2026-10-31 11:15:00 UTC"
    assert S.on_calendar(S.parse_at("2026-10-31T11:15:00Z")) == "2026-10-31 11:15:00 UTC"
