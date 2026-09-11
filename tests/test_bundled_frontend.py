"""The daemon's own bundled UI — the only shipped frontend, previously unread.

`src/prometheus/web/static/` is served as a catch-all mount on the
authenticated API and is the one UI this daemon ships. Nobody had ever
opened it. It held:

* **an XSS sink.** The skills list built an `onclick="openSkill(...)"`
  attribute by interpolating a skill's NAME, guarded only by
  `JSON.stringify(name).replace(/"/g, '&quot;')`. That guard is defeated by
  a name containing the literal text `&quot;`, because the HTML parser
  decodes the attribute before the JS parser sees it:

      name    : x&quot;)+alert(document.domain)+(&quot;
      source  : onclick="openSkill(&quot;x&quot;)+alert(...)+(&quot;&quot;)"
      decoded : openSkill("x")+alert(...)+("")

  and `class="state ${stateClass}"` interpolated the state with no escaping
  at all. Neither was reachable in practice — `skill_creator._slugify`
  confines a stem to `[a-z0-9-]` and `SkillStateStore` validates state to
  an enum — which is the point: the page's only defence was two rules in
  modules it does not own and cannot see.

* **no Content-Security-Policy of any kind**, while holding the API bearer
  token in `localStorage`. Any script that ever ran in that origin could
  read the token and drive every route it opens, `bash` included.

⚠ THE TWO HALVES ARE LOAD-BEARING TOGETHER. `script-src 'self'` makes an
inline `on*=""` handler DEAD — it does not fire, and nothing reports it. So
a future inline handler is not just a re-opened sink, it is a silently
broken button. `test_no_inline_event_handlers` is what keeps them in step.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

REPO = Path(__file__).resolve().parent.parent
STATIC = REPO / "src" / "prometheus" / "web" / "static"
HTML = STATIC / "index.html"
JS = STATIC / "app.js"
CSS = STATIC / "app.css"


def _code_only(text: str) -> str:
    """Source with comments removed, for the structural scans below.

    ⚠ NEEDED BECAUSE THE COMMENTS QUOTE THE OLD CODE. app.js documents the
    defeated guard verbatim so the next reader recognises the shape, and a
    naive scan matches its own explanation — the test then fails on the fix
    that removed the bug.

    Only WHOLE-LINE comments are stripped, never a `//` mid-line: the page
    builds URLs like `${proto}//${host}`, and stripping from the first `//`
    would delete real code from the scan and hide an offender behind a
    trailing comment.
    """
    out, in_block = [], False
    for line in text.splitlines():
        stripped = line.strip()
        if in_block:
            if "*/" in stripped:
                in_block = False
            continue
        if stripped.startswith("/*"):
            in_block = "*/" not in stripped
            continue
        if stripped.startswith("//") or stripped.startswith("*"):
            continue
        out.append(line)
    return "\n".join(out)


def _app(ws_port: int = 8010):
    from prometheus.web.server import create_app

    return create_app(
        {"web": {"enabled": True, "ws_port": ws_port}}, static_dir=str(STATIC)
    )


# ── the injection class, removed rather than escaped ─────────────────

def test_no_inline_event_handlers():
    """Not one `on*=""` in the shipped markup.

    Both a security property and a liveness one: under `script-src 'self'`
    an inline handler never runs, so re-adding one produces a button that
    silently does nothing.
    """
    offenders = []
    for f in (HTML, JS):
        for n, line in enumerate(_code_only(f.read_text(encoding="utf-8")).splitlines(), 1):
            if re.search(r"""\son[a-z]+\s*=\s*["']""", line):
                offenders.append(f"{f.name}: {line.strip()[:120]}")
    assert not offenders, (
        "inline event handlers are back. They are JavaScript written into an "
        "HTML attribute, so every value near them must survive HTML decoding "
        "AND JS parsing — the bug this file documents — and `script-src "
        "'self'` means they will not fire at all:\n  " + "\n  ".join(offenders)
    )


def test_the_defeated_guard_is_gone():
    """The exact expression that shipped the hole.

    Named rather than described, because a reader who sees this pattern
    elsewhere should recognise it: JSON.stringify protects against a literal
    quote and does nothing about the TEXT `&quot;`.
    """
    src = _code_only(JS.read_text(encoding="utf-8"))
    assert "JSON.stringify(s.name).replace" not in src
    assert not re.search(r"""JSON\.stringify\([^)]*\)\.replace\(/"/g""", src), (
        "the JSON.stringify(...).replace(/\"/g,'&quot;') attribute guard is "
        "back. It is defeated by a value containing the literal text &quot;"
    )


def test_the_double_encoding_payload_would_still_defeat_that_guard():
    """Proves the guard is unsound, so nobody reinstates it as 'good enough'.

    Not a test of shipped code — a test of the reasoning that removed it.
    Runs the historical expression and shows the browser gets executable JS.
    """
    from html.parser import HTMLParser

    name = r"x&quot;)+alert(document.domain)+(&quot;"
    attr = "openSkill(%s)" % json.dumps(name).replace('"', "&quot;")
    seen: list[str] = []

    class P(HTMLParser):
        def handle_starttag(self, tag, attrs):
            seen.extend(v for k, v in attrs if k == "onclick")

    P().feed(f'<div onclick="{attr}">x</div>')
    assert seen == ['openSkill("x")+alert(document.domain)+("")'], seen
    assert not seen[0].startswith('openSkill("x&quot;'), (
        "if this ever holds the payload inside the string literal, the old "
        "guard was sound after all and this file's premise needs revisiting"
    )


def test_the_skills_renderer_never_builds_markup_from_data():
    """It uses textContent and dataset, neither of which parses markup."""
    src = _code_only(JS.read_text(encoding="utf-8"))
    block = src[src.index("skillsEl.replaceChildren"):]
    block = block[: block.index("\n  }")]
    assert "innerHTML" not in block
    assert "textContent" in block and "dataset" in block
    assert "${" not in block, (
        "the skills renderer is interpolating again — the whole point of the "
        "DOM-API rewrite is that there is no encoding left to get right"
    )


def test_the_state_class_comes_from_an_allowlist():
    """`class=\"state ${stateClass}\"` took the value verbatim. Its safety was
    SkillStateStore's enum validation, two modules away."""
    src = JS.read_text(encoding="utf-8")
    assert 'known.includes(' in src or '["active", "stale", "archived"]' in src
    from prometheus.learning.skill_state import _VALID_STATES

    for state in _VALID_STATES:
        assert f'"{state}"' in src, (
            f"skill state {state!r} exists in skill_state.py but the page's "
            f"allowlist does not carry it — it would render as 'active'"
        )


# ── the CSP, and what makes it expressible ──────────────────────────

def test_no_inline_script_or_style_blocks():
    """`script-src 'self'` and `style-src 'self'` are only possible because
    these moved out. Re-inlining either silently blanks the page."""
    html = HTML.read_text(encoding="utf-8")
    assert not re.search(r"<script(?![^>]*\ssrc=)[^>]*>", html), "inline <script> is back"
    assert "<style" not in html, "inline <style> is back"
    assert not re.search(r"""\sstyle\s*=\s*["']""", html), "inline style= attribute is back"


def test_the_page_and_its_assets_all_carry_the_policy():
    c = TestClient(_app())
    for path in ("/index.html", "/app.js", "/app.css"):
        r = c.get(path)
        assert r.status_code == 200, path
        assert "content-security-policy" in r.headers, path
        assert r.headers["x-content-type-options"] == "nosniff"
        assert r.headers["x-frame-options"] == "DENY"
        assert r.headers["referrer-policy"] == "no-referrer"


@pytest.mark.parametrize("directive,value", [
    ("default-src", "'self'"),
    ("script-src", "'self'"),
    ("style-src", "'self'"),
    ("object-src", "'none'"),
    ("frame-ancestors", "'none'"),
    ("form-action", "'none'"),
    ("base-uri", "'none'"),
])
def test_policy_directives(directive, value):
    csp = TestClient(_app()).get("/index.html").headers["content-security-policy"]
    got = {d.split(" ", 1)[0]: d.split(" ", 1)[1] for d in csp.split("; ")}
    assert got.get(directive) == value, f"{directive} is {got.get(directive)!r}"


def test_connect_src_names_the_configured_ws_port_not_a_wildcard():
    """`ws://*:PORT` would also let an injected script stream the page out to
    attacker.example on that port. The host is this request's own."""
    csp = TestClient(_app(ws_port=9999)).get(
        "/index.html", headers={"Host": "box.example:8005"}
    ).headers["content-security-policy"]
    connect = [d for d in csp.split("; ") if d.startswith("connect-src ")][0]
    assert connect == "connect-src 'self' ws://box.example:9999 wss://box.example:9999", connect
    assert "*" not in connect


def test_a_request_with_no_host_gets_no_websocket_permission():
    from prometheus.web.server import _static_security_headers

    headers = dict(_static_security_headers({"web": {"ws_port": 8010}}, scope={"headers": []}))
    csp = headers[b"content-security-policy"].decode()
    assert "connect-src 'self' 'none'" in csp, csp


def test_the_policy_does_not_carry_unsafe_inline_or_eval():
    csp = TestClient(_app()).get("/index.html").headers["content-security-policy"]
    for bad in ("'unsafe-inline'", "'unsafe-eval'", "data: 'self' script"):
        assert bad not in csp, f"{bad} in policy: {csp}"


# ── the page still has to work ──────────────────────────────────────

@pytest.mark.skipif(shutil.which("node") is None, reason="node not on PATH")
def test_app_js_parses():
    r = subprocess.run(["node", "--check", str(JS)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_every_element_id_the_script_needs_exists_in_the_page():
    """Catches the wiring half of moving handlers out of the markup: a
    delegation target that names an id the page does not have fails
    silently at runtime, with the button simply doing nothing."""
    html = HTML.read_text(encoding="utf-8")
    ids = set(re.findall(r'id="([^"]+)"', html))
    wanted = set(re.findall(r"""getElementById\(["']([^"']+)["']\)""",
                            JS.read_text(encoding="utf-8")))
    missing = sorted(wanted - ids)
    assert not missing, f"app.js looks up ids the page does not define: {missing}"


def test_every_delegated_action_has_a_handler():
    """The other half of the id check. A `data-action` with no `case` is a
    dead control — the click is caught by the delegator and then dropped,
    which looks identical to a working button that does nothing."""
    js = _code_only(JS.read_text(encoding="utf-8"))
    html = HTML.read_text(encoding="utf-8")
    produced = set(re.findall(r'dataset\.action = "([a-z-]+)"', js))
    produced |= set(re.findall(r'data-action="([a-z-]+)"', js + html))
    handled = set(re.findall(r'case "([a-z-]+)":', js))
    assert produced, "no delegated actions found — did the delegation revert?"
    assert produced <= handled, (
        f"actions with no handler: {sorted(produced - handled)}"
    )
    assert handled <= produced, (
        f"handlers for actions nothing emits: {sorted(handled - produced)}"
    )


def test_the_deferred_buttons_map_onto_the_values_the_api_takes():
    """`data-deferred` replaced `onclick=\"setDeferred(true)\"`. The literal
    true/false/'auto' had to survive that move."""
    html = HTML.read_text(encoding="utf-8")
    assert set(re.findall(r'data-deferred="([a-z]+)"', html)) == {"auto", "on", "off"}
    js = _code_only(JS.read_text(encoding="utf-8"))
    assert 'v === "auto" ? "auto" : v === "on"' in js, (
        "the auto/on/off mapping changed — 'off' must reach setDeferred as "
        "false, not as the string 'off'"
    )


def test_the_page_references_only_assets_that_are_served():
    c = TestClient(_app())
    html = HTML.read_text(encoding="utf-8")
    refs = re.findall(r'(?:src|href)="(/[^"]+)"', html)
    assert refs, "the page references no local assets — did the split revert?"
    for ref in refs:
        assert c.get(ref).status_code == 200, f"{ref} is referenced but not served"


def test_the_css_pulls_nothing_from_another_origin():
    """`default-src 'self'` would block it, and a blocked font is a silent
    layout change rather than an error."""
    css = CSS.read_text(encoding="utf-8")
    assert "@import" not in css
    assert not re.search(r"url\(\s*['\"]?https?:", css)


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv not on PATH")
def test_the_split_assets_ship_in_the_wheel(tmp_path: Path):
    """The page used to be ONE self-contained file. It now has dependencies.

    A wheel carrying index.html but not app.js is a blank page for anyone
    who installed rather than cloned — the identity-template defect (#417)
    in a new place, and the reason that one went unnoticed for so long was
    that nobody asserted against the artefact the consumer receives.
    """
    dist = tmp_path / "dist"
    build = subprocess.run(
        ["uv", "build", "--wheel", "--offline", "--out-dir", str(dist)],
        cwd=REPO, capture_output=True, text=True,
    )
    if build.returncode != 0:
        pytest.skip(f"wheel build unavailable here: {build.stderr.strip()[:200]}")
    wheels = list(dist.glob("*.whl"))
    assert len(wheels) == 1, wheels

    import zipfile

    names = set(zipfile.ZipFile(wheels[0]).namelist())
    for asset in ("index.html", "app.js", "app.css"):
        want = f"prometheus/web/static/{asset}"
        assert want in names, (
            f"{want} is not in the wheel. The page references it by URL, so "
            f"an installed daemon serves a document whose script or styles "
            f"404 — silently, because a missing <script src> raises nothing."
        )
