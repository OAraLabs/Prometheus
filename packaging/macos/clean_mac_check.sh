#!/bin/bash
# clean_mac_check.sh: checks to run on a Mac that has never seen Prometheus, step by step.
#
# Uses only what a stock Mac ships (bash 3.2, codesign, spctl, curl, lsof, launchctl): no Homebrew, no
# Python, no Xcode tools. `xcrun` and `stapler` are NOT used: on a Mac without developer tools they pop up
# an "install the command line tools" dialog. spctl is enough to prove notarization: it says
# "source=Notarized Developer ID" only when Apple's ticket is valid.
#
#   ./clean_mac_check.sh before        the app is in ~/Applications, not opened yet
#   ./clean_mac_check.sh running       after opening it and clicking "Set up"
#   ./clean_mac_check.sh pair          optional: pair like Beacon would (uses up the one-time secret)
#   ./clean_mac_check.sh killtest      stop the daemon and time how long until it is back
#   ./clean_mac_check.sh uninstall     remove everything the app put on the Mac
#
# Every line is also appended to ~/Desktop/prometheus-test-report.txt; send that file back.
# Set PROMETHEUS_APP to test an app somewhere else, PROMETHEUS_REPORT to write the report elsewhere.

APP="${PROMETHEUS_APP:-$HOME/Applications/Prometheus.app}"
LAUNCHER="$APP/Contents/MacOS/Prometheus"
LABEL="com.oaralabs.prometheus.daemon"
REPORT="${PROMETHEUS_REPORT:-$HOME/Desktop/prometheus-test-report.txt}"
PAIRDIR="$HOME/Library/Application Support/Prometheus/pairing"
BASE="http://127.0.0.1:8005"
FAILS=0

say()  { printf '%s\n' "$*" | tee -a "$REPORT"; }
pass() { say "PASS  $*"; }
fail() { say "FAIL  $*"; FAILS=$((FAILS + 1)); }
info() { say "info  $*"; }
head_() { say ""; say "=== $* ($(date '+%Y-%m-%d %H:%M:%S'))"; }

status_json() { curl -s -m 3 "$BASE/api/setup/status" 2>/dev/null; }
daemon_pid()  { launchctl print "gui/$(id -u)/$LABEL" 2>/dev/null | sed -n 's/^[[:space:]]*pid = \([0-9][0-9]*\)$/\1/p' | head -1; }

stage_before() {
  head_ "BEFORE: this Mac and the app on disk"
  info "macOS $(sw_vers -productVersion), $(uname -m), $(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
  for tool in brew python3 pip3 git; do
    if command -v "$tool" >/dev/null 2>&1; then info "$tool present at $(command -v "$tool") (expected absent for a clean test)"; else info "$tool absent"; fi
  done
  [ "$(uname -m)" = "arm64" ] && pass "Apple silicon" || fail "this app is for Apple silicon (arm64); this Mac is $(uname -m)"
  [ -d "$APP" ] && pass "app found at $APP" || { fail "no app at $APP: move Prometheus.app into ~/Applications first"; return; }
  if xattr -p com.apple.quarantine "$APP" >/dev/null 2>&1; then
    info "downloaded-file flag (quarantine) IS set: this is the normal path for a file from a browser or AirDrop"
  else
    info "no quarantine flag: Gatekeeper will not check it at first open (how Beacon's own download behaves)"
  fi
  if codesign --verify --deep --strict "$APP" 2>/dev/null; then pass "signature is intact"; else fail "codesign --verify failed"; fi
  if codesign --verify --deep --strict '-R=anchor apple generic and certificate leaf[subject.OU] = "53JM8W47RL"' "$APP" 2>/dev/null; then
    pass "signed by team 53JM8W47RL (code requirement)"; else fail "the signature does not satisfy the 53JM8W47RL requirement"; fi
  codesign -dvv "$APP" 2>&1 | grep -E '^(Identifier|TeamIdentifier|Authority=Developer ID Application|Timestamp)' | sed 's/^/        /' | tee -a "$REPORT"
  verdict="$(spctl -a -t exec -vv "$APP" 2>&1)"; say "        $(printf '%s' "$verdict" | tr '\n' ' ')"
  case "$verdict" in *"source=Notarized Developer ID"*) pass "Gatekeeper accepts it as Notarized Developer ID" ;; *) fail "Gatekeeper does not call it Notarized Developer ID" ;; esac
  if command -v syspolicy_check >/dev/null 2>&1; then
    if syspolicy_check distribution "$APP" >/dev/null 2>&1; then pass "syspolicy_check distribution"; else fail "syspolicy_check distribution"; fi
  else info "syspolicy_check not on this macOS (needs macOS 14 or later)"; fi
}

stage_running() {
  head_ "RUNNING: after opening the app and clicking Set up"
  body="$(status_json)"
  case "$body" in *'"setup_mode"'*) pass "the daemon answers on 127.0.0.1:8005: $body" ;; *) fail "nothing (or not Prometheus) answers on 127.0.0.1:8005" ;; esac
  listen="$(lsof -nP -iTCP:8005 -sTCP:LISTEN 2>/dev/null | awk 'NR>1{print $9}' | sort -u | tr '\n' ' ')"
  info "listening on: ${listen:-nothing}"
  case "$listen" in "") fail "nothing is listening on 8005" ;; *"*:"*|*"0.0.0.0"*) fail "listening on every network, not just this Mac" ;; *"127.0.0.1:8005"*) pass "listening on this Mac only" ;; esac
  lan="$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null)"
  if [ -n "$lan" ]; then
    if curl -s -m 3 -o /dev/null "http://$lan:8005/api/setup/status"; then fail "the Wi-Fi address $lan:8005 ANSWERS: it must refuse"; else pass "the Wi-Fi address $lan:8005 refuses the connection"; fi
  else info "no Wi-Fi/Ethernet address found to test against"; fi
  pid="$(daemon_pid)"; [ -n "$pid" ] && pass "launchd has the agent running (pid $pid)" || fail "launchd does not show the agent running"
  [ -n "$pid" ] && info "process: $(ps -o command= -p "$pid" | cut -c1-140)"
  if [ -f "$PAIRDIR/pair.secret" ]; then
    mode="$(stat -f '%Sp' "$PAIRDIR/pair.secret")"; dmode="$(stat -f '%Sp' "$PAIRDIR")"
    [ "$mode" = "-rw-------" ] && [ "$dmode" = "drwx------" ] && pass "pairing secret is private ($dmode / $mode)" || fail "pairing secret permissions are $dmode / $mode"
  else info "no pairing secret on disk (already used, or the app is not in setup mode)"; fi
  [ -e "$HOME/.prometheus" ] && info "~/.prometheus exists" || info "~/.prometheus does not exist yet (expected in setup mode)"
  say "MANUAL: open System Settings > General > Login Items & Extensions and note what the entry is called and whether it has the O. icon. Screenshot it."
}

stage_pair() {
  head_ "PAIR: what Beacon will do, by hand"
  [ -f "$PAIRDIR/pair.secret" ] || { fail "no pairing secret to use"; return; }
  secret="$(tr -d '\n' < "$PAIRDIR/pair.secret")"
  code="$(curl -s -m 5 -o /tmp/pair-reply.json -w '%{http_code}' -H 'Content-Type: application/json' -d "{\"code\":\"$secret\"}" "$BASE/api/setup/pair")"
  [ "$code" = "200" ] && pass "pairing answered 200 (keys: $(sed 's/"token":"[^"]*"/"token":"..."/' /tmp/pair-reply.json))" || fail "pairing answered $code"
  rm -f /tmp/pair-reply.json
  [ -f "$PAIRDIR/pair.secret" ] && fail "the secret is still on disk after use" || pass "the secret was deleted when used"
  again="$(curl -s -m 5 -o /dev/null -w '%{http_code}' -H 'Content-Type: application/json' -d "{\"code\":\"$secret\"}" "$BASE/api/setup/pair")"
  [ "$again" = "401" ] && pass "a second use is refused (401)" || fail "a second use answered $again (expected 401)"
}

stage_killtest() {
  head_ "KILLTEST: does it come back by itself?"
  old="$(daemon_pid)"; [ -n "$old" ] || { fail "no running agent to stop"; return; }
  info "stopping pid $old"; kill -TERM "$old"
  start="$(date +%s)"; new=""
  for _ in $(seq 1 40); do sleep 1; new="$(daemon_pid)"; [ -n "$new" ] && [ "$new" != "$old" ] && case "$(status_json)" in *'"setup_mode"'*|*unauthorized*) break ;; esac; done
  secs=$(( $(date +%s) - start ))
  if [ -n "$new" ] && [ "$new" != "$old" ]; then pass "back as pid $new after about ${secs}s (launchd waits 10 s between starts)"; else fail "not back after ${secs}s"; fi
}

stage_uninstall() {
  head_ "UNINSTALL"
  [ -x "$LAUNCHER" ] || { fail "no launcher at $LAUNCHER"; return; }
  out="$("$LAUNCHER" --uninstall 2>&1)"; say "        $out"
  case "$out" in *'"state":"uninstalled"'*) pass "uninstall reports done" ;; *) fail "uninstall did not report done" ;; esac
  sleep 2
  [ -e "$APP" ] && fail "the app is still at $APP" || pass "the app is gone from $(dirname "$APP")"
  ls "$HOME/.Trash" 2>/dev/null | grep -q '^Prometheus' && pass "the app is in the Trash" || info "could not see it in the Trash (Terminal may lack permission to look)"
  launchctl print "gui/$(id -u)/$LABEL" >/dev/null 2>&1 && fail "launchd still has the agent" || pass "launchd no longer has the agent"
  curl -s -m 3 -o /dev/null "$BASE/api/setup/status" && fail "something still answers on 8005" || pass "nothing answers on 8005"
  [ -d "$PAIRDIR" ] && fail "the pairing folder is still there" || pass "the pairing folder is gone"
  say "MANUAL: open Login Items & Extensions again: the Prometheus entry should be gone."
}

case "$1" in
  before) stage_before ;;
  running) stage_running ;;
  pair) stage_pair ;;
  killtest) stage_killtest ;;
  uninstall) stage_uninstall ;;
  *) echo "usage: $0 before | running | pair | killtest | uninstall"; exit 2 ;;
esac
say ""
if [ "$FAILS" -eq 0 ]; then say "RESULT: no failures in this stage. Report file: $REPORT"; else say "RESULT: $FAILS failure(s) in this stage. Report file: $REPORT"; fi
exit $(( FAILS > 0 ))
