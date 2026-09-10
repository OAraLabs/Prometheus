  // -------------- config / API endpoints -----------------
  const API_BASE = `${window.location.protocol}//${window.location.host}`;
  // The WS bridge runs on a separate port (default 8010) — let the API
  // tell us where to connect by reading the host and switching port. If
  // the user has reverse-proxied everything to one port, /ws would be the
  // upgrade path; we'll try the conventional port first, then fall back.
  const WS_URL = (() => {
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const host = window.location.hostname || "localhost";
    return `${proto}//${host}:8010`;
  })();
  const ICONS = {
    tool_call_start: "🔧", tool_call_end: "🔧", tool_call_complete: "🔧",
    model_call_start: "🧠", model_call_complete: "🧠",
    skill_created: "🎓", skill_refined: "📚",
    memory_updated: "📝", curator_report: "📋",
    steer_received: "📍", prompt_queued: "📥",
    file_mutation_summary: "📁",
    dream_start: "💤", dream_phase: "💤", dream_complete: "✨",
    chat_message: "💬", chat_delta: "💬",
    sentinel_signal: "📡",
    connected: "🔌", subscribed: "🔌",
    agent_state: "⚙️",
  };

  // -------------- state ---------------------------------
  const feedEl = document.getElementById("feed");
  const memoryEl = document.getElementById("memory");
  const skillsEl = document.getElementById("skills");
  const feedCountEl = document.getElementById("feed-count");
  const skillsCountEl = document.getElementById("skills-count");
  const modelPill = document.getElementById("model-pill");
  const uptimePill = document.getElementById("uptime-pill");
  const wsPill = document.getElementById("ws-pill");

  let feedEvents = [];   // newest first
  const FEED_LIMIT = 200;

  // -------------- formatting helpers --------------------
  function escapeHtml(s) {
    if (s == null) return "";
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function relativeTime(timestamp) {
    if (!timestamp) return "?";
    let ms;
    if (typeof timestamp === "string") {
      const d = new Date(timestamp);
      if (isNaN(d.getTime())) return timestamp;
      ms = d.getTime();
    } else {
      // Treat as seconds since epoch if small enough, else ms.
      ms = timestamp < 1e12 ? timestamp * 1000 : timestamp;
    }
    const diff = (Date.now() - ms) / 1000;
    if (diff < 5) return "just now";
    if (diff < 60) return Math.floor(diff) + "s ago";
    if (diff < 3600) return Math.floor(diff / 60) + "m ago";
    if (diff < 86400) return Math.floor(diff / 3600) + "h ago";
    return Math.floor(diff / 86400) + "d ago";
  }

  function summarisePayload(type, payload) {
    if (!payload) return "";
    // Common high-signal fields, in priority order.
    const KEYS = [
      "skill_name", "name", "path", "summary", "tool_name", "operation",
      "target", "message", "kind", "state", "session_id",
    ];
    for (const k of KEYS) {
      if (payload[k]) {
        const v = String(payload[k]);
        return v.length > 100 ? v.slice(0, 97) + "..." : v;
      }
    }
    return "";
  }

  // -------------- feed rendering ------------------------
  function renderFeed() {
    feedCountEl.textContent = feedEvents.length;
    if (feedEvents.length === 0) {
      feedEl.innerHTML = '<div class="feed-empty">No activity yet — interact with the agent to populate.</div>';
      return;
    }
    const rows = feedEvents.map((ev, idx) => {
      const icon = ICONS[ev.type] || ICONS[ev.signal_type] || "•";
      const type = ev.type || ev.signal_type || "event";
      const summary = ev.summary || summarisePayload(type, ev.payload);
      const time = relativeTime(ev.timestamp);
      const payloadJson = JSON.stringify(ev.payload || {}, null, 2);
      return `
        <div class="feed-row" data-idx="${idx}" data-action="toggle-row">
          <div class="feed-icon">${icon}</div>
          <div>
            <div class="feed-summary"><span class="feed-type">${escapeHtml(type)}</span>${
              summary ? " — " + escapeHtml(summary) : ""
            }</div>
            <div class="feed-meta">${escapeHtml(ev.source || "")}</div>
            <div class="feed-payload">${escapeHtml(payloadJson)}</div>
          </div>
          <div class="feed-time">${time}</div>
        </div>
      `;
    });
    feedEl.innerHTML = rows.join("");
  }
  const toggleRow = function(row) { row.classList.toggle("expanded"); };

  function pushEvent(ev) {
    // Normalise event shape: hydrated rows have signal_type + payload + timestamp.
    const normalised = {
      type: ev.type || ev.signal_type,
      signal_type: ev.signal_type,
      payload: ev.payload || {},
      source: ev.source || ev.source_subsystem || "",
      timestamp: ev.timestamp,
      summary: ev.summary,
    };
    feedEvents.unshift(normalised);
    if (feedEvents.length > FEED_LIMIT) feedEvents.length = FEED_LIMIT;
    renderFeed();
  }

  // -------------- memory rendering ----------------------
  function renderMemory(data) {
    if (!data) {
      memoryEl.innerHTML = '<div class="feed-empty">Memory module unavailable.</div>';
      return;
    }
    const sections = [];
    for (const [key, label] of [["memory", "MEMORY.md"], ["user", "USER.md"]]) {
      const section = data[key];
      if (!section) {
        sections.push(`<div class="mem-section"><h3>${label}</h3><div class="meta">unavailable</div></div>`);
        continue;
      }
      const pct = section.char_limit > 0
        ? Math.min(100, (section.char_count / section.char_limit) * 100)
        : 0;
      const barClass = pct > 90 ? "error" : pct > 75 ? "warn" : "";
      const preview = (section.content || "").slice(0, 4000);
      const truncated = (section.content || "").length > 4000;
      sections.push(`
        <div class="mem-section">
          <h3>
            <span>${label}</span>
            <span class="meta">${section.char_count}/${section.char_limit} · ${section.entry_count} entries</span>
          </h3>
          <div class="bar"><div class="${barClass}" data-width="${pct.toFixed(1)}"></div></div>
          <pre>${escapeHtml(preview) || "(empty)"}${truncated ? "\n… (truncated)" : ""}</pre>
        </div>
      `);
    }
    memoryEl.innerHTML = sections.join("");
    // The width was an inline `style="width: N%"`. Applied here instead so
    // the page needs no `style-src 'unsafe-inline'` — see app.css.
    memoryEl.querySelectorAll("[data-width]").forEach(el => {
      el.style.width = `${el.dataset.width}%`;
    });
  }

  // -------------- skills rendering ----------------------
  function renderSkills(list) {
    skillsCountEl.textContent = list.length;
    if (!list || list.length === 0) {
      skillsEl.innerHTML = '<div class="feed-empty">No auto-skills yet. They appear when SkillCreator fires.</div>';
      return;
    }
    // BUILT WITH DOM APIs, NOT MARKUP. A skill's name is a FILENAME from
    // ~/.prometheus/skills/auto/, and the old renderer interpolated it into an
    // `onclick="openSkill(...)"` attribute guarded only by
    // JSON.stringify(...).replace(/"/g,'&quot;'). That guard is defeated by a
    // name containing the literal TEXT `&quot;`, because the HTML parser
    // decodes the attribute BEFORE the JS parser sees it:
    //
    //   name    : x&quot;)+alert(document.domain)+(&quot;
    //   source  : onclick="openSkill(&quot;x&quot;)+alert(...)+(&quot;&quot;)"
    //   decoded : openSkill("x")+alert(...)+("")     <- alert() runs
    //
    // Verified with a real HTML parser, not by reading. Nothing that writes
    // into auto/ today can produce that name — every writer goes through
    // skill_creator._slugify, which confines the stem to [a-z0-9-] — so this
    // was latent rather than live. That is exactly the reason to fix it here:
    // the page's only defence was a slug rule three modules away that it does
    // not own and cannot see. `state` had the same shape and was worse
    // (interpolated raw into `class="state ${...}"`, no escaping at all),
    // saved only by SkillStateStore validating it to active|stale|archived.
    //
    // textContent and dataset never parse their input as markup, so there is
    // no encoding to get right and no upstream rule to depend on.
    skillsEl.replaceChildren(...list.map(s => {
      const row = document.createElement("div");
      row.className = "skill-row";

      const name = document.createElement("div");
      name.className = "name";
      name.textContent = s.name;
      name.dataset.skill = s.name;
      name.dataset.action = "open-skill";

      const state = document.createElement("span");
      // From an allowlist, never from the value: an unexpected state renders
      // its text but cannot pick a class name.
      const known = ["active", "stale", "archived"];
      const stateValue = String(s.state || "active");
      state.className = "state " + (known.includes(stateValue.toLowerCase())
        ? stateValue.toLowerCase() : "active");
      state.textContent = stateValue;

      const pin = document.createElement("button");
      pin.type = "button";
      pin.className = s.pinned ? "pin pinned" : "pin";
      pin.textContent = s.pinned ? "📌 pinned" : "📌";
      pin.dataset.skill = s.name;
      pin.dataset.pinned = s.pinned ? "1" : "";
      pin.dataset.action = "toggle-pin";

      row.append(name, state, pin);
      return row;
    }));
  }

  // -------------- skill modal ---------------------------
  const modalOverlay = document.getElementById("modal-overlay");
  const modalTitle = document.getElementById("modal-title");
  const modalBody = document.getElementById("modal-body");

  const openSkill = async function(name) {
    modalTitle.textContent = name;
    modalBody.textContent = "Loading…";
    modalOverlay.classList.add("open");
    try {
      const resp = await fetch(`${API_BASE}/api/skills/${encodeURIComponent(name)}`);
      if (!resp.ok) {
        modalBody.textContent = `(error fetching skill: HTTP ${resp.status})`;
        return;
      }
      const data = await resp.json();
      modalBody.textContent = data.content || "(empty skill file)";
    } catch (e) {
      modalBody.textContent = `(network error: ${e.message})`;
    }
  };
  const closeModal = function() { modalOverlay.classList.remove("open"); };

  const togglePin = async function(name, currentlyPinned) {
    const method = currentlyPinned ? "DELETE" : "POST";
    try {
      const resp = await fetch(`${API_BASE}/api/skills/${encodeURIComponent(name)}/pin`, { method });
      if (resp.ok) {
        await refreshSkills();
      }
    } catch (e) {
      console.error("pin toggle failed", e);
    }
  };

  // -------------- header status -------------------------
  async function refreshStatus() {
    try {
      const resp = await fetch(`${API_BASE}/api/status`);
      const s = await resp.json();
      modelPill.textContent = `Model: ${s.model || "?"} (${s.provider || "?"})`;
      const up = Math.floor(s.uptime_seconds || 0);
      const h = Math.floor(up / 3600), m = Math.floor((up % 3600) / 60);
      uptimePill.textContent = `Uptime: ${h}h ${m}m`;
    } catch {
      modelPill.textContent = "Model: (unavailable)";
      uptimePill.textContent = "Uptime: ?";
    }
  }

  async function refreshMemory() {
    try {
      const resp = await fetch(`${API_BASE}/api/memory/current`);
      const data = await resp.json();
      renderMemory(data);
    } catch (e) {
      memoryEl.innerHTML = `<div class="feed-empty">Memory error: ${escapeHtml(e.message)}</div>`;
    }
  }

  async function refreshSkills() {
    try {
      const resp = await fetch(`${API_BASE}/api/skills/list`);
      const data = await resp.json();
      renderSkills(data);
    } catch (e) {
      skillsEl.innerHTML = `<div class="feed-empty">Skills error: ${escapeHtml(e.message)}</div>`;
    }
  }

  // ── Deferred tool loading toggle ─────────────────────────────────────
  // REST is the source of truth; the WS merely triggers re-hydration (same
  // pattern as memory/skills). Auth: attach the localStorage bearer when the
  // user has set one (same convention as the WS auth frame above).
  function authHeaders() {
    try {
      const token = localStorage.getItem("prometheus_token");
      return token ? { "Authorization": `Bearer ${token}` } : {};
    } catch { return {}; }
  }

  function renderDeferred(d) {
    const badge = document.getElementById("deferred-badge");
    const status = document.getElementById("deferred-status");
    const note = document.getElementById("deferred-note");
    // Active button = the CONFIGURED tri-state (what the user chose), while
    // the status line shows the EFFECTIVE result and its source — the spec's
    // "auto → enabled (local provider)" vs "explicitly disabled" split.
    const mode = d.configured === true ? "on" : d.configured === false ? "off" : "auto";
    document.querySelectorAll("#deferred-modes button").forEach((b) => {
      b.classList.toggle("active", b.dataset.mode === mode);
    });
    if (d.effective === null || d.effective === undefined) {
      badge.textContent = "?";
      status.textContent = d.source || "effective state unknown";
    } else {
      const eff = d.effective ? "deferred" : "full catalog";
      badge.textContent = d.advertised_count !== null ? `${d.advertised_count}/${d.total_tools}` : eff;
      status.innerHTML =
        `Effective: <span class="${d.effective ? "eff-on" : "eff-off"}">${eff}</span>` +
        ` — ${escapeHtml(d.source || "")}` +
        (d.advertised_count !== null
          ? ` · advertising ${d.advertised_count} of ${d.total_tools} tools`
          : "");
    }
    if (d.applies) note.textContent = `Changes apply at the ${d.applies} — not to a session already running.`;
  }

  async function refreshDeferred() {
    try {
      const resp = await fetch(`${API_BASE}/api/tools/deferred`, { headers: authHeaders() });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      renderDeferred(await resp.json());
    } catch (e) {
      document.getElementById("deferred-status").textContent = `error: ${e.message}`;
    }
  }

  const setDeferred = async function(value) {
    try {
      const resp = await fetch(`${API_BASE}/api/tools/deferred`, {
        method: "PUT",
        headers: { "Content-Type": "application/json", ...authHeaders() },
        body: JSON.stringify({ enabled: value }),
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      renderDeferred(await resp.json());
    } catch (e) {
      document.getElementById("deferred-status").textContent = `save failed: ${e.message}`;
    }
  };

  async function hydrateFeed() {
    try {
      const resp = await fetch(`${API_BASE}/api/activity/recent?limit=100`);
      const rows = await resp.json();
      // Hydrated rows are in DESC order already.
      feedEvents = rows.map(r => ({
        type: r.signal_type,
        signal_type: r.signal_type,
        payload: r.payload || {},
        source: r.source_subsystem || "",
        timestamp: r.timestamp,
      }));
      renderFeed();
    } catch (e) {
      feedEl.innerHTML = `<div class="feed-empty">Feed hydration failed: ${escapeHtml(e.message)}</div>`;
    }
  }

  // -------------- websocket -----------------------------
  let ws = null;
  let reconnectTimer = null;

  function connectWs() {
    try {
      ws = new WebSocket(WS_URL);
    } catch (e) {
      wsPill.textContent = "WS: error";
      wsPill.className = "pill ws-state down";
      return;
    }
    ws.onopen = () => {
      wsPill.textContent = "WS: live";
      wsPill.className = "pill ws-state up";
      // First-frame auth: the daemon requires {type:"auth",token} as the FIRST
      // frame when a token is configured (it closes 4401 otherwise). The token
      // lives in this browser's localStorage — entered by the user, NEVER
      // embedded in the served HTML. When the daemon has no token (auth off),
      // this frame is harmlessly ignored. Auth must precede subscribe.
      try {
        const token = localStorage.getItem("prometheus_token");
        if (token) ws.send(JSON.stringify({ type: "auth", token }));
        ws.send(JSON.stringify({ type: "subscribe", payload: { channels: ["*"] } }));
      } catch {}
    };
    ws.onclose = (ev) => {
      if (ev && ev.code === 4401) {
        // Distinct, non-reconnecting auth-failure state — a retry loop would
        // just hammer the daemon with the same bad/absent token.
        wsPill.textContent = "WS: unauthorized — set token";
        wsPill.className = "pill ws-state down";
        return;
      }
      wsPill.textContent = "WS: down — reconnecting…";
      wsPill.className = "pill ws-state down";
      clearTimeout(reconnectTimer);
      reconnectTimer = setTimeout(connectWs, 3000);
    };
    ws.onerror = () => { /* close will follow */ };
    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        // Filter out the noisy welcome / subscribe ack from the feed.
        if (msg.type === "connected" || msg.type === "subscribed") return;
        pushEvent(msg);
        // Side-effect: refresh memory/skills panels when something interesting lands.
        if (msg.type === "memory_updated") refreshMemory();
        if (msg.type === "skill_created" || msg.type === "skill_refined") refreshSkills();
      } catch {
        // ignore malformed
      }
    };
  }

  // -------------- init ---------------------------------
  connectWs();
  hydrateFeed();
  refreshStatus();
  refreshMemory();
  refreshSkills();
  refreshDeferred();
  // Light periodic refresh — covers cases where signals don't trigger.
  setInterval(refreshStatus, 30_000);
  // No WS signal exists for config changes; a slow poll keeps a second
  // browser tab honest after a toggle elsewhere.
  setInterval(refreshDeferred, 30_000);
  setInterval(() => { feedEvents = [...feedEvents]; renderFeed(); }, 10_000);

  // ── Event delegation ────────────────────────────────────────────────
  // Replaces nine inline `on*=""` attributes. Two reasons, and the second is
  // the one that matters:
  //
  //  1. an inline handler is JavaScript written into an ATTRIBUTE, so every
  //     value interpolated near it has to survive HTML decoding AND then JS
  //     parsing. Getting that right twice, at every call site, forever, is
  //     the bug class this file had.
  //  2. `script-src 'self'` cannot allow inline handlers. Without removing
  //     them there is no Content-Security-Policy worth setting, and the
  //     bearer token in localStorage has nothing standing between it and the
  //     first script that ever does land.
  //
  // Data travels in `dataset`, which is never parsed as markup.
  document.addEventListener("click", (e) => {
    const el = e.target.closest("[data-action], [data-deferred]");
    if (!el) return;
    if (el.dataset.deferred !== undefined) {
      const v = el.dataset.deferred;
      setDeferred(v === "auto" ? "auto" : v === "on");
      return;
    }
    switch (el.dataset.action) {
      case "open-skill":
        openSkill(el.dataset.skill);
        break;
      case "toggle-pin":
        togglePin(el.dataset.skill, el.dataset.pinned === "1");
        break;
      case "toggle-row":
        el.classList.toggle("expanded");
        break;
    }
  });

  // The modal: backdrop closes, the panel itself does not, and Escape works —
  // which the inline version never offered.
  modalOverlay.addEventListener("click", (e) => {
    if (e.target === modalOverlay) closeModal();
  });
  document.getElementById("modal-close").addEventListener("click", closeModal);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && modalOverlay.classList.contains("open")) closeModal();
  });
