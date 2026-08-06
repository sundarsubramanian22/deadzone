/* ==========================================================================
   Deadzone dashboard — vanilla JS, hand-rolled SVG. No framework, no network,
   no build step. Opens from file:// with the wifi off.

   ONE data door: loadData(). Everything on this page is read through it and
   nothing else, so swapping the synthetic table for the real one is a one-line
   change in build.py — the front end never knows which it got.

   Every panel is defensive twice over. A payload arrives as
   {status, reason, data}: if it is missing, the panel renders a labelled empty
   state saying WHY; if a render throws, it is caught and turned into the same
   thing. There is no code path that leaves a blank chart with no explanation —
   on a projector, an unexplained blank and a crash look identical.

   THE ONE SEMANTIC RULE IN THIS FILE
   ----------------------------------
   Each condition has TWO word-error rates and they are not interchangeable:

     wer_spoke      WER over the clips on which the model actually emitted
                    words. This is the ONLY one that may be paired with a
                    confidence, because confidence is averaged over exactly
                    those clips. gap = conf - (1 - wer_spoke) is a subtraction
                    of two quantities measured on the same subset.

     wer_all_clips  WER over every clip, including the ones that came back
                    EMPTY. This is "how much of the corpus did this condition
                    destroy". Empty clips carry no confidence, so this number
                    must never be subtracted from a confidence.

   Pairing a subset confidence against a whole-corpus WER manufactures a gap
   out of silence. The hero scatter therefore plots wer_spoke; the factor grid
   plots wer_all_clips (so the fully-silent cells stay on the page instead of
   vanishing, which would delete the worst conditions in the study); and the
   readout prints both, labelled, side by side.

   Three condition categories follow from that, and they are NOT the same
   failure:
     DEAD ZONE      confidently WRONG   — words came back, and they were wrong
     SILENCE-DRIVEN flagged only by the mismatched all-clips pairing
     MUTE ZONE      entirely ABSENT     — no words on any clip, so no
                    confidence exists and a confidence-based monitor is
                    structurally blind to it
   A mute zone is never styled or labelled as a dead zone.
   ========================================================================== */
(function () {
  "use strict";

  // ---------------------------------------------------------------- the door
  function loadData() {
    return window.__DEADZONE_DATA__ || { meta: {}, models: {}, cross: {}, default_model: null };
  }

  var DATA = loadData();
  var STATE = {
    model: DATA.default_model || Object.keys(DATA.models || {})[0] || null,
    pinned: null,      // condition_name the user clicked, or null
    focus: null,       // condition_name currently in the hero readout
    alStep: 0,
    revealed: false    // the page-load reveal fires exactly once
  };

  // ------------------------------------------------------------- tiny helpers
  var NS = "http://www.w3.org/2000/svg";

  function el(tag, attrs, kids) { return apply(document.createElement(tag), attrs, kids); }
  function sv(tag, attrs, kids) { return apply(document.createElementNS(NS, tag), attrs, kids, true); }
  function apply(n, attrs, kids, isSvg) {
    attrs = attrs || {};
    for (var k in attrs) {
      var v = attrs[k];
      if (v === null || v === undefined) continue;
      if (k === "text") { n.textContent = String(v); }
      else if (k.slice(0, 2) === "on" && typeof v === "function") { n.addEventListener(k.slice(2), v); }
      else if (!isSvg && k === "className") { n.className = v; }
      else { n.setAttribute(k === "className" ? "class" : k, String(v)); }
    }
    (kids || []).forEach(function (c) {
      if (c === null || c === undefined || c === false) return;
      n.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
    });
    return n;
  }
  function clear(n) { while (n.firstChild) n.removeChild(n.firstChild); return n; }
  function num(v) { return typeof v === "number" && isFinite(v); }
  function pick() {  // first finite number among the arguments
    for (var i = 0; i < arguments.length; i++) if (num(arguments[i])) return arguments[i];
    return null;
  }
  function f1(v) { return num(v) ? v.toFixed(1) : "n/a"; }
  function f2(v) { return num(v) ? v.toFixed(2) : "n/a"; }
  function f3(v) { return num(v) ? v.toFixed(3) : "n/a"; }
  function pct(v) { return num(v) ? (100 * v).toFixed(0) + "%" : "n/a"; }
  function sign3(v) { return num(v) ? (v >= 0 ? "+" : "") + v.toFixed(3) : "n/a"; }
  function tidy(v) { return typeof v === "number" ? String(Math.round(v * 1000) / 1000) : String(v); }

  /* THE TWO ESTIMANDS. Read under both the current and the older field names so
     the page still renders from an older results table — a dashboard that
     hard-fails on a git tag is a dashboard you cannot demo from one. */
  function werSpoke(p) { return pick(p.wer_spoke, p.y_wer_spoke, p.y_wer); }
  function werAll(p) { return pick(p.wer_all_clips, p.y_wer_all_clips, p.y_wer); }
  function gapSpoke(p) { return pick(p.gap_spoke, p.gap); }
  function gapAll(p) { return pick(p.gap_all_clips); }
  function isMute(p) { return p.mute === true || p.category === "mute_zone" || p.mute_zone === true; }
  function isSilenceDriven(p) { return p.silence_driven === true || p.category === "silence_driven"; }
  function isDeadZone(p) { return p.dead_zone === true && !isMute(p); }

  /* WER -> the ONE sequential ramp. Six stops over a FIXED domain [0,1], so a
     colour means the same thing in every panel and between models: cold deep
     navy = intact, hot red = destroyed. The hot end IS --danger, which is why
     red on this page is always earned by the data and never merely applied. */
  var WER_STOPS = [0.0, 0.10, 0.22, 0.38, 0.58, 0.80];
  function werIdx(v) {
    var i = 0;
    for (var k = 0; k < WER_STOPS.length; k++) if (v >= WER_STOPS[k]) i = k;
    return i;
  }
  function werColor(v) { return num(v) ? "var(--wer-" + werIdx(v) + ")" : "var(--paper-2)"; }
  function werInk(v) { return num(v) ? "var(--wer-" + werIdx(v) + "-ink)" : "var(--ink-3)"; }

  /* The categorical set. FACTORS only, never magnitude — four hues, all cool or
     green-shifted so none can be mistaken for the ramp's warm end. */
  var FACTOR_COLOR = {
    rt60: "var(--f-reverb)", snr_db: "var(--f-noise)",
    codec: "var(--f-channel)", mic_rolloff: "var(--f-mic)", noise_type: "var(--ink-2)"
  };
  var FACTOR_KEYS = ["rt60", "snr_db", "noise_type", "codec", "mic_rolloff"];
  var FACTOR_UNIT = { rt60: " s", snr_db: " dB" };

  function scale(d0, d1, r0, r1) {
    var span = (d1 - d0) || 1;
    var f = function (v) { return r0 + (v - d0) * (r1 - r0) / span; };
    f.invert = function (p) { return d0 + (p - r0) * span / (r1 - r0); };
    return f;
  }
  function extent(arr, get) {
    var lo = Infinity, hi = -Infinity;
    arr.forEach(function (d) { var v = get(d); if (num(v)) { if (v < lo) lo = v; if (v > hi) hi = v; } });
    return isFinite(lo) ? [lo, hi] : [0, 1];
  }
  function uniq(arr) {
    var seen = [], out = [];
    arr.forEach(function (v) { if (seen.indexOf(String(v)) < 0) { seen.push(String(v)); out.push(v); } });
    return out;
  }
  /* Evenly spaced, used where the label must line up with a cell of a grid. */
  function ticks(lo, hi, n) {
    var out = [], step = (hi - lo) / n;
    for (var i = 0; i <= n; i++) out.push(lo + i * step);
    return out;
  }
  /* Round numbers, used on every continuous axis. An instrument that prints
     0.38 / 0.51 / 0.63 on its scale is not an instrument anyone reads. */
  function niceTicks(lo, hi, target) {
    var span = hi - lo;
    if (!(span > 0)) return [lo];
    var raw = span / (target || 5);
    var mag = Math.pow(10, Math.floor(Math.log(raw) / Math.LN10));
    var norm = raw / mag;
    var step = (norm >= 5 ? 10 : norm >= 2.5 ? 5 : norm >= 1.2 ? 2 : 1) * mag;
    var out = [], t = Math.ceil(lo / step - 1e-9) * step;
    for (; t <= hi + 1e-9; t += step) out.push(Math.round(t / step) * step);
    return out;
  }
  function snapUp(v, step) { return Math.ceil(v / step - 1e-9) * step; }
  function snapDown(v, step) { return Math.floor(v / step + 1e-9) * step; }
  /* Monospace at a known size has a knowable advance width, so a label that
     will not fit inside the dark field can be dropped before it spills out of
     it rather than after. */
  function fitsMono(text, px, size) { return String(text).length * size * 0.60 <= px; }
  function subhead(t) { return el("div", { className: "subhead", text: t }); }
  function sighead(t) { return el("div", { className: "sighead", text: t }); }
  /* A sub-state inside a panel that DID render. Deliberately NOT class "empty",
     so the panel-level "is this an unexplained hole" count stays honest. */
  function noticed(title, body) {
    return el("div", { className: "subempty" }, [el("b", { text: title }), el("span", { text: body })]);
  }

  // --------------------------------------------------------- the empty state
  function emptyState(host, title, reason) {
    clear(host).appendChild(el("div", { className: "empty" }, [
      el("div", { className: "t", text: title }),
      el("div", { className: "r", text: reason || "no reason recorded" })
    ]));
  }

  function panel(hostId, payload, render, missingTitle) {
    var host = document.getElementById(hostId);
    if (!host) return;
    if (!payload || payload.status !== "ok" || !payload.data) {
      emptyState(host, missingTitle || "No data for this panel yet",
        (payload && payload.reason) || "This payload was not present in the build. "
        + "Rebuild with `python3 dashboard/build.py` once the upstream analysis "
        + "module and its results file exist.");
      return;
    }
    try {
      clear(host);
      render(host, payload.data);
    } catch (err) {
      emptyState(host, "This panel failed to render",
        String((err && err.message) || err) + "\n(the rest of the dashboard is unaffected)");
    }
  }

  // ======================================================================
  // HERO — the silent-failure map
  // ======================================================================

  function renderHero(host, d) {
    var pts = (d.points || []).filter(function (p) { return num(p.x_mean_conf) && num(werSpoke(p)); });
    if (!pts.length) throw new Error("no condition has both a confidence and a WER");

    // The readout leads. It is the whole thesis in one block: a high confidence
    // number sitting directly above visibly wrong words.
    var lead = el("div", { className: "hero-lead detail" });
    host.appendChild(lead);

    var map = el("div", { className: "hero-map" });
    var left = el("div", {});
    map.appendChild(left);
    map.appendChild(tallyStrip(d));
    host.appendChild(map);

    // ---- the scatter --------------------------------------------------
    var q = d.quadrant || {};
    var W = 800, H = 470, M = { t: 18, r: 20, b: 52, l: 62 };
    var xs = extent(pts, function (p) { return p.x_mean_conf; });
    var ys = extent(pts, function (p) { return werSpoke(p); });
    var xlo = Math.max(0, snapDown(Math.min(xs[0], num(q.conf_hi_raw) ? q.conf_hi_raw : 1) - 0.03, 0.05));
    var xhi = Math.min(1.0, snapUp(Math.max(xs[1], 0.6) + 0.01, 0.05));
    /* NOT clamped to 1.0. A model that hallucinates fluent text returns more
       words than the reference, so WER goes above 1 — clamping the axis would
       silently push exactly the worst conditions off the top of the chart. */
    var yraw = Math.max(ys[1], num(q.wer_hi) ? q.wer_hi + 0.12 : 0.5) + 0.02;
    var yhi = snapUp(yraw, yraw > 1.4 ? 0.5 : 0.1);
    var x = scale(xlo, xhi, M.l, W - M.r);
    var y = scale(0, yhi, H - M.b, M.t);

    var svg = sv("svg", { className: "chart", viewBox: "0 0 " + W + " " + H,
      preserveAspectRatio: "xMidYMid meet", role: "img",
      "aria-label": "mean word confidence against word error rate, one point per acoustic condition" });

    niceTicks(0, yhi, 5).forEach(function (v) {
      svg.appendChild(sv("line", { className: "gridline", x1: M.l, x2: W - M.r, y1: y(v), y2: y(v) }));
      svg.appendChild(sv("text", { className: "tick", x: M.l - 9, y: y(v) + 4, "text-anchor": "end", text: v.toFixed(2) }));
    });

    /* THE DARK FIELD. The only one on the page. Not a highlight — the finding:
       the coverage shadow the model drives into while still reporting a high
       confidence. */
    var fieldLabel = null;
    if (num(q.conf_hi_raw) && num(q.wer_hi)) {
      var fx = x(q.conf_hi_raw), fw = Math.max(0, x(xhi) - fx);
      var fy = y(yhi), fh = Math.max(0, y(q.wer_hi) - fy);
      svg.appendChild(sv("rect", { x: fx, y: fy, width: fw, height: fh, fill: "var(--field)" }));
      // Only what fits inside the field is drawn inside it. A caption that
      // spills past the shadow's edge breaks the one metaphor this page has.
      // Held back and appended after the points, so a model with a crowded
      // dead zone does not bury its own label.
      var inner = fw - 26, ly = fy + 25;
      fieldLabel = sv("g", {});
      fieldLabel.appendChild(sv("rect", { x: fx, y: fy, width: Math.min(fw, 260), height: 74,
        fill: "var(--field)", "fill-opacity": 0.92 }));
      fieldLabel.appendChild(sv("text", { className: "field-label", x: fx + 13, y: ly, "font-size": 13, text: "DEAD ZONE" }));
      [[13, "confident AND wrong", 11.5],
       [11.5, "conf >= " + f2(q.conf_hi_raw) + "   WER >= " + f2(q.wer_hi), 11],
       [11.5, "top " + pct(1 - (q.conf_pct_hi || 0)) + " of its own confidence", 11]
      ].forEach(function (L) {
        if (!fitsMono(L[1], inner, L[2])) return;
        ly += L[0] + 5;
        fieldLabel.appendChild(sv("text", { className: "field-sub", x: fx + 13, y: ly, "font-size": L[2], text: L[1] }));
      });
    }

    niceTicks(xlo, xhi, 5).forEach(function (v) {
      svg.appendChild(sv("text", { className: "tick", x: x(v), y: H - M.b + 19, "text-anchor": "middle", text: v.toFixed(2) }));
    });
    svg.appendChild(sv("line", { className: "axis-line", x1: M.l, x2: W - M.r, y1: H - M.b, y2: H - M.b }));
    svg.appendChild(sv("line", { className: "axis-line", x1: M.l, x2: M.l, y1: M.t, y2: H - M.b }));
    svg.appendChild(sv("text", { className: "axis-title", x: (M.l + W - M.r) / 2, y: H - 10,
      "text-anchor": "middle", text: (d.axes && d.axes.x) || "mean word confidence" }));
    svg.appendChild(sv("text", { className: "axis-title", x: -(M.t + H - M.b) / 2, y: 16,
      "text-anchor": "middle", transform: "rotate(-90)", text: "WER, clips the model spoke on" }));

    // dead zones drawn last so they sit on top of the crowd
    pts.slice().sort(function (a, b) { return (isDeadZone(a) ? 1 : 0) - (isDeadZone(b) ? 1 : 0); })
      .forEach(function (p) {
        var dz = isDeadZone(p);
        var inField = num(q.conf_hi_raw) && p.x_mean_conf >= q.conf_hi_raw && werSpoke(p) >= q.wer_hi;
        // Red on near-black is the one place the danger accent loses contrast,
        // and it is the one place it matters most. A paper halo behind the ring
        // keeps the flagged points legible from the back of a room.
        if (dz) svg.appendChild(sv("circle", { cx: x(p.x_mean_conf), cy: y(werSpoke(p)), r: 10.5,
          fill: "none", stroke: "var(--paper)", "stroke-width": 2 }));
        var c = sv("circle", {
          className: "pt",
          cx: x(p.x_mean_conf), cy: y(werSpoke(p)), r: dz ? 7.5 : 5,
          fill: werColor(werSpoke(p)),
          stroke: dz ? "var(--danger)" : (inField ? "var(--field-ink)" : "var(--ink-3)"),
          "stroke-width": dz ? 2.5 : 1,
          tabindex: 0, role: "button",
          "aria-label": p.condition_name + ": confidence " + f2(p.x_mean_conf)
            + ", WER " + f2(werSpoke(p)) + (dz ? ", dead zone" : "")
        });
        function show() { if (!STATE.pinned) setLead(lead, p, d, false); }
        c.addEventListener("mouseenter", show);
        c.addEventListener("focus", show);
        c.addEventListener("click", function () {
          STATE.pinned = (STATE.pinned === p.condition_name) ? null : p.condition_name;
          setLead(lead, p, d, false);
        });
        svg.appendChild(c);
      });
    if (fieldLabel) svg.appendChild(fieldLabel);

    left.appendChild(svg);

    /* The pairing note. Not optional: the vertical axis is one half of a
       subtraction whose other half is the horizontal axis, and a reader cannot
       tell from a scatter which WER it is looking at. */
    var sc = d.silence_counts || {}, cat = d.categories || {};
    left.appendChild(el("div", { className: "note" }, [
      el("b", { text: "Same-subset pairing. " }),
      "The vertical axis is WER over the clips on which the model emitted words — the same clips its "
      + "confidence is averaged over — so the two axes are subtractable and the gap is real. Clips that "
      + "came back EMPTY carry no confidence at all: "
      + (num(sc.conditions_with_any_silent_clip) ? sc.conditions_with_any_silent_clip : 0)
      + " conditions had at least one, and "
      + (num(sc.mute_no_words_on_any_clip) ? sc.mute_no_words_on_any_clip : (cat.n_mute_zones || 0))
      + " returned nothing on every clip, so they cannot appear on this plot at all. They are counted at "
      + "right and they stay on the factor grid below, which plots the all-clips WER."
    ]));
    left.appendChild(el("div", { className: "legend-inline", text:
      "Fill is WER on the shared ramp; a ringed point is a flagged dead zone. Hover, tab or click any point "
      + "to load it into the readout above; click again to unpin." }));

    var corr = d.correlation || {};
    var rho = pick(corr.spearman_confpct_vs_wer, corr.spearman);
    if (num(rho)) {
      left.appendChild(el("div", { className: "rho" }, [
        "spearman(confidence, WER) = " + f3(rho) + "   n = " + (corr.n || "?") + "   ",
        el("em", { text: (corr.verdict ? corr.verdict + ". " : "")
          + "The finding is not that the model is blind — globally it tracks its own error well. It is that a "
          + "system calibrated on that average behaviour will trust it precisely inside the dark field."
          + (num(d.n_failed_rows_excluded) && d.n_failed_rows_excluded > 0
             ? " " + d.n_failed_rows_excluded + " failed rows excluded, never averaged in." : "") })
      ]));
    }

    setLead(lead, pickOpening(pts), d, !STATE.revealed);
    STATE.revealed = true;
  }

  /* The hero opens on the ranked worst dead zone — worst meaning the largest
     same-subset confidence-accuracy gap, which is the quantity this project is
     about. It prefers a condition with a stored transcript, because a hero with
     no words in it makes no argument. */
  function pickOpening(pts) {
    var byGap = function (a, b) { return (num(gapSpoke(b)) ? gapSpoke(b) : -9) - (num(gapSpoke(a)) ? gapSpoke(a) : -9); };
    if (STATE.focus) {
      var held = pts.filter(function (p) { return p.condition_name === STATE.focus; });
      if (held.length) return held[0];
    }
    var dz = pts.filter(isDeadZone).sort(byGap);
    var withEx = dz.filter(function (p) { return p.example && (p.example.edits || []).length; });
    return withEx[0] || dz[0] || pts.slice().sort(byGap)[0];
  }

  function setLead(host, p, d, animate) {
    STATE.focus = p.condition_name;
    var mute = isMute(p), dz = isDeadZone(p), sd = isSilenceDriven(p);
    clear(host);
    host.className = "hero-lead detail" + (dz ? " dz" : (mute ? " mz" : ""));

    // Three categories, three treatments. A mute zone is ABSENT, not wrong; a
    // confidence-based monitor is structurally blind to it, so it must never
    // wear the dead-zone badge.
    var flag = dz ? el("span", { className: "dzflag", text: "DEAD ZONE" })
      : (mute ? el("span", { className: "dzflag mute", text: "MUTE ZONE" })
        : (sd ? el("span", { className: "dzflag muted", text: "SILENCE-DRIVEN" }) : null));
    host.appendChild(el("div", { className: "lead-top" }, [
      el("span", { className: "cname", text: p.condition_name }),
      flag,
      STATE.pinned === p.condition_name ? el("span", { className: "dzflag muted", text: "PINNED" }) : null
    ]));

    var chips = el("div", { className: "chips" });
    FACTOR_KEYS.forEach(function (k, i) {
      if (p[k] === undefined || p[k] === null || p[k] === "") return;
      chips.appendChild(el("span", { className: "chip f" + i }, [
        el("i", { text: k }), " " + tidy(p[k]) + (FACTOR_UNIT[k] || "")
      ]));
    });
    host.appendChild(chips);

    // Both estimands, side by side, labelled. Never one without the other.
    /* Only a flagged condition earns the accent, and CONFIDENCE never takes it:
       the point of the readout is that the model's own number stays calm while
       the two beside it are on fire. */
    var ws = werSpoke(p), wa = werAll(p), gs = gapSpoke(p), ga = gapAll(p);
    host.appendChild(el("div", { className: "readouts" + (animate ? " reveal" : "") }, [
      readout("confidence", f3(p.x_mean_conf), "what the model reported", 0, false),
      readout("WER", f3(ws), "spoke " + f3(ws) + "  /  all clips " + f3(wa), 1, dz),
      readout("gap", sign3(gs), "same-subset " + sign3(gs) + "  /  all-clips " + sign3(ga), 2, dz)
    ]));

    var ex = p.example;
    if (ex && (ex.edits || []).length) {
      host.appendChild(el("div", { className: "diffhead",
        text: "transcript diff · clip " + (ex.clip_id || "?") + " · WER " + f2(ex.wer)
              + (ex.truncated ? " · truncated" : "") }));
      host.appendChild(renderDiff(ex.edits, animate));
      host.appendChild(el("div", { className: "diffkey" }, [
        keyBox("m", "correct"), keyBox("sub", "substituted"),
        keyBox("del", "deleted — the hole is a word that never arrived"), keyBox("ins", "inserted")
      ]));
    } else {
      host.appendChild(el("div", { className: "diffhead", text: "transcript diff · none stored" }));
      host.appendChild(el("p", { className: "hint", text:
        "No representative clip was stored for this condition, so there is no word-level diff to show." }));
    }

    var silent = num(p.n_silent) ? p.n_silent : 0;
    var rows = [
      ["confidence percentile, within this model", num(p.conf_pct) ? pct(p.conf_pct) : "n/a"],
      ["clips in cell", (num(p.n_clips) ? p.n_clips : "?")
        + (num(p.n_ref_total) ? "  (" + p.n_ref_total + " reference words)" : "")],
      ["clips returning an EMPTY transcript", silent + " of " + (num(p.n_clips) ? p.n_clips : "?")]
    ];
    if (num(p.gap_inflation) && Math.abs(p.gap_inflation) > 1e-9) {
      rows.push(["gap inflation contributed by those silent clips", sign3(p.gap_inflation)]);
    }
    host.appendChild(statList(rows.map(function (r) { return [r[0], r[1], ""]; })));

    if (p.label) host.appendChild(el("p", { className: "prov", text: p.label }));
    if (mute && d.category_meaning && d.category_meaning.mute_zone) {
      host.appendChild(el("p", { className: "prov", text: d.category_meaning.mute_zone }));
    } else if (sd && d.category_meaning && d.category_meaning.silence_driven) {
      host.appendChild(el("p", { className: "prov", text: d.category_meaning.silence_driven }));
    }
  }

  function readout(k, v, unit, i, hot) {
    return el("div", { className: "readout" + (hot ? " hot" : ""), style: "--r:" + i }, [
      el("span", { className: "k", text: k }),
      el("span", { className: "v", text: v }),
      el("span", { className: "u", text: unit })
    ]);
  }
  function keyBox(cls, label) {
    return el("span", {}, [el("i", { className: cls }), el("span", { text: label })]);
  }

  /* Reference above, model below, word for word. A deletion is drawn as an
     explicit hole the width of the word that never arrived: deletions are the
     dominant error type in this study, and a diff that renders them as nothing
     at all would misrepresent the shape of the failure. */
  function renderDiff(edits, animate) {
    var box = el("div", { className: "tokens" + (animate ? " reveal" : "") });
    box.appendChild(el("div", { className: "tok rail" }, [
      el("span", { className: "ref", text: "REFERENCE" }),
      el("span", { className: "hyp", text: "MODEL" })
    ]));
    edits.forEach(function (e, i) {
      var op = e[0], ref = e[1], hyp = e[2];
      var w = String(op === "ins" ? (hyp || "") : (ref || "")).length || 3;
      var kids;
      if (op === "del") kids = [el("span", { className: "ref", text: ref }), el("span", { className: "hyp" })];
      else if (op === "ins") kids = [el("span", { className: "ref" }), el("span", { className: "hyp", text: hyp })];
      else kids = [el("span", { className: "ref", text: ref }), el("span", { className: "hyp", text: hyp })];
      box.appendChild(el("div", { className: "tok " + op, style: "--i:" + i + ";--w:" + w }, kids));
    });
    return box;
  }

  /* The strip is a first-class result, not a footnote. Seven of nova-3's
     conditions returned an empty transcript on EVERY clip: they are the worst
     conditions in the study and they cannot appear on the scatter at all. */
  function tallyStrip(d) {
    var c = d.quadrant_counts || {}, sc = d.silence_counts || {}, cat = d.categories || {};
    var t = el("div", { className: "tally" });
    function row(n, k, cls) {
      t.appendChild(el("div", { className: "row" + (cls ? " " + cls : "") }, [
        el("span", { className: "n", text: num(n) ? String(n) : "—" }),
        el("span", { className: "k", text: k })
      ]));
    }
    t.appendChild(el("div", { className: "sep", text: "plotted above · by quadrant" }));
    row(pick(c.dead_zone_confident_and_wrong, cat.n_dead_zones),
        "DEAD ZONE — words came back, and they were wrong", "dz");
    row(c.confident_and_right, "confident and right");
    row(c.loud_failure_unconfident_and_wrong, "loud failure — unconfident and wrong");
    row(c.unconfident_but_right, "unconfident but right");
    t.appendChild(el("div", { className: "sep", text: "cannot be plotted above" }));
    row(pick(sc.silence_driven, cat.n_silence_driven),
        "SILENCE-DRIVEN — a gap that comes from clips that vanished, not from wrong words", "sd");
    row(pick(sc.mute_no_words_on_any_clip, cat.n_mute_zones, c.unplaceable_no_confidence),
        "MUTE ZONE — nothing returned on any clip, so no confidence exists to be wrong", "mz");
    return t;
  }

  // ======================================================================
  // FACTOR-SPACE GRID — the same conditions, laid out in the design
  // ======================================================================

  /* Faceted by (noise_type, codec, mic_rolloff) so that every cell holds
     EXACTLY ONE condition. Faceting on only two of the three collapsed three
     mic_rolloff levels into one cell and silently let the last one win, which
     discarded two thirds of the grid with no error message.

     This grid plots wer_all_clips: it is the "how much of the corpus did this
     condition destroy" question, and it is the only pairing under which the
     mute cells stay on the page instead of vanishing. */
  function renderHeatmap(host, d) {
    var pts = (d.points || []).filter(function (p) { return num(werAll(p)) && num(p.rt60) && num(p.snr_db); });
    if (!pts.length) throw new Error("no conditions carry both a WER and factor coordinates");

    var facets = {};
    pts.forEach(function (p) {
      var fk = [p.noise_type || "?", p.codec || "none", num(p.mic_rolloff) ? p.mic_rolloff : 0].join("|");
      (facets[fk] = facets[fk] || []).push(p);
    });
    var keys = Object.keys(facets).filter(function (k) { return facets[k].length >= 2; });
    if (!keys.length) throw new Error("not enough conditions per facet to draw a grid");
    var NOISE_ORDER = { babble: 0, engine: 1, road: 2 };
    keys.sort(function (a, b) {
      var A = a.split("|"), B = b.split("|");
      var na = NOISE_ORDER[A[0]] === undefined ? 9 : NOISE_ORDER[A[0]];
      var nb = NOISE_ORDER[B[0]] === undefined ? 9 : NOISE_ORDER[B[0]];
      return na - nb || A[1].localeCompare(B[1]) || (+A[2]) - (+B[2]);
    });

    // The gap encoding is normalised across the WHOLE grid, never per facet, so
    // a thick border means the same thing in every panel of the small multiple.
    var gaps = pts.filter(function (p) { return !isMute(p) && num(gapSpoke(p)); })
                  .map(function (p) { return gapSpoke(p); });
    var gmin = gaps.length ? Math.min.apply(null, gaps) : 0;
    var gmax = gaps.length ? Math.max.apply(null, gaps) : 1;
    var gspan = (gmax - gmin) || 1;

    var wrap = el("div", { className: "facets" });
    keys.forEach(function (k) { wrap.appendChild(facetSvg(k, facets[k], gmin, gspan)); });
    host.appendChild(wrap);

    var ramp = el("div", { className: "ramp" }, [
      el("span", { className: "lbl", text: "WER, all clips" }),
      el("span", { className: "sw" }),
      el("span", { text: "0.00 to 1.00" })
    ]);
    var sw = ramp.querySelector(".sw");
    for (var i = 0; i < 6; i++) sw.appendChild(el("i", { style: "background:var(--wer-" + i + ")" }));
    ramp.appendChild(el("span", { className: "key" }, [
      el("i", { className: "thin" }), el("span", { text: "small gap" }),
      el("i", { className: "thick" }), el("span", { text: "large gap — more overconfident" })
    ]));
    ramp.appendChild(el("span", { className: "key" }, [
      el("i", {}), el("span", { text: "solid red = dead zone, confidently wrong" }),
      el("i", { className: "dashed" }), el("span", { text: "dashed red = MUTE, nothing returned — no gap exists to encode" })
    ]));
    host.appendChild(ramp);
    host.appendChild(el("p", { className: "prov", text:
      "One condition per cell: the facets split all three of noise type, codec and mic_rolloff, so nothing is "
      + "averaged and nothing is dropped. The engine and road arms were run at a reduced design (two rt60 x two "
      + "SNR levels, two rolloff levels), which is why their facets are smaller — that is the experiment, not a "
      + "missing panel." }));
  }

  function facetSvg(key, conds, gmin, gspan) {
    var parts = key.split("|");
    var xsv = uniq(conds.map(function (c) { return c.rt60; })).sort(function (a, b) { return a - b; });
    var ysv = uniq(conds.map(function (c) { return c.snr_db; })).sort(function (a, b) { return b - a; });
    var cw = 62, ch = 40, M = { t: 8, r: 8, b: 34, l: 44 };
    var W = M.l + cw * xsv.length + M.r, H = M.t + ch * ysv.length + M.b;
    /* max-width pins the cell to a constant size across the small multiple. The
       engine and road arms have four cells where babble has sixteen; letting
       them stretch to fill the card would make a reduced design look like a
       coarse one, and a cell's size would stop meaning anything. */
    var svg = sv("svg", { className: "chart", viewBox: "0 0 " + W + " " + H,
      preserveAspectRatio: "xMinYMin meet", role: "img", style: "max-width:" + W + "px",
      "aria-label": "WER by rt60 and SNR for " + parts[0] + " noise, codec " + parts[1] + ", mic rolloff " + parts[2] });

    var byKey = {};
    conds.forEach(function (c) { byKey[c.rt60 + "|" + c.snr_db] = c; });

    ysv.forEach(function (snr, r) {
      svg.appendChild(sv("text", { className: "tick", x: M.l - 7, y: M.t + r * ch + ch / 2 + 4,
        "text-anchor": "end", text: String(snr) }));
      xsv.forEach(function (rt, c) {
        var cell = byKey[rt + "|" + snr];
        var gx = M.l + c * cw, gy = M.t + r * ch;
        if (!cell) {
          svg.appendChild(sv("rect", { x: gx + 2, y: gy + 2, width: cw - 4, height: ch - 4,
            fill: "var(--paper-2)", stroke: "var(--rule)", "stroke-dasharray": "3 3" }));
          svg.appendChild(sv("text", { x: gx + cw / 2, y: gy + ch / 2 + 4, "text-anchor": "middle",
            "font-size": 11, fill: "var(--ink-3)", text: "–" }));
          return;
        }
        var wv = werAll(cell), mute = isMute(cell), dz = isDeadZone(cell);
        /* Second encoding: border WEIGHT is the same-subset confidence-accuracy
           gap. A MUTE cell has NO confidence and therefore NO gap, so it does
           not borrow the encoding at all — a thin border would read "barely
           overconfident" when the truth is "there was nothing to be confident
           about". Dashed red says absent; solid red says confidently wrong. */
        var bw, stroke, dash = null;
        if (mute) { bw = 2; stroke = "var(--danger)"; dash = "4 3"; }
        else if (dz) { bw = 2.5; stroke = "var(--danger)"; }
        else {
          var g = num(gapSpoke(cell)) ? gapSpoke(cell) : gmin;
          bw = 1 + 3.5 * Math.max(0, Math.min(1, (g - gmin) / gspan));
          stroke = "var(--ink-3)";
        }
        var rect = sv("rect", { x: gx + 2, y: gy + 2, width: cw - 4, height: ch - 4,
          fill: werColor(wv), stroke: stroke, "stroke-width": bw });
        if (dash) rect.setAttribute("stroke-dasharray", dash);
        svg.appendChild(rect);
        svg.appendChild(sv("text", { x: gx + cw / 2, y: gy + ch / 2 + 5, "text-anchor": "middle",
          "font-size": 12.5, "font-weight": 600, "letter-spacing": "-0.03em",
          fill: werInk(wv), text: f2(wv) }));
        svg.appendChild(sv("title", { text: cell.condition_name
          + "\nWER all clips  " + f3(wv)
          + "\nWER spoke      " + (mute ? "no words returned" : f3(werSpoke(cell)))
          + "\nsilent clips   " + (num(cell.n_silent) ? cell.n_silent : 0) + " of " + (num(cell.n_clips) ? cell.n_clips : "?")
          + "\ngap same-subset " + (mute ? "none — no confidence exists" : sign3(gapSpoke(cell)))
          + (dz ? "\nDEAD ZONE — confidently wrong" : "")
          + (mute ? "\nMUTE ZONE — nothing returned on any clip" : "")
          + (isSilenceDriven(cell) ? "\nSILENCE-DRIVEN — flagged only by the all-clips pairing" : "") }));
      });
    });
    xsv.forEach(function (rt, c) {
      svg.appendChild(sv("text", { className: "tick", x: M.l + c * cw + cw / 2, y: H - M.b + 16,
        "text-anchor": "middle", text: String(rt) }));
    });
    svg.appendChild(sv("text", { className: "axis-title", x: M.l + (W - M.l - M.r) / 2, y: H - 5,
      "text-anchor": "middle", "font-size": 10.5, text: "rt60 (s)" }));
    svg.appendChild(sv("text", { className: "axis-title", x: -(M.t + (H - M.t - M.b) / 2), y: 10,
      "text-anchor": "middle", transform: "rotate(-90)", "font-size": 10.5, text: "SNR (dB)" }));

    return el("div", { className: "facet" }, [
      el("h3", {}, [el("b", { text: parts[0] }), " · " + parts[1] + " · roll " + parts[2]]),
      el("div", { className: "scroll-x" }, [svg])
    ]);
  }

  // ======================================================================
  // FINGERPRINTS
  // ======================================================================

  function renderFingerprints(host, d) {
    var fams = d.families || [];
    if (!fams.length) throw new Error("no factor families had enough rows to profile");

    host.appendChild(el("div", { className: "edit-key" }, [
      swatch("var(--ink)", "deletions — the word never reached the decoder"),
      swatch("var(--f-channel)", "substitutions — a different word came back"),
      swatch("var(--f-noise)", "insertions — a word nobody said"),
      el("span", {}, [el("i", { style: "background:var(--ink);width:3px;height:15px" }),
                      el("span", { text: "entity error rate" })])
    ]));

    var maxRate = Math.max(0.35, Math.max.apply(null, fams.map(function (f) {
      return Math.max(f.wer || 0, f.entity_error_rate || 0);
    })));

    var tbl = el("table", { className: "grid" });
    tbl.appendChild(el("thead", {}, [el("tr", {}, [
      el("th", { text: "condition family" }),
      el("th", { text: "edit composition, share of reference words" }),
      el("th", { text: "dominant" }),
      el("th", { className: "num", text: "WER" }),
      el("th", { className: "num", text: "entity ER" }),
      el("th", { text: "implied fix" })
    ])]));
    var tb = el("tbody", {});
    fams.forEach(function (f) {
      tb.appendChild(el("tr", {}, [
        el("td", { className: "name" }, [
          el("code", { text: f.family }),
          el("div", { className: "sub2", text: (f.contrast || "")
            + (f.n_conditions ? "  ·  " + f.n_conditions + " conditions, " + f.n_ref + " ref words" : "") })
        ]),
        el("td", {}, [stackedBar(f, maxRate)]),
        el("td", {}, [el("span", { className: "dom " + (f.dominant_edit || "sub"), text: f.dominant_edit || "?" })]),
        el("td", { className: "num", text: f2(f.wer) }),
        el("td", { className: "num", text: num(f.entity_error_rate) ? f2(f.entity_error_rate) : "n/a" }),
        el("td", { className: "fix" }, [
          el("span", { text: f.implied_fix || "—" }),
          f.caveat ? el("div", { className: "sub2", text: f.caveat }) : null
        ])
      ]));
    });
    tbl.appendChild(tb);
    host.appendChild(el("div", { className: "scroll-x" }, [tbl]));

    if ((d.signature_notes || []).length) {
      host.appendChild(sighead("measured signatures — effect size against the rest of the grid"));
      var ul = el("ul", { className: "sigs" });
      d.signature_notes.forEach(function (s) { ul.appendChild(el("li", { text: s })); });
      host.appendChild(ul);
    }
    if (d.entity_note) host.appendChild(el("p", { className: "prov", text: d.entity_note }));
  }

  function swatch(color, label) {
    return el("span", {}, [el("i", { style: "background:" + color }), el("span", { text: label })]);
  }

  function stackedBar(f, maxRate) {
    var W = 250, H = 24;
    var svg = sv("svg", { viewBox: "0 0 " + W + " " + H, width: W, height: H,
      role: "img", "aria-label": f.family + " edit composition" });
    var x = scale(0, maxRate, 0, W - 2);
    var segs = [["del", f.del_rate, "var(--ink)"], ["sub", f.sub_rate, "var(--f-channel)"],
                ["ins", f.ins_rate, "var(--f-noise)"]];
    var acc = 0;
    segs.forEach(function (s) {
      var v = num(s[1]) ? s[1] : 0;
      if (v <= 0) return;
      svg.appendChild(sv("rect", { x: x(acc), y: 4, width: Math.max(1, x(acc + v) - x(acc)), height: H - 9,
        fill: s[2] }, [sv("title", { text: s[0] + " " + f3(v) + " of reference words" })]));
      acc += v;
    });
    svg.appendChild(sv("rect", { x: 0, y: 4, width: W - 2, height: H - 9, fill: "none", stroke: "var(--rule-2)" }));
    if (num(f.entity_error_rate)) {
      var ex = x(Math.min(f.entity_error_rate, maxRate));
      svg.appendChild(sv("line", { x1: ex, x2: ex, y1: 0, y2: H, stroke: "var(--ink)", "stroke-width": 3 },
        [sv("title", { text: "entity error rate " + f3(f.entity_error_rate) })]));
    }
    return svg;
  }

  // ======================================================================
  // SENSITIVITY + THE PRE-REGISTRATION VERDICT
  // ======================================================================

  function renderSensitivity(host, d) {
    var v = d.verdict || {};
    var cls = v.status === "CONFIRMED" ? "confirmed"
      : (v.status === "NOT CONFIRMED" ? "notconfirmed" : "unresolved");
    host.appendChild(el("div", { className: "verdict " + cls }, [
      el("span", { className: "st", text: "pre-registered " + (v.pair ? v.pair.join(" × ") : "interaction")
        + " — " + (v.status || "UNRESOLVED") }),
      el("p", { text: v.sentence || "no verdict sentence was produced" })
    ]));

    var bars = d.gap_bars || [];
    var cols = el("div", { className: "two-col" });
    cols.appendChild(bars.length ? gapChart(bars)
      : noticed("No Sobol indices in this build",
                "results/sobol.json was absent and no surrogate could be fitted from the master table."));

    var right = el("div", {});
    right.appendChild(subhead("which pair interacts — ranking only"));
    var ol = el("ol", { className: "ranked" });
    (d.s2_pairs || []).slice(0, 6).forEach(function (p) {
      ol.appendChild(el("li", {}, [
        el("code", { text: (p.pair || []).join(" × ") }),
        num(p.S2) ? el("span", { text: "  S2 " + f3(p.S2) }) : null
      ]));
    });
    if (!(d.s2_pairs || []).length) ol.appendChild(el("li", { text: "no second-order pairs available" }));
    right.appendChild(ol);
    if (d.s2_caveat) right.appendChild(el("div", { className: "caveat", text: d.s2_caveat }));
    cols.appendChild(right);
    host.appendChild(cols);

    var ci = d.counterintuitive;
    if (ci && (ci.cells || []).length) {
      /* The proposer can emit the same cell many times over. Six identical
         lines read as six findings when they are one; collapse them and say
         how many proposals each line stands for. */
      var seen = {}, rows = [];
      ci.cells.forEach(function (c) {
        var k = [c.kind, c.factor, c.moderator, c.note].join("|");
        if (seen[k]) { seen[k].n += 1; return; }
        seen[k] = { n: 1, c: c }; rows.push(seen[k]);
      });
      host.appendChild(sighead("counterintuitive cells · " + (ci.presentable_as_measured
        ? "confirmed against the oracle" : "proposed by the surrogate, NOT reproduced by the oracle")));
      var ul = el("ul", { className: "sigs" });
      rows.slice(0, 6).forEach(function (r) {
        ul.appendChild(el("li", { text: (r.c.kind || "cell") + " · " + (r.c.factor || "?")
          + (r.c.moderator ? " moderated by " + r.c.moderator : "")
          + (r.c.note ? " — " + r.c.note : "")
          + (r.n > 1 ? "   [" + r.n + " proposals collapsed]" : "") }));
      });
      host.appendChild(ul);
      if (!ci.presentable_as_measured) {
        host.appendChild(el("div", { className: "caveat", text:
          "Surrogate predictions that real transcription did not reproduce (" + (ci.n_confirmed || 0)
          + " of " + (ci.n_candidates || rows.length) + " confirmed), so they are NOT presented as measured "
          + "surprises. The surrogate treats rt60 as a continuous coordinate, but every rt60 request is served "
          + "by the nearest measured impulse response — a different real room each time — so the axis is not "
          + "smooth and the surrogate keeps proposing cells the oracle cannot reproduce." }));
      }
    }
    if (d.provenance) host.appendChild(el("p", { className: "prov", text: d.provenance }));
  }

  function gapChart(bars) {
    var rowH = 46, M = { t: 30, r: 26, b: 42, l: 112 };
    var W = 580, H = M.t + rowH * bars.length + M.b;
    var lo = Math.min(0, Math.min.apply(null, bars.map(function (b) { return num(b.gap_lo) ? b.gap_lo : 0; })));
    var hi = Math.max.apply(null, bars.map(function (b) {
      return Math.max(num(b.ST) ? b.ST : 0, num(b.gap_hi) ? b.gap_hi : 0);
    }));
    var x = scale(Math.min(lo, 0), Math.max(hi, 0.1) * 1.06, M.l, W - M.r);
    var svg = sv("svg", { className: "chart", viewBox: "0 0 " + W + " " + H,
      preserveAspectRatio: "xMidYMid meet", role: "img",
      "aria-label": "first-order and total-order Sobol indices with interaction confidence intervals" });

    ticks(Math.min(lo, 0), Math.max(hi, 0.1) * 1.06, 5).forEach(function (t) {
      svg.appendChild(sv("line", { className: "gridline", x1: x(t), x2: x(t), y1: M.t - 8, y2: H - M.b }));
      svg.appendChild(sv("text", { className: "tick", x: x(t), y: H - M.b + 17, "text-anchor": "middle", text: t.toFixed(2) }));
    });
    svg.appendChild(sv("line", { className: "axis-line", x1: x(0), x2: x(0), y1: M.t - 8, y2: H - M.b }));
    svg.appendChild(sv("text", { className: "axis-title", x: (M.l + W - M.r) / 2, y: H - 6,
      "text-anchor": "middle", text: "variance share  ·  bar = S1 alone, notch = ST total, whisker = ST−S1 with 95% CI" }));

    bars.forEach(function (b, i) {
      var yy = M.t + i * rowH, col = FACTOR_COLOR[b.factor] || "var(--ink-2)";
      svg.appendChild(sv("text", { x: M.l - 10, y: yy + 14, "text-anchor": "end", "font-size": 12.5,
        "font-weight": 600, "letter-spacing": "-0.02em", fill: col, text: b.factor }));
      svg.appendChild(sv("text", { x: M.l - 10, y: yy + 30, "text-anchor": "end", "font-size": 10.5,
        fill: "var(--ink-3)", text: "S1 " + f3(b.S1) + " · ST " + f3(b.ST) }));
      if (num(b.S1)) svg.appendChild(sv("rect", { x: x(Math.min(0, b.S1)), y: yy + 2,
        width: Math.abs(x(b.S1) - x(0)), height: 16, fill: col }, [sv("title", { text: "S1 = " + f3(b.S1) })]));
      if (num(b.ST)) svg.appendChild(sv("line", { x1: x(b.ST), x2: x(b.ST), y1: yy - 2, y2: yy + 22,
        stroke: "var(--ink)", "stroke-width": 3 }, [sv("title", { text: "ST = " + f3(b.ST) })]));
      if (num(b.gap_lo) && num(b.gap_hi)) {
        var cy = yy + 32, st = b.significant ? "var(--ink)" : "var(--ink-3)";
        svg.appendChild(sv("line", { x1: x(b.gap_lo), x2: x(b.gap_hi), y1: cy, y2: cy, stroke: st, "stroke-width": 1.5 }));
        [b.gap_lo, b.gap_hi].forEach(function (e) {
          svg.appendChild(sv("line", { x1: x(e), x2: x(e), y1: cy - 4, y2: cy + 4, stroke: st, "stroke-width": 1.5 }));
        });
        svg.appendChild(sv("circle", { cx: x(num(b.gap) ? b.gap : 0), cy: cy, r: 3.5, fill: st },
          [sv("title", { text: "ST−S1 = " + sign3(b.gap) + " [" + f3(b.gap_lo) + ", " + f3(b.gap_hi) + "]"
            + (b.significant ? " — clears the pre-registered threshold" : "") })]));
      }
    });
    return el("div", { className: "scroll-x" }, [svg]);
  }

  // ======================================================================
  // ACTIVE LEARNING (steppable)
  // ======================================================================

  function renderAL(host, d) {
    var frames = d.frames || [], curves = d.curves;

    if (frames.length) {
      STATE.alStep = Math.min(STATE.alStep, frames.length - 1);
      var bar = el("div", { className: "stepper" });
      var slider = el("input", { type: "range", min: 0, max: frames.length - 1, value: STATE.alStep,
        "aria-label": "active-learning iteration" });
      var cnt = el("span", { className: "cnt" });
      var post = el("div", {});
      function draw() {
        var fr = frames[STATE.alStep];
        cnt.textContent = "step " + STATE.alStep + "/" + (frames.length - 1) + " · " + fr.n_evals + " oracle calls";
        clear(post).appendChild(posteriorSvg(fr, d));
      }
      slider.addEventListener("input", function () { STATE.alStep = +slider.value; draw(); });
      bar.appendChild(el("button", { text: "back", onclick: function () {
        STATE.alStep = Math.max(0, STATE.alStep - 1); slider.value = STATE.alStep; draw(); } }));
      bar.appendChild(el("button", { text: "step", onclick: function () {
        STATE.alStep = Math.min(frames.length - 1, STATE.alStep + 1); slider.value = STATE.alStep; draw(); } }));
      bar.appendChild(slider);
      bar.appendChild(cnt);
      host.appendChild(bar);
      host.appendChild(post);
      draw();
    } else {
      host.appendChild(noticed("No GP posterior frames in this build",
        "The loop was not run for this arm — pass --al to build.py to regenerate them."));
    }

    if (curves && (curves.series || []).length) {
      host.appendChild(el("div", { style: "margin-top:22px" }, [alCurveSvg(curves)]));
      if (curves.headline) {
        host.appendChild(el("div", { className: "verdict unresolved" }, [
          el("span", { className: "st", text: "active vs random — the result" }),
          el("p", { text: curves.headline })
        ]));
      }
      if (curves.provenance && curves.provenance.statement) {
        host.appendChild(el("p", { className: "prov", text: curves.provenance.statement }));
      }
      if (curves.evidence_level === "INSUFFICIENT") {
        host.appendChild(el("div", { className: "caveat", text:
          "Evidence level INSUFFICIENT — too few seeds to support a savings claim. The curve is shown; the "
          + "number is not." }));
      }
    }
    if (d.oracle && d.oracle.provenance) {
      host.appendChild(el("p", { className: "prov", text: "Oracle: " + d.oracle.provenance
        + ", fitted on " + (d.oracle.n_train || "?") + " measured conditions and evaluated against "
        + (d.oracle.n_test || "?") + " held-out real measurements taken from the master table ("
        + (num(d.oracle.oracle_calls_to_build) ? d.oracle.oracle_calls_to_build : "?")
        + " fresh API calls were needed to build that test set)." }));
    }
  }

  function posteriorSvg(fr, d) {
    var nx = fr.nx, ny = fr.ny, mu = fr.mu;
    var cw = 46, chh = 32, M = { t: 12, r: 216, b: 42, l: 54 };
    var W = M.l + cw * nx + M.r, H = M.t + chh * ny + M.b;
    var svg = sv("svg", { className: "chart", viewBox: "0 0 " + W + " " + H,
      preserveAspectRatio: "xMinYMin meet", role: "img", style: "max-width:" + W + "px",
      "aria-label": "GP posterior mean WER over rt60 and SNR at iteration " + fr.step });

    for (var r = 0; r < ny; r++) {
      for (var c = 0; c < nx; c++) {
        var v = mu[r * nx + c];
        svg.appendChild(sv("rect", { x: M.l + c * cw, y: M.t + r * chh, width: cw, height: chh,
          fill: werColor(v) }, [sv("title", { text: "predicted WER " + f2(v) })]));
      }
    }
    svg.appendChild(sv("rect", { x: M.l, y: M.t, width: cw * nx, height: chh * ny, fill: "none", stroke: "var(--ink)" }));

    (fr.points || []).forEach(function (p) {
      var px = M.l + ((p.rt60 - d.x_domain[0]) / (d.x_domain[1] - d.x_domain[0])) * cw * nx;
      var py = M.t + (1 - (p.snr_db - d.y_domain[0]) / (d.y_domain[1] - d.y_domain[0])) * chh * ny;
      var isNew = p.new === true;
      svg.appendChild(sv("circle", { cx: px, cy: py, r: isNew ? 8 : 4,
        fill: isNew ? "var(--danger)" : "var(--paper)",
        stroke: isNew ? "var(--paper)" : "var(--ink)", "stroke-width": isNew ? 2.5 : 1.5 },
        [sv("title", { text: (isNew ? "chosen at this step" : "already evaluated")
          + "\nrt60 " + f2(p.rt60) + " · snr " + f2(p.snr_db) + " · WER " + f2(p.wer) })]));
    });

    ticks(d.y_domain[1], d.y_domain[0], ny - 1).forEach(function (v, i) {
      svg.appendChild(sv("text", { className: "tick", x: M.l - 8, y: M.t + i * chh + chh / 2 + 4,
        "text-anchor": "end", text: v.toFixed(0) }));
    });
    ticks(d.x_domain[0], d.x_domain[1], nx - 1).forEach(function (v, i) {
      svg.appendChild(sv("text", { className: "tick", x: M.l + i * cw + cw / 2, y: H - M.b + 16,
        "text-anchor": "middle", text: v.toFixed(2) }));
    });
    svg.appendChild(sv("text", { className: "axis-title", x: M.l + cw * nx / 2, y: H - 6,
      "text-anchor": "middle", "font-size": 11, text: "rt60 (s)" }));
    svg.appendChild(sv("text", { className: "axis-title", x: -(M.t + chh * ny / 2), y: 12,
      "text-anchor": "middle", transform: "rotate(-90)", "font-size": 11, text: "SNR (dB)" }));

    var lx = M.l + cw * nx + 18;
    svg.appendChild(sv("text", { x: lx, y: M.t + 14, "font-size": 11, "font-weight": 600,
      "letter-spacing": ".14em", fill: "var(--ink-3)", text: "GP POSTERIOR MEAN" }));
    [["already evaluated", "var(--paper)", "var(--ink)"],
     ["chosen at this step", "var(--danger)", "var(--paper)"]].forEach(function (L, i) {
      svg.appendChild(sv("circle", { cx: lx + 8, cy: M.t + 36 + i * 21, r: 5.5, fill: L[1], stroke: L[2], "stroke-width": 2 }));
      svg.appendChild(sv("text", { x: lx + 22, y: M.t + 40 + i * 21, "font-size": 11.5, text: L[0] }));
    });
    if (num(fr.chosen_wer)) {
      svg.appendChild(sv("text", { x: lx, y: M.t + 96, "font-size": 11.5, text: "oracle WER here " + f2(fr.chosen_wer) }));
    }
    if (num(d.threshold)) {
      svg.appendChild(sv("text", { x: lx, y: M.t + 116, "font-size": 11.5,
        text: "hunting WER = " + f2(d.threshold) }));
    }
    if (d.slice) {
      svg.appendChild(sv("text", { x: lx, y: M.t + 142, "font-size": 11, fill: "var(--ink-3)",
        text: "slice " + (d.slice.noise_type || "?") + " · " + (d.slice.codec || "?") }));
      svg.appendChild(sv("text", { x: lx, y: M.t + 158, "font-size": 11, fill: "var(--ink-3)",
        text: "rolloff " + (d.slice.mic_rolloff !== undefined ? d.slice.mic_rolloff : "?") }));
    }
    return el("div", { className: "scroll-x" }, [svg]);
  }

  var ARM_COLOR = { active_boundary: "var(--f-reverb)", active_uncertainty: "var(--f-mic)", random: "var(--ink-2)" };

  function alCurveSvg(c) {
    var series = c.series || [];
    var W = 880, H = 320, M = { t: 16, r: 200, b: 52, l: 66 };
    var allx = [], ally = [];
    series.forEach(function (s) {
      s.n_evals.forEach(function (v) { allx.push(v); });
      s.lo.concat(s.hi, s.median).forEach(function (v) { if (num(v)) ally.push(v); });
    });
    if (!allx.length || !ally.length) throw new Error("active-vs-random curves are empty");
    var ymax = snapUp(Math.max.apply(null, ally) * 1.06, 0.05);
    var x = scale(Math.min.apply(null, allx), Math.max.apply(null, allx), M.l, W - M.r);
    var y = scale(0, ymax, H - M.b, M.t);
    var svg = sv("svg", { className: "chart", viewBox: "0 0 " + W + " " + H,
      preserveAspectRatio: "xMidYMid meet", role: "img", style: "max-width:" + W + "px",
      "aria-label": "active versus random learning curves with a band across seeds" });

    niceTicks(0, ymax, 5).forEach(function (v) {
      svg.appendChild(sv("line", { className: "gridline", x1: M.l, x2: W - M.r, y1: y(v), y2: y(v) }));
      svg.appendChild(sv("text", { className: "tick", x: M.l - 8, y: y(v) + 4, "text-anchor": "end", text: v.toFixed(2) }));
    });
    series.forEach(function (s, i) {
      var col = ARM_COLOR[s.arm] || "var(--f-channel)";
      var band = "", back = "";
      s.n_evals.forEach(function (n, k) {
        if (!num(s.lo[k]) || !num(s.hi[k])) return;
        band += (band ? "L" : "M") + x(n) + "," + y(s.lo[k]);
      });
      for (var k = s.n_evals.length - 1; k >= 0; k--) {
        if (!num(s.hi[k])) continue;
        back += "L" + x(s.n_evals[k]) + "," + y(s.hi[k]);
      }
      if (band) svg.appendChild(sv("path", { d: band + back + "Z", fill: col, "fill-opacity": 0.13, stroke: "none" }));
      var line = "";
      s.n_evals.forEach(function (n, kk) {
        if (!num(s.median[kk])) return;
        line += (line ? "L" : "M") + x(n) + "," + y(s.median[kk]);
      });
      if (line) svg.appendChild(sv("path", { d: line, fill: "none", stroke: col, "stroke-width": 2.5 }));
      svg.appendChild(sv("text", { x: W - M.r + 14, y: M.t + 16 + i * 20, "font-size": 12,
        "font-weight": 600, fill: col, text: s.arm }));
    });
    if (num(c.target)) {
      svg.appendChild(sv("line", { x1: M.l, x2: W - M.r, y1: y(c.target), y2: y(c.target),
        stroke: "var(--ink)", "stroke-dasharray": "5 4" }));
      // Sits in the paper margin to the right of the plot, not on top of the
      // curves it is describing.
      svg.appendChild(sv("text", { x: W - M.r + 14, y: y(c.target) + 4, "font-size": 11,
        fill: "var(--ink)", text: "target " + f3(c.target) }));
    }
    niceTicks(Math.min.apply(null, allx), Math.max.apply(null, allx), 5).forEach(function (v) {
      svg.appendChild(sv("text", { className: "tick", x: x(v), y: H - M.b + 18, "text-anchor": "middle", text: v.toFixed(0) }));
    });
    svg.appendChild(sv("line", { className: "axis-line", x1: M.l, x2: W - M.r, y1: H - M.b, y2: H - M.b }));
    svg.appendChild(sv("text", { className: "axis-title", x: (M.l + W - M.r) / 2, y: H - 8,
      "text-anchor": "middle", text: "oracle evaluations" }));
    svg.appendChild(sv("text", { className: "axis-title", x: -(M.t + H - M.b) / 2, y: 16,
      "text-anchor": "middle", transform: "rotate(-90)", text: c.metric || "boundary RMSE" }));
    svg.appendChild(sv("text", { x: W - M.r + 14, y: M.t + 28 + series.length * 20, "font-size": 11,
      fill: "var(--ink-3)", text: "band = min-max across" }));
    svg.appendChild(sv("text", { x: W - M.r + 14, y: M.t + 44 + series.length * 20, "font-size": 11,
      fill: "var(--ink-3)", text: (c.n_seeds || "?") + " seeds, not a CI" }));
    return el("div", { className: "scroll-x" }, [svg]);
  }

  // ======================================================================
  // SIM VS REAL
  // ======================================================================

  function renderSim2Real(host, d) {
    var pts = (d.scatter || []).filter(function (p) { return num(p.wer_real) && num(p.wer_sim); });
    if (!pts.length) throw new Error("no measured/simulated condition pairs matched on measured RT60");
    var h = d.headline || {};

    var W = 620, H = 430, M = { t: 16, r: 20, b: 52, l: 60 };
    var lim = d.identity || [0, 1];
    var lo = Math.max(0, snapDown(Math.min(lim[0], 0), 0.1));
    var hi = Math.min(1.0, snapUp(lim[1] + 0.01, 0.1));
    var x = scale(lo, hi, M.l, W - M.r), y = scale(lo, hi, H - M.b, M.t);
    var svg = sv("svg", { className: "chart", viewBox: "0 0 " + W + " " + H,
      preserveAspectRatio: "xMidYMid meet", role: "img",
      "aria-label": "WER through measured impulse responses against WER through RT60-matched synthetic ones" });

    niceTicks(lo, hi, 5).forEach(function (v) {
      svg.appendChild(sv("line", { className: "gridline", x1: M.l, x2: W - M.r, y1: y(v), y2: y(v) }));
      svg.appendChild(sv("text", { className: "tick", x: M.l - 8, y: y(v) + 4, "text-anchor": "end", text: v.toFixed(1) }));
      svg.appendChild(sv("text", { className: "tick", x: x(v), y: H - M.b + 18, "text-anchor": "middle", text: v.toFixed(1) }));
    });
    svg.appendChild(sv("line", { x1: x(lo), y1: y(lo), x2: x(hi), y2: y(hi),
      stroke: "var(--ink)", "stroke-dasharray": "6 4", "stroke-width": 1.25 }));
    // below the identity line, where no point can sit: the simulator being
    // kinder than reality is exactly what pushes points to the other side.
    svg.appendChild(sv("text", { x: x(hi) - 6, y: y(hi) + 26, "text-anchor": "end", "font-size": 11,
      fill: "var(--ink-2)", text: "y = x, a perfect simulator" }));

    pts.forEach(function (p) {
      var dz = p.dead_zone_real || p.dead_zone_sim;
      svg.appendChild(sv("circle", { cx: x(p.wer_real), cy: y(p.wer_sim), r: dz ? 7 : 4.5,
        fill: werColor(p.wer_real),
        stroke: dz ? "var(--danger)" : "var(--ink-3)", "stroke-width": dz ? 2.5 : 1 },
        [sv("title", { text: p.condition + "\nmeasured " + f3(p.wer_real) + " · simulated " + f3(p.wer_sim)
          + "\ngap " + sign3(p.gap) + (dz ? "\nflagged a dead zone in at least one arm" : "") })]));
    });
    svg.appendChild(sv("line", { className: "axis-line", x1: M.l, x2: W - M.r, y1: H - M.b, y2: H - M.b }));
    svg.appendChild(sv("line", { className: "axis-line", x1: M.l, x2: M.l, y1: M.t, y2: H - M.b }));
    svg.appendChild(sv("text", { className: "axis-title", x: (M.l + W - M.r) / 2, y: H - 8,
      "text-anchor": "middle", text: "WER through MEASURED impulse responses" }));
    svg.appendChild(sv("text", { className: "axis-title", x: -(M.t + H - M.b) / 2, y: 16,
      "text-anchor": "middle", transform: "rotate(-90)", text: "WER through SIMULATED" }));

    var right = el("div", {}, [
      el("div", { className: "verdict " + s2rClass(h.verdict) }, [
        el("span", { className: "st", text: "sim vs real — " + (h.verdict || "see the numbers") }),
        el("p", { text: d.verdict_sentence || "Points below the dashed line mean the simulator is KINDER than "
          + "the measured rooms. The useful question is not whether the level matches but whether the ORDERING "
          + "does." })
      ]),
      statList([
        ["matched pairs", num(h.n_pairs) ? String(h.n_pairs) : "n/a", (h.n_clips || "?") + " clips per condition"],
        ["mean sim − real WER", sign3(h.mean_gap),
          (h.ci && num(h.ci[0])) ? "95% CI [" + f3(h.ci[0]) + ", " + f3(h.ci[1]) + "]" : ""],
        ["Spearman rho", f3(h.spearman), "does sim rank conditions like reality?"],
        ["dead-zone Jaccard", f2(h.dead_zone_jaccard), "does it find the same dangerous cells?"]
      ]),
      el("p", { className: "prov", text: "Pairs are matched on the MEASURED Schroeder RT60 of both files, never "
        + "on the value requested from the simulator — a simulator that misses its target would otherwise be "
        + "compared against the wrong room." })
    ]);

    host.appendChild(el("div", { className: "two-col" }, [el("div", { className: "scroll-x" }, [svg]), right]));
  }

  function s2rClass(v) {
    v = String(v || "");
    if (v.indexOf("SIM TRACKS") === 0) return "confirmed";
    if (v.indexOf("NOT PRESERVED") >= 0 || v.indexOf("INSUFFICIENT") >= 0) return "notconfirmed";
    return "unresolved";
  }

  /* label / value / sub-label, in the machine register. Values right, labels
     left, so a column of numbers stays scannable. */
  function statList(rows) {
    var t = el("div", { className: "tally", style: "margin-top:16px" });
    rows.forEach(function (r) {
      var v = String(r[1]);
      t.appendChild(el("div", { className: "row" }, [
        el("span", { className: "n" + (v.length > 8 ? " long" : ""), text: v }),
        el("span", { className: "k", text: r[0] + (r[2] ? " · " + r[2] : "") })
      ]));
    });
    return t;
  }

  // ======================================================================
  // L1 — MULTI-MODEL COMPARISON (outside the model toggle)
  // ======================================================================

  function renderModelArms(host, d) {
    var arms = d.models || [];
    if (arms.length < 2) throw new Error("fewer than two model arms are present in this build");
    var ov = d.overlap || {}, hall = d.hallucination || {};

    host.appendChild(el("div", { className: "verdict unresolved" }, [
      el("span", { className: "st", text: "confidence is not comparable across models" }),
      el("p", { text: d.caption || "" })
    ]));

    // Absolute WER is shown twice on purpose. The strict column is the spine
    // measurement; the cross-model column is what remains once both arms are
    // made to agree about orthography. Showing only the second would hide how
    // large the formatting artefact was.
    // WER COMPARABILITY. This table IS a cross-arm WER ranking, so the arms
    // excluded from that comparison must be marked here or the page quietly
    // invites the exact reading the analysis refuses. The dead-zone rate and
    // the confidence shape are NOT marked: those are within-model and every arm
    // is a first-class member of them.
    var wc = d.wer_comparability || {};
    var werOut = wc.excluded || [];
    var mark = function (a, s) { return a.wer_comparable === false ? s + " ‡" : s; };
    var head = ["model", "WER strict", "WER cross-model", "dead-zone rate", "sub", "del", "ins", "conf vs WER"];
    var rows = arms.map(function (a) {
      var e = a.edits || {};
      return [a.model, mark(a, f3(a.wer_strict)), mark(a, f3(a.wer_crossmodel)),
              pct(a.dead_zone_rate), mark(a, f3(e.sub)), mark(a, f3(e.del)),
              mark(a, f3(e.ins)), f2((a.shape || {}).spearman)];
    });
    host.appendChild(simpleTable(head, rows));
    if (werOut.length) {
      host.appendChild(el("p", { className: "note", text:
        "‡ " + werOut.join(", ") + " is EXCLUDED from cross-model WER "
        + "comparison: its orthography is non-deterministic across identical "
        + "calls, so its WER offset is a per-call draw and not a constant that "
        + "can be measured once and subtracted. Its WER and edit-composition "
        + "cells are its own internal scale — do not rank them against another "
        + "arm's, and in particular a high `sub` here is orthography, not "
        + "evidence that it substitutes more. Its dead-zone rate and "
        + "confidence-vs-WER shape are unmarked because they are computed "
        + "within the arm and are unaffected." }));
    }

    // Overlap over N arms. `jaccard` is the ALL-ARM value (identical to the
    // pairwise one when there are exactly two), so the label must say which,
    // or a three-arm number reads as a two-arm one.
    var nArms = arms.length;
    var stats = [
      [nArms > 2 ? "Jaccard (all " + nArms + " arms)" : "Jaccard", f2(ov.jaccard),
       nArms > 2 ? "cells every arm flags dangerous" : "same cells flagged dangerous?"],
      ["shared", String((ov.shared || []).length),
       nArms > 2 ? "dead zones present in EVERY arm" : "dead zones present in both arms"],
      ["hallucination rate", pct(hall.frac_rows_over_2x),
       "rows where hyp > 2x ref length" + (hall.model ? " (" + hall.model + ", worst arm)" : "")]
    ];
    var pw = ov.pairwise || {};
    if (nArms > 2) {
      // An all-arm Jaccard of 0 cannot distinguish "no two arms agree" from
      // "two agree perfectly and the third is disjoint". Name the pairs.
      Object.keys(pw).forEach(function (k) {
        stats.push([k.replace("|", " vs "), f2(pw[k].jaccard),
                    "shared " + pw[k].n_shared + " of " + pw[k].n_union]);
      });
    }

    host.appendChild(el("div", { className: "two-col", style: "margin-top:22px" }, [
      el("div", {}, [subhead(nArms > 2 ? "where the arms disagree most" : "where the two arms disagree most"),
                     divergenceList(d.divergence_regions || [])]),
      el("div", {}, [
        subhead("dead-zone overlap"),
        statList(stats),
        censusNote(d.census),
        normNote(d.normalization_shift)
      ])
    ]));

    // A fluent invented sentence is a DIFFERENT failure from acoustic
    // confusion, and an insertion count cannot tell them apart. One real
    // transcript makes the distinction in a way no summary statistic does.
    var ex = (hall.examples || [])[0];
    if (ex) {
      host.appendChild(sighead("exhibit — " + (hall.model || "the open baseline")
        + " invents fluent text under heavy degradation"));
      host.appendChild(el("div", { className: "exhibit" }, [
        el("p", { className: "meta", text: "clip " + ex.clip_id + " at " + ex.condition_name + " — "
          + ex.n_ref + " reference words came back as " + ex.n_hyp
          + (num(ex.len_ratio) ? " (" + f1(ex.len_ratio) + "x)" : "")
          + (num(ex.foreign_frac) ? ", " + pct(ex.foreign_frac) + " of them foreign to the reference" : "") }),
        el("p", { className: "line" }, [el("span", { className: "lab", text: "REF" }), el("span", { text: ex.reference })]),
        el("p", { className: "line" }, [el("span", { className: "lab", text: "HYP" }), el("span", { className: "bad", text: ex.transcript })])
      ]));
    }
  }

  function divergenceList(regions) {
    if (!regions.length) return noticed("No divergence regions", "The arms did not separate on any single factor slice.");
    var ul = el("ul", { className: "notes" });
    regions.slice(0, 5).forEach(function (r) {
      var span = Array.isArray(r.span) ? r.span.map(tidy).join(" to ") : String(r.span);
      var by = r.wer_by_model || {};
      ul.appendChild(el("li", { text: r.factor + " = " + span + "   WER gap " + f2(r.wer_gap) + "   ("
        + Object.keys(by).map(function (m) { return m + " " + f2(by[m]); }).join("  vs  ") + ")" }));
    });
    return ul;
  }

  // The arms are intersected to the cells EVERY arm ran, and that intersection
  // shrinks as arms are added. A page that shows only the resulting numbers
  // cannot show that they narrowed, so the census travels with them.
  function censusNote(cen) {
    if (!cen) return null;
    var bits = "compared over " + cen.n_common_cells + " cells ("
      + cen.n_common_clips + " clips x " + cen.n_common_conditions + " conditions) that all "
      + cen.n_arms + " arms ran";
    if (!cen.cells_matched) {
      bits += ". The arms ran different cell sets, so "
        + cen.n_rows_dropped_by_intersection
        + " rows fall outside the comparison — these are not each arm's corpus-wide numbers.";
    }
    if ((cen.arms_excluded || []).length) {
      bits += " Arms present in the table but NOT compared: " + cen.arms_excluded.join(", ") + ".";
    }
    return el("p", { className: "prov", text: bits });
  }

  function normNote(sh) {
    if (!sh) return null;
    var bits = Object.keys(sh).map(function (m) { return m + " " + sign3(-sh[m].mean_shift); });
    if (!bits.length) return null;
    return el("p", { className: "prov", text: "Cross-model normalization moved each arm by " + bits.join(" · ")
      + ". The arm that already writes numbers as words should barely move; a large shift there would mean the "
      + "normalizer is changing more than orthography." });
  }

  // ======================================================================
  // L3 — PARALINGUISTIC DECOUPLING (outside the model toggle)
  // ======================================================================

  function renderDecoupling(host, d) {
    var factors = d.factors || [];
    if (!factors.length) throw new Error("no swept factors in the decoupling result");

    // Six sweeps, six verdicts, readable in one pass before any chart.
    var tbl = el("div", { className: "sweeps" });
    tbl.appendChild(el("div", { className: "sweep-row head" }, [
      el("span", { text: "sweep" }), el("span", { text: "verdict" }), el("span", { text: "what it means" })
    ]));
    factors.forEach(function (f) {
      var on = /DECOUPLED/.test(String(f.verdict || ""));
      tbl.appendChild(el("div", { className: "sweep-row" }, [
        el("span", { className: "fac", text: f.factor }),
        el("span", { className: "vd " + (on ? "on" : "off"), text: String(f.verdict || "—").split(" — ")[0] }),
        el("span", { className: "stmt", text: f.statement || "" })
      ]));
    });
    host.appendChild(tbl);

    factors.forEach(function (f) {
      host.appendChild(sighead("sweep · " + f.factor + " · " + (f.n_clips || "?") + " clips · "
        + (f.severity_order || "")));
      var deg = f.degeneracy || {};
      var stats = [
        ["features trending reliably", (f.n_features_trend_reliable != null ? f.n_features_trend_reliable : "?")
          + " of " + (f.n_features != null ? f.n_features : "?"), "spearman against severity rank"],
        ["which stream leads", String(f.leads || "n/a"), ""],
        ["spearman(drift, WER)", f2(f.spearman), ""]
      ];
      /* Half-degradation levels are deliberately NOT shown as numbers when the
         analysis refused to quote them: a crossing point read off a curve that
         never travelled is arithmetic, and a stat box would launder it into a
         finding. */
      if (f.half_levels_quotable) {
        stats.push(["half-degradation, feature / lexical",
          f2(f.feature_half_level) + " / " + f2(f.lexical_half_level), "in the factor's own units"]);
      } else {
        stats.push(["half-degradation", "not quotable",
          "WER range " + f3(deg.range) + " below the required " + f3(deg.min_range_required)]);
      }
      var chart = decouplingChart(f), fd = featureDrift(f);
      host.appendChild(el("div", { className: "two-col" }, [
        chart ? el("div", {}, [
          el("div", { className: "scroll-x" }, [chart]),
          el("div", { className: "chart-key" }, [
            el("span", { className: "lex" }, [el("i", {}), el("span", { text: "WER — absolute 0 to 1 scale, never auto-scaled" })]),
            el("span", { className: "fea" }, [
              el("i", { className: fd && !fd.reliable ? "dash" : "" }),
              el("span", { text: (f.primary_feature || "feature") + " drift, scaled to its own max"
                + (fd && !fd.reliable ? " — dashed: does not trend with severity" : "") })
            ])
          ])
        ]) : noticed("No curve for this sweep", "The level ladder and the WER series did not line up."),
        el("div", {}, [statList(stats), deg.degenerate && deg.why
          ? el("p", { className: "prov", text: "Why no threshold: " + deg.why }) : null])
      ]));
    });

    if (d.caption) host.appendChild(el("p", { className: "prov", text: d.caption }));
    var notes = (factors[0] || {}).notes || [];
    if (notes.length) {
      host.appendChild(sighead("how these curves were measured"));
      var ul = el("ul", { className: "sigs" });
      notes.forEach(function (n) { ul.appendChild(el("li", { text: n })); });
      host.appendChild(ul);
    }
  }

  /* The feature curve is the primary feature's drift away from its own clean
     value, scaled to its own maximum excursion. It is drawn DASHED when the
     analysis judged that the feature does not trend reliably with severity —
     an unreadable curve must LOOK unreadable. */
  function featureDrift(f) {
    var pf = (f.per_feature || {})[f.primary_feature];
    if (Array.isArray(f.feature_curve) && f.feature_curve.length) {
      var mx0 = Math.max.apply(null, f.feature_curve.map(Math.abs)) || 1;
      return { curve: f.feature_curve.map(function (v) { return num(v) ? Math.abs(v) / mx0 : null; }),
               reliable: pf ? pf.trend_reliable !== false : true };
    }
    if (!pf || !Array.isArray(pf.drift_curve_mean)) return null;
    var clean = num(pf.clean_value_mean) ? pf.clean_value_mean : 0;
    var raw = pf.drift_curve_mean.map(function (v) { return num(v) ? Math.abs(v - clean) : null; });
    var mx = Math.max.apply(null, raw.filter(num)) || 1;
    return { curve: raw.map(function (v) { return num(v) ? v / mx : null; }),
             reliable: pf.trend_reliable !== false };
  }

  function decouplingChart(f) {
    var lv = f.levels || [], wer = f.wer_mean || [];
    if (lv.length < 2 || wer.length !== lv.length) return null;
    var W = 620, H = 280, M = { t: 18, r: 20, b: 48, l: 56 };
    var lo = Math.min.apply(null, lv), hi = Math.max.apply(null, lv);
    var x = scale(lo, hi, M.l, W - M.r), y = scale(0, 1, H - M.b, M.t);
    var svg = sv("svg", { className: "chart", viewBox: "0 0 " + W + " " + H,
      preserveAspectRatio: "xMidYMid meet", role: "img",
      "aria-label": "lexical accuracy and paralinguistic drift over the " + f.factor + " sweep" });

    ticks(0, 1, 5).forEach(function (v) {
      svg.appendChild(sv("line", { className: "gridline", x1: M.l, x2: W - M.r, y1: y(v), y2: y(v) }));
      svg.appendChild(sv("text", { className: "tick", x: M.l - 8, y: y(v) + 4, "text-anchor": "end", text: v.toFixed(1) }));
    });
    niceTicks(lo, hi, 5).forEach(function (v) {
      svg.appendChild(sv("text", { className: "tick", x: x(v), y: H - M.b + 18, "text-anchor": "middle",
        text: Math.abs(v) >= 10 ? v.toFixed(0) : v.toFixed(2) }));
    });

    /* WER is drawn against a FIXED 0..1 axis, never auto-scaled. That is the
       degeneracy guard made visual: a flat curve must LOOK flat, and
       auto-scaling would stretch a 0.05 wander to fill the panel. */
    svg.appendChild(polyline(lv, wer, x, y, "var(--wer-5)", false));
    var fd = featureDrift(f);
    if (fd && fd.curve.length === lv.length) {
      svg.appendChild(polyline(lv, fd.curve, x, y, "var(--f-noise)", !fd.reliable));
    }

    svg.appendChild(sv("line", { className: "axis-line", x1: M.l, x2: W - M.r, y1: H - M.b, y2: H - M.b }));
    svg.appendChild(sv("line", { className: "axis-line", x1: M.l, x2: M.l, y1: M.t, y2: H - M.b }));
    svg.appendChild(sv("text", { className: "axis-title", x: (M.l + W - M.r) / 2, y: H - 8,
      "text-anchor": "middle", text: f.factor + "  ·  " + (f.severity_order || "") }));
    return svg;
  }

  function polyline(xs, ys, x, y, colour, dashed) {
    var pts = [];
    for (var i = 0; i < xs.length; i++) {
      if (!num(ys[i])) continue;
      pts.push(x(xs[i]).toFixed(1) + "," + y(Math.max(0, Math.min(1, ys[i]))).toFixed(1));
    }
    return sv("polyline", { points: pts.join(" "), fill: "none", stroke: colour,
      "stroke-width": 2.5, "stroke-linejoin": "round", "stroke-dasharray": dashed ? "6 5" : null });
  }

  // ---------------------------------------------------------------- shared
  function simpleTable(head, rows) {
    var thead = el("tr", {}, head.map(function (h, i) {
      return el("th", { className: i ? "num" : "", text: h });
    }));
    var body = rows.map(function (r) {
      return el("tr", {}, r.map(function (c, i) {
        return el("td", { className: i ? "num" : "name", text: String(c) });
      }));
    });
    return el("div", { className: "scroll-x" }, [
      el("table", { className: "grid" }, [el("thead", {}, [thead]), el("tbody", {}, body)])
    ]);
  }

  // ======================================================================
  // SHELL
  // ======================================================================

  function renderHeader() {
    var meta = DATA.meta || {};
    var badge = document.getElementById("source-badge");
    if (badge) {
      badge.className = "badge " + (meta.is_synthetic ? "synthetic" : "real");
      badge.textContent = (meta.is_synthetic ? "synthetic data" : "real grid")
        + " · " + (meta.n_rows || 0) + " rows · " + (meta.n_conditions || 0) + " conditions";
      badge.setAttribute("title", "source: " + (meta.source || "unknown"));
    }
    var built = document.getElementById("built-at");
    if (built) built.textContent = "built " + (meta.generated || "?") + " from " + (meta.source || "?");

    var host = document.getElementById("model-toggle");
    if (!host) return;
    clear(host);
    var models = Object.keys(DATA.models || {});
    host.appendChild(el("span", { className: "lbl", text: "arm" }));
    if (!models.length) {
      host.appendChild(el("button", { text: "no models in table", disabled: "disabled" }));
      return;
    }
    models.forEach(function (m) {
      host.appendChild(el("button", {
        text: m, "aria-pressed": String(m === STATE.model),
        onclick: function () {
          STATE.model = m; STATE.pinned = null; STATE.focus = null; STATE.alStep = 0; renderAll();
        }
      }));
    });
  }

  function renderAll() {
    renderHeader();
    var m = (DATA.models || {})[STATE.model] || {};
    var missing = { status: "missing", reason: "No model is selected, or the master table held no rows for "
      + "this model. Check the model column of the source table." };

    panel("panel-hero", m.silent_failure || missing, renderHero, "The silent-failure map is not available");
    panel("panel-heatmap", m.silent_failure || missing, renderHeatmap, "The factor-space grid is not available");
    panel("panel-fingerprints", m.fingerprints || missing, renderFingerprints, "Failure fingerprints are not available");
    panel("panel-sensitivity", m.sensitivity || missing, renderSensitivity, "The sensitivity panel is not available");
    panel("panel-al", m.active_learning || missing, renderAL, "The active-learning panel is not available");
    panel("panel-sim2real", m.sim2real || missing, renderSim2Real, "The sim-vs-real panel is not available");

    // Panels 7 and 8 sit OUTSIDE the model toggle: panel 7 IS the comparison
    // between arms, and panel 8 reads audio rather than the per-model table.
    var cross = DATA.cross || {};
    var noCross = { status: "missing", reason: "This build produced no cross-model section." };
    panel("panel-model-arms", cross.model_arms || noCross, renderModelArms, "The multi-model comparison is not available");
    panel("panel-decoupling", cross.decoupling || noCross, renderDecoupling, "The paralinguistic decoupling panel is not available");

    var notes = document.getElementById("build-notes");
    if (notes) {
      clear(notes);
      ((DATA.meta || {}).notes || []).forEach(function (n) { notes.appendChild(el("li", { text: n })); });
    }
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", renderAll);
  else renderAll();

  window.DEADZONE = { loadData: loadData, state: STATE, render: renderAll };
})();
