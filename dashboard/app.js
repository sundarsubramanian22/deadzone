/* ==========================================================================
   Deadzone dashboard — vanilla JS, hand-rolled SVG. No framework, no network.

   ONE data door: loadData(). Everything on this page is read through it and
   nothing else, so swapping the synthetic table for the real one is a one-line
   change in build.py — the front end never knows which it got.

   Every panel is defensive. A payload arrives as {status, reason, data}: if a
   payload is missing, its panel renders a labelled empty state saying WHY, and
   the rest of the dashboard carries on. There is no code path that leaves a
   blank chart with no explanation.
   ========================================================================== */
(function () {
  "use strict";

  // ---------------------------------------------------------------- the door
  function loadData() {
    return window.__DEADZONE_DATA__ || { meta: {}, models: {}, default_model: null };
  }

  var DATA = loadData();
  var STATE = { model: DATA.default_model || Object.keys(DATA.models || {})[0] || null,
                pinned: null, alStep: 0 };

  // ------------------------------------------------------------- tiny helpers
  var NS = "http://www.w3.org/2000/svg";

  function el(tag, attrs, kids) {
    var n = document.createElement(tag);
    apply(n, attrs, kids);
    return n;
  }
  function sv(tag, attrs, kids) {
    var n = document.createElementNS(NS, tag);
    apply(n, attrs, kids, true);
    return n;
  }
  function apply(n, attrs, kids, isSvg) {
    attrs = attrs || {};
    for (var k in attrs) {
      var v = attrs[k];
      if (v === null || v === undefined) continue;
      if (k === "text") { n.textContent = String(v); }
      else if (k === "html") { n.innerHTML = v; }
      else if (k.slice(0, 2) === "on" && typeof v === "function") {
        n.addEventListener(k.slice(2), v);
      } else if (!isSvg && (k === "className")) { n.className = v; }
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
  function f2(v) { return num(v) ? v.toFixed(2) : "n/a"; }
  function f3(v) { return num(v) ? v.toFixed(3) : "n/a"; }
  function pct(v) { return num(v) ? (100 * v).toFixed(0) + "%" : "n/a"; }
  function sign3(v) { return num(v) ? (v >= 0 ? "+" : "") + v.toFixed(3) : "n/a"; }
  function css(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  /* WER -> the ONE sequential ramp. Six stops, fixed domain [0, 1] so the colour
     of a cell means the same thing in every panel and between models. */
  var WER_STOPS = [0.0, 0.10, 0.22, 0.38, 0.58, 0.80];
  function werColor(v) {
    if (!num(v)) return "var(--surface-2)";
    var i = 0;
    for (var k = 0; k < WER_STOPS.length; k++) if (v >= WER_STOPS[k]) i = k;
    return "var(--wer-" + i + ")";
  }
  function werInk(v) { return num(v) && v >= 0.38 ? "#fff" : "var(--ink)"; }

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

  // --------------------------------------------------------- the empty state
  function emptyState(host, title, reason) {
    clear(host).appendChild(el("div", { className: "empty" }, [
      el("div", { className: "t", text: title }),
      el("div", { className: "r", text: reason || "no reason recorded" })
    ]));
  }

  /* Every panel goes through here: it either has a payload or it explains
     itself. `render` is only ever called with a real `data`. */
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
  // PANEL 1 — SILENT-FAILURE MAP (hero)
  // ======================================================================

  var FACTOR_KEYS = ["rt60", "snr_db", "noise_type", "codec", "mic_rolloff"];

  function renderHero(host, d) {
    var pts = (d.points || []).filter(function (p) { return num(p.x_mean_conf) && num(p.y_wer); });
    if (!pts.length) throw new Error("no condition has both a confidence and a WER");

    var q = d.quadrant || {};
    var counts = d.quadrant_counts || {};

    var strip = el("div", { className: "quad-strip" }, [
      quadBox("dead zone: confident AND wrong", counts.dead_zone_confident_and_wrong, true),
      quadBox("confident and right", counts.confident_and_right),
      quadBox("loud failure: unconfident and wrong", counts.loud_failure_unconfident_and_wrong),
      quadBox("unconfident but right", counts.unconfident_but_right),
      quadBox("unplaceable (no confidence)", counts.unplaceable_no_confidence)
    ]);

    var grid = el("div", { className: "hero-grid" });
    var left = el("div", {});
    var detail = el("div", { className: "detail" });
    grid.appendChild(left); grid.appendChild(detail);

    // ---- the scatter
    var W = 760, H = 470, M = { t: 16, r: 18, b: 46, l: 58 };
    var xs = extent(pts, function (p) { return p.x_mean_conf; });
    var ys = extent(pts, function (p) { return p.y_wer; });
    var xlo = Math.max(0, Math.min(xs[0], num(q.conf_hi_raw) ? q.conf_hi_raw : 1) - 0.05);
    var xhi = Math.min(1.0, Math.max(xs[1], 0.6) + 0.03);
    var yhi = Math.max(ys[1], num(q.wer_hi) ? q.wer_hi + 0.1 : 0.5) * 1.04;
    var x = scale(xlo, xhi, M.l, W - M.r);
    var y = scale(0, yhi, H - M.b, M.t);

    var svg = sv("svg", { className: "chart", viewBox: "0 0 " + W + " " + H,
                          preserveAspectRatio: "xMidYMid meet",
                          role: "img", "aria-label": "confidence versus word error rate, one point per condition" });

    // dead-zone quadrant — the ONLY place the danger accent appears in this chart
    if (num(q.conf_hi_raw) && num(q.wer_hi)) {
      svg.appendChild(sv("rect", {
        x: x(q.conf_hi_raw), y: y(yhi), width: Math.max(0, x(xhi) - x(q.conf_hi_raw)),
        height: Math.max(0, y(q.wer_hi) - y(yhi)),
        fill: "var(--danger-soft)", stroke: "var(--danger)",
        "stroke-width": 1.5, "stroke-dasharray": "5 4"
      }));
      svg.appendChild(sv("text", {
        x: x(q.conf_hi_raw) + 9, y: y(yhi) + 19, "font-size": 13, "font-weight": 700,
        fill: "var(--danger)", text: "DEAD ZONE — confident and wrong"
      }));
      svg.appendChild(sv("text", {
        x: x(q.conf_hi_raw) + 9, y: y(yhi) + 36, "font-size": 11.5, fill: "var(--danger)",
        text: "conf ≥ " + f2(q.conf_hi_raw) + " (top " + pct(1 - (q.conf_pct_hi || 0))
              + " within model) and WER ≥ " + f2(q.wer_hi)
      }));
    }

    // axes
    var tx = ticks(xlo, xhi, 6), ty = ticks(0, yhi, 6);
    ty.forEach(function (v) {
      svg.appendChild(sv("line", { className: "gridline", x1: M.l, x2: W - M.r, y1: y(v), y2: y(v) }));
      svg.appendChild(sv("text", { className: "tick", x: M.l - 8, y: y(v) + 4, "text-anchor": "end", text: v.toFixed(2) }));
    });
    tx.forEach(function (v) {
      svg.appendChild(sv("text", { className: "tick", x: x(v), y: H - M.b + 18, "text-anchor": "middle", text: v.toFixed(2) }));
    });
    svg.appendChild(sv("line", { className: "axis-line", x1: M.l, x2: W - M.r, y1: H - M.b, y2: H - M.b }));
    svg.appendChild(sv("line", { className: "axis-line", x1: M.l, x2: M.l, y1: M.t, y2: H - M.b }));
    svg.appendChild(sv("text", { className: "axis-title", x: (M.l + W - M.r) / 2, y: H - 8,
      "text-anchor": "middle", text: (d.axes && d.axes.x) || "mean word confidence" }));
    svg.appendChild(sv("text", { className: "axis-title", x: -(M.t + H - M.b) / 2, y: 15,
      "text-anchor": "middle", transform: "rotate(-90)", text: (d.axes && d.axes.y) || "WER" }));

    // points — dead zones drawn last so they sit on top
    pts.slice().sort(function (a, b) { return (a.dead_zone ? 1 : 0) - (b.dead_zone ? 1 : 0); })
      .forEach(function (p) {
        var c = sv("circle", {
          cx: x(p.x_mean_conf), cy: y(p.y_wer), r: p.dead_zone ? 7 : 5.5,
          fill: werColor(p.y_wer),
          stroke: p.dead_zone ? "var(--danger)" : "var(--rule-strong)",
          "stroke-width": p.dead_zone ? 2.5 : 1,
          tabindex: 0, role: "button",
          "aria-label": p.condition_name + ", WER " + f2(p.y_wer) + ", confidence " + f2(p.x_mean_conf)
        });
        c.style.cursor = "pointer";
        function show() { if (!STATE.pinned) setDetail(detail, p, d); }
        c.addEventListener("mouseenter", show);
        c.addEventListener("focus", show);
        c.addEventListener("click", function () {
          STATE.pinned = (STATE.pinned === p.condition_name) ? null : p.condition_name;
          setDetail(detail, p, d);
        });
        svg.appendChild(c);
      });

    left.appendChild(svg);
    left.appendChild(el("div", { className: "legend-inline", text:
      "Fill = WER (shared ramp). Ring = dead zone. Hover or tab a point for its factor values and "
      + "the actual transcript diff; click to pin it." }));
    var rho = d.correlation && (num(d.correlation.spearman_confpct_vs_wer)
      ? d.correlation.spearman_confpct_vs_wer : d.correlation.spearman);
    if (num(rho)) {
      left.appendChild(el("div", { className: "note", text:
        "Confidence vs WER across conditions: Spearman rho = " + f2(rho)
        + (d.correlation.verdict ? " (" + d.correlation.verdict + ")" : "")
        + (num(d.n_failed_rows_excluded) ? " · " + d.n_failed_rows_excluded
           + " failed rows excluded, never averaged in" : "") }));
    }

    setDetailPlaceholder(detail);
    host.appendChild(strip);
    host.appendChild(grid);
  }

  function quadBox(label, n, danger) {
    return el("div", { className: "quad" + (danger ? " dz" : "") }, [
      el("div", { className: "n", text: num(n) ? String(n) : "—" }),
      el("div", { className: "k", text: label })
    ]);
  }

  function setDetailPlaceholder(host) {
    clear(host).appendChild(el("p", { className: "hint", text:
      "Hover a point to inspect the condition: its factor values, its confidence–WER gap, "
      + "and the word-level diff of one representative clip. The red-ringed points in the shaded "
      + "box are the ones this instrument exists to find." }));
  }

  function setDetail(host, p, d) {
    clear(host);
    host.className = "detail" + (p.dead_zone ? " dz" : "");
    host.appendChild(el("div", { className: "cname", text: p.condition_name }));
    if (p.dead_zone) host.appendChild(el("span", { className: "dzflag", text: "DEAD ZONE" }));

    var chips = el("div", { className: "chips" });
    FACTOR_KEYS.forEach(function (k, i) {
      if (p[k] === undefined || p[k] === null || p[k] === "") return;
      var v = typeof p[k] === "number" ? (Math.round(p[k] * 1000) / 1000) : p[k];
      chips.appendChild(el("span", { className: "chip f" + i, text: k + " " + v }));
    });
    host.appendChild(chips);

    var kv = el("dl", { className: "kv" });
    [["WER", f3(p.y_wer)],
     ["mean confidence", f3(p.x_mean_conf)],
     ["confidence pctile", num(p.conf_pct) ? pct(p.conf_pct) : "n/a"],
     ["confidence–accuracy gap", sign3(p.gap)],
     ["clips in cell", num(p.n_clips) ? String(p.n_clips) : "n/a"]
    ].forEach(function (r) {
      kv.appendChild(el("dt", { text: r[0] }));
      kv.appendChild(el("dd", { text: r[1] }));
    });
    host.appendChild(kv);

    if (p.label) host.appendChild(el("p", { className: "prov", text: p.label }));

    var ex = p.example;
    if (ex && ex.edits && ex.edits.length) {
      host.appendChild(el("div", { className: "k", style: "font-size:11.5px;color:var(--ink-3);text-transform:uppercase;letter-spacing:.06em;margin-top:6px",
        text: "transcript diff · clip " + (ex.clip_id || "?") + " · WER " + f2(ex.wer) }));
      host.appendChild(renderDiff(ex.edits));
      host.appendChild(el("div", { className: "legend-inline", text:
        "grey = correct · struck = deleted · red = substituted (ref→hyp) · underlined = inserted" }));
    } else {
      host.appendChild(el("p", { className: "hint", text:
        "No stored edit list for this condition, so no diff to show." }));
    }
  }

  function renderDiff(edits) {
    var box = el("div", { className: "diff" });
    edits.forEach(function (e) {
      var op = e[0], ref = e[1], hyp = e[2];
      if (op === "match") box.appendChild(el("span", { className: "m", text: ref + " " }));
      else if (op === "sub") box.appendChild(el("span", { className: "s", text: ref + "→" + hyp })), box.appendChild(document.createTextNode(" "));
      else if (op === "del") box.appendChild(el("span", { className: "d", text: ref })), box.appendChild(document.createTextNode(" "));
      else if (op === "ins") box.appendChild(el("span", { className: "i", text: hyp })), box.appendChild(document.createTextNode(" "));
    });
    return box;
  }

  function ticks(lo, hi, n) {
    var out = [], step = (hi - lo) / n;
    for (var i = 0; i <= n; i++) out.push(lo + i * step);
    return out;
  }

  // ======================================================================
  // PANEL 2 — FACTOR-SPACE HEATMAP (derived from the SAME hero points)
  // ======================================================================

  function renderHeatmap(host, d) {
    // Derived, not re-analysed: the heatmap and the hero scatter can never
    // disagree about a cell because they are the same list of conditions.
    var pts = (d.points || []).filter(function (p) { return num(p.y_wer); });
    if (!pts.length) throw new Error("no conditions with a WER to lay out");

    var facets = {};
    pts.forEach(function (p) {
      var key = (p.noise_type || "?") + " · codec " + (p.codec || "none");
      (facets[key] = facets[key] || []).push(p);
    });
    var keys = Object.keys(facets).filter(function (k) { return facets[k].length >= 2; }).sort();
    if (!keys.length) throw new Error("not enough cells per (noise_type, codec) facet to draw a grid");

    // The gap encoding is normalised across the WHOLE grid, not per facet, so a
    // thick border means the same thing in every panel of the small multiple.
    var gaps = pts.map(function (p) { return num(p.gap) ? p.gap : 0; });
    var gmin = Math.min.apply(null, gaps), gmax = Math.max.apply(null, gaps);
    var gspan = (gmax - gmin) || 1;

    var wrap = el("div", { className: "facets" });
    keys.forEach(function (k) { wrap.appendChild(facetSvg(k, facets[k], gmin, gspan)); });
    host.appendChild(wrap);

    var ramp = el("div", { className: "ramp" }, [
      el("span", { text: "WER" }),
      el("span", { className: "sw" }),
      el("span", { text: "low → high" }),
      el("span", { className: "dzk" }, [el("i", {}), el("span", { text:
        "border weight = confidence–accuracy gap (thicker = more overconfident); red = dead zone" })])
    ]);
    var sw = ramp.querySelector(".sw");
    for (var i = 0; i < 6; i++) sw.appendChild(el("i", { style: "background:var(--wer-" + i + ")" }));
    host.appendChild(ramp);
  }

  function facetSvg(title, cells, gmin, gspan) {
    var xsv = uniq(cells.map(function (c) { return c.rt60; })).sort(function (a, b) { return a - b; });
    var ysv = uniq(cells.map(function (c) { return c.snr_db; })).sort(function (a, b) { return b - a; });
    var cw = 62, ch = 40, M = { t: 8, r: 8, b: 34, l: 46 };
    var W = M.l + cw * xsv.length + M.r, H = M.t + ch * ysv.length + M.b;
    var svg = sv("svg", { className: "chart", viewBox: "0 0 " + W + " " + H,
                          preserveAspectRatio: "xMinYMin meet",
                          role: "img", "aria-label": "WER by rt60 and SNR for " + title });

    var byKey = {};
    cells.forEach(function (c) { byKey[c.rt60 + "|" + c.snr_db] = c; });

    ysv.forEach(function (snr, r) {
      svg.appendChild(sv("text", { className: "tick", x: M.l - 7, y: M.t + r * ch + ch / 2 + 4,
        "text-anchor": "end", text: String(snr) }));
      xsv.forEach(function (rt, c) {
        var cell = byKey[rt + "|" + snr];
        var gx = M.l + c * cw, gy = M.t + r * ch;
        if (!cell) {
          svg.appendChild(sv("rect", { x: gx + 1, y: gy + 1, width: cw - 2, height: ch - 2,
            fill: "var(--surface-2)", stroke: "var(--rule)", "stroke-dasharray": "3 3" }));
          svg.appendChild(sv("text", { x: gx + cw / 2, y: gy + ch / 2 + 4, "text-anchor": "middle",
            "font-size": 11, fill: "var(--ink-3)", text: "–" }));
          return;
        }
        // Second encoding: border WEIGHT = confidence–accuracy gap, normalised
        // over the grid. Colour is not used for it — the danger accent belongs to
        // dead zones alone, and a page where everything is red says nothing.
        var g = num(cell.gap) ? cell.gap : gmin;
        var bw = 1 + 3.5 * Math.max(0, Math.min(1, (g - gmin) / gspan));
        svg.appendChild(sv("rect", {
          x: gx + 2, y: gy + 2, width: cw - 4, height: ch - 4, rx: 3,
          fill: werColor(cell.y_wer),
          stroke: cell.dead_zone ? "var(--danger)" : "var(--ink-3)",
          "stroke-width": bw
        }));
        var t = sv("text", { x: gx + cw / 2, y: gy + ch / 2 + 4, "text-anchor": "middle",
          "font-size": 12, "font-weight": 600, fill: werInk(cell.y_wer), text: f2(cell.y_wer) });
        svg.appendChild(t);
        var tip = sv("title", { text: cell.condition_name + "\nWER " + f3(cell.y_wer)
          + "\nmean conf " + f3(cell.x_mean_conf) + "\ngap " + sign3(cell.gap)
          + (cell.dead_zone ? "\nDEAD ZONE" : "") });
        svg.appendChild(tip);
      });
    });
    xsv.forEach(function (rt, c) {
      svg.appendChild(sv("text", { className: "tick", x: M.l + c * cw + cw / 2, y: H - M.b + 16,
        "text-anchor": "middle", text: String(rt) }));
    });
    svg.appendChild(sv("text", { className: "axis-title", x: M.l + (W - M.l - M.r) / 2, y: H - 6,
      "text-anchor": "middle", "font-size": 11.5, text: "rt60 (s)" }));
    svg.appendChild(sv("text", { className: "axis-title", x: -(M.t + (H - M.t - M.b) / 2), y: 11,
      "text-anchor": "middle", transform: "rotate(-90)", "font-size": 11.5, text: "SNR (dB)" }));

    return el("div", { className: "facet" }, [
      el("h3", { text: title }),
      el("div", { className: "scroll-x" }, [svg])
    ]);
  }

  // ======================================================================
  // PANEL 3 — FINGERPRINTS
  // ======================================================================

  function renderFingerprints(host, d) {
    var fams = d.families || [];
    if (!fams.length) throw new Error("no factor families had enough rows to profile");

    host.appendChild(el("div", { className: "edit-key" }, [
      keyItem("substitutions", "var(--cat-5)"),
      keyItem("deletions", "var(--cat-1)"),
      keyItem("insertions", "var(--cat-4)"),
      el("span", {}, [el("i", { style: "background:transparent;border-left:3px solid var(--danger);border-radius:0" }),
                      el("span", { text: "entity error rate" })])
    ]));

    var maxRate = Math.max(0.35, Math.max.apply(null, fams.map(function (f) {
      return Math.max(f.wer || 0, f.entity_error_rate || 0);
    })));

    var tbl = el("table", { className: "grid" });
    tbl.appendChild(el("thead", {}, [el("tr", {}, [
      el("th", { text: "condition family" }),
      el("th", { text: "edit composition (share of reference words)" }),
      el("th", { text: "dominant" }),
      el("th", { className: "num", text: "WER" }),
      el("th", { className: "num", text: "entity ER" }),
      el("th", { text: "implied fix" })
    ])]));
    var tb = el("tbody", {});
    fams.forEach(function (f) {
      tb.appendChild(el("tr", {}, [
        el("td", {}, [el("code", { style: "font:600 12.5px var(--mono)", text: f.family }),
                      f.n_conditions ? el("div", { style: "font-size:11.5px;color:var(--ink-3)",
                        text: f.n_conditions + " conditions · " + f.n_ref + " ref words" }) : null]),
        el("td", {}, [stackedBar(f, maxRate)]),
        el("td", {}, [el("span", { className: "dom " + (f.dominant_edit || "sub"), text: f.dominant_edit || "?" })]),
        el("td", { className: "num", text: f2(f.wer) }),
        el("td", { className: "num", text: num(f.entity_error_rate) ? f2(f.entity_error_rate) : "n/a" }),
        el("td", { className: "fix", text: f.implied_fix || "—" })
      ]));
    });
    tbl.appendChild(tb);
    host.appendChild(el("div", { className: "scroll-x" }, [tbl]));

    if (d.signature_notes && d.signature_notes.length) {
      var ol = el("ol", { className: "ranked" });
      d.signature_notes.forEach(function (s) { ol.appendChild(el("li", { text: s })); });
      host.appendChild(el("div", { style: "margin-top:12px" }, [
        el("div", { className: "k", style: "font-size:11.5px;color:var(--ink-3);text-transform:uppercase;letter-spacing:.06em",
          text: "measured signatures (effect size vs the rest of the grid)" }), ol]));
    }
    if (d.entity_note) host.appendChild(el("p", { className: "prov", text: d.entity_note }));
  }

  function keyItem(label, color) {
    return el("span", {}, [el("i", { style: "background:" + color }), el("span", { text: label })]);
  }

  function stackedBar(f, maxRate) {
    var W = 260, H = 22;
    var svg = sv("svg", { viewBox: "0 0 " + W + " " + H, width: W, height: H,
      role: "img", "aria-label": f.family + " edit composition" });
    var x = scale(0, maxRate, 0, W - 2);
    var segs = [["del", f.del_rate, "var(--cat-1)"], ["sub", f.sub_rate, "var(--cat-5)"],
                ["ins", f.ins_rate, "var(--cat-4)"]];
    var acc = 0;
    segs.forEach(function (s) {
      var v = num(s[1]) ? s[1] : 0;
      if (v <= 0) { return; }
      svg.appendChild(sv("rect", { x: x(acc), y: 3, width: Math.max(1, x(acc + v) - x(acc)), height: H - 8,
        fill: s[2] }, [sv("title", { text: s[0] + " " + f3(v) })]));
      acc += v;
    });
    svg.appendChild(sv("rect", { x: 0, y: 3, width: W - 2, height: H - 8, fill: "none", stroke: "var(--rule)" }));
    if (num(f.entity_error_rate)) {
      var ex = x(Math.min(f.entity_error_rate, maxRate));
      svg.appendChild(sv("line", { x1: ex, x2: ex, y1: 0, y2: H, stroke: "var(--danger)", "stroke-width": 2.5 },
        [sv("title", { text: "entity error rate " + f3(f.entity_error_rate) })]));
    }
    return svg;
  }

  // ======================================================================
  // PANEL 4 — SENSITIVITY + PRE-REGISTRATION VERDICT
  // ======================================================================

  function renderSensitivity(host, d) {
    var v = d.verdict || {};
    var cls = v.status === "CONFIRMED" ? "confirmed"
      : (v.status === "NOT CONFIRMED" ? "notconfirmed" : "unresolved");
    host.appendChild(el("div", { className: "verdict " + cls }, [
      el("span", { className: "st", text: "PRE-REGISTERED " + (v.pair ? v.pair.join(" × ") : "interaction")
        + " — " + (v.status || "UNRESOLVED") }),
      el("p", { text: v.sentence || "no verdict sentence was produced" })
    ]));

    var bars = d.gap_bars || [];
    var cols = el("div", { className: "two-col" });
    if (bars.length) cols.appendChild(gapChart(bars));
    else cols.appendChild(el("div", { className: "empty" }, [
      el("div", { className: "t", text: "No Sobol indices in this build" }),
      el("div", { className: "r", text: "results/sobol.json was absent and no surrogate could be fitted." })]));

    var right = el("div", {});
    right.appendChild(el("div", { className: "k", style: "font-size:11.5px;color:var(--ink-3);text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px",
      text: "which pair interacts (ranking only)" }));
    var ol = el("ol", { className: "ranked" });
    (d.s2_pairs || []).slice(0, 6).forEach(function (p) {
      ol.appendChild(el("li", {}, [el("code", { text: p.pair.join(" × ") })]));
    });
    if (!(d.s2_pairs || []).length) ol.appendChild(el("li", { text: "no second-order pairs available" }));
    right.appendChild(ol);
    if (d.s2_caveat) right.appendChild(el("div", { className: "caveat", text: d.s2_caveat }));
    cols.appendChild(right);
    host.appendChild(cols);

    var ci = d.counterintuitive;
    if (ci && (ci.cells || []).length) {
      var lbl = ci.presentable_as_measured ? "confirmed against the oracle" : "UNCONFIRMED — GP predictions, not measurements";
      var ul = el("ol", { className: "ranked" });
      ci.cells.slice(0, 6).forEach(function (c) {
        ul.appendChild(el("li", { text: (c.kind || "cell") + ": " + (c.factor || "?")
          + (c.moderator ? " moderated by " + c.moderator : "")
          + (c.note ? " — " + c.note : "") }));
      });
      host.appendChild(el("div", { style: "margin-top:14px" }, [
        el("div", { className: "k", style: "font-size:11.5px;color:var(--ink-3);text-transform:uppercase;letter-spacing:.06em",
          text: "counterintuitive cells · " + lbl }), ul]));
    }
    if (d.provenance) host.appendChild(el("p", { className: "prov", text: d.provenance }));
  }

  function gapChart(bars) {
    var rowH = 34, M = { t: 24, r: 22, b: 34, l: 96 };
    var W = 560, H = M.t + rowH * bars.length + M.b;
    var lo = Math.min(0, Math.min.apply(null, bars.map(function (b) { return num(b.gap_lo) ? b.gap_lo : 0; })));
    var hi = Math.max.apply(null, bars.map(function (b) {
      return Math.max(num(b.ST) ? b.ST : 0, num(b.gap_hi) ? b.gap_hi : 0);
    }));
    var x = scale(Math.min(lo, 0), Math.max(hi, 0.1) * 1.06, M.l, W - M.r);
    var svg = sv("svg", { className: "chart", viewBox: "0 0 " + W + " " + H,
      preserveAspectRatio: "xMidYMid meet", role: "img", "aria-label": "first and total order Sobol indices" });

    svg.appendChild(sv("line", { className: "axis-line", x1: x(0), x2: x(0), y1: M.t - 6, y2: H - M.b }));
    ticks(Math.min(lo, 0), Math.max(hi, 0.1) * 1.06, 5).forEach(function (t) {
      svg.appendChild(sv("text", { className: "tick", x: x(t), y: H - M.b + 16, "text-anchor": "middle", text: t.toFixed(2) }));
    });
    svg.appendChild(sv("text", { className: "axis-title", x: (M.l + W - M.r) / 2, y: H - 5,
      "text-anchor": "middle", text: "variance share (Sobol index) — bar = S1, notch = ST, whisker = ST−S1 CI" }));

    bars.forEach(function (b, i) {
      var yy = M.t + i * rowH;
      svg.appendChild(sv("text", { x: M.l - 8, y: yy + 15, "text-anchor": "end", "font-size": 12,
        "font-weight": 600, fill: "var(--ink)", text: b.factor }));
      if (num(b.S1)) svg.appendChild(sv("rect", { x: x(Math.min(0, b.S1)), y: yy + 4,
        width: Math.abs(x(b.S1) - x(0)), height: 15, fill: "var(--cat-1)", rx: 2 },
        [sv("title", { text: "S1 = " + f3(b.S1) })]));
      if (num(b.ST)) {
        svg.appendChild(sv("line", { x1: x(b.ST), x2: x(b.ST), y1: yy + 1, y2: yy + 22,
          stroke: "var(--cat-2)", "stroke-width": 3 }, [sv("title", { text: "ST = " + f3(b.ST) })]));
      }
      if (num(b.gap_lo) && num(b.gap_hi)) {
        var cy = yy + 27;
        svg.appendChild(sv("line", { x1: x(b.gap_lo), x2: x(b.gap_hi), y1: cy, y2: cy,
          stroke: b.significant ? "var(--danger)" : "var(--ink-3)", "stroke-width": 2 }));
        [b.gap_lo, b.gap_hi].forEach(function (e) {
          svg.appendChild(sv("line", { x1: x(e), x2: x(e), y1: cy - 4, y2: cy + 4,
            stroke: b.significant ? "var(--danger)" : "var(--ink-3)", "stroke-width": 2 }));
        });
        svg.appendChild(sv("circle", { cx: x(num(b.gap) ? b.gap : 0), cy: cy, r: 3,
          fill: b.significant ? "var(--danger)" : "var(--ink-3)" },
          [sv("title", { text: "ST−S1 = " + sign3(b.gap) + " [" + f3(b.gap_lo) + ", " + f3(b.gap_hi) + "]" })]));
      }
    });
    return el("div", { className: "scroll-x" }, [svg]);
  }

  // ======================================================================
  // PANEL 5 — ACTIVE LEARNING (steppable)
  // ======================================================================

  function renderAL(host, d) {
    var frames = d.frames || [];
    var curves = d.curves;

    if (frames.length) {
      STATE.alStep = Math.min(STATE.alStep, frames.length - 1);
      var bar = el("div", { className: "stepper" });
      var slider = el("input", { type: "range", min: 0, max: frames.length - 1, value: STATE.alStep,
        "aria-label": "active-learning iteration" });
      var cnt = el("span", { className: "cnt" });
      var post = el("div", {});
      function draw() {
        var fr = frames[STATE.alStep];
        cnt.textContent = "step " + (STATE.alStep) + "/" + (frames.length - 1)
          + " · " + fr.n_evals + " oracle calls";
        clear(post).appendChild(posteriorSvg(fr, d));
      }
      slider.addEventListener("input", function () { STATE.alStep = +slider.value; draw(); });
      bar.appendChild(el("button", { text: "◀ back", onclick: function () {
        STATE.alStep = Math.max(0, STATE.alStep - 1); slider.value = STATE.alStep; draw(); } }));
      bar.appendChild(el("button", { text: "step ▶", onclick: function () {
        STATE.alStep = Math.min(frames.length - 1, STATE.alStep + 1); slider.value = STATE.alStep; draw(); } }));
      bar.appendChild(slider);
      bar.appendChild(cnt);
      host.appendChild(bar);
      host.appendChild(post);
      draw();
    } else {
      host.appendChild(el("div", { className: "empty" }, [
        el("div", { className: "t", text: "No GP posterior frames in this build" }),
        el("div", { className: "r", text: "the loop was not run (pass --al to build.py)" })]));
    }

    if (curves && (curves.series || []).length) {
      host.appendChild(el("div", { style: "margin-top:16px" }, [alCurveSvg(curves)]));
      var head = curves.headline || "";
      host.appendChild(el("p", { className: curves.presentable ? "note" : "caveat", text: head }));
      if (curves.provenance && curves.provenance.statement) {
        host.appendChild(el("p", { className: "prov", text: curves.provenance.statement }));
      }
      if (curves.evidence_level === "INSUFFICIENT") {
        host.appendChild(el("div", { className: "caveat", text:
          "Evidence level INSUFFICIENT — too few seeds for a savings claim. The curve is shown; the number is not." }));
      }
    }
  }

  function posteriorSvg(fr, d) {
    var nx = fr.nx, ny = fr.ny, mu = fr.mu;
    var cw = 44, chh = 30, M = { t: 10, r: 210, b: 40, l: 52 };
    var W = M.l + cw * nx + M.r, H = M.t + chh * ny + M.b;
    var svg = sv("svg", { className: "chart", viewBox: "0 0 " + W + " " + H,
      preserveAspectRatio: "xMinYMin meet", role: "img",
      style: "max-width:" + W + "px",   // don't let a small chart's type balloon
      "aria-label": "GP posterior mean WER over rt60 and SNR at iteration " + fr.step });

    for (var r = 0; r < ny; r++) {
      for (var c = 0; c < nx; c++) {
        var v = mu[r * nx + c];
        svg.appendChild(sv("rect", { x: M.l + c * cw, y: M.t + r * chh, width: cw, height: chh,
          fill: werColor(v) }, [sv("title", { text: "predicted WER " + f2(v) })]));
      }
    }
    // the failure contour the acquisition is chasing
    svg.appendChild(sv("rect", { x: M.l, y: M.t, width: cw * nx, height: chh * ny,
      fill: "none", stroke: "var(--rule-strong)" }));

    // evaluated points, projected onto the slice
    (fr.points || []).forEach(function (p) {
      var px = M.l + ((p.rt60 - d.x_domain[0]) / (d.x_domain[1] - d.x_domain[0])) * cw * nx;
      var py = M.t + (1 - (p.snr_db - d.y_domain[0]) / (d.y_domain[1] - d.y_domain[0])) * chh * ny;
      var isNew = p.new === true;
      svg.appendChild(sv("circle", { cx: px, cy: py, r: isNew ? 7 : 4,
        fill: isNew ? "var(--danger)" : "var(--surface)",
        stroke: isNew ? "var(--danger)" : "var(--ink)", "stroke-width": isNew ? 2.5 : 1.5,
        "fill-opacity": isNew ? 0.6 : 0.9 },
        [sv("title", { text: (isNew ? "chosen at this step" : "already evaluated")
          + "\nrt60 " + f2(p.rt60) + " snr " + f2(p.snr_db) + " WER " + f2(p.wer) })]));
    });

    ticks(d.y_domain[1], d.y_domain[0], ny - 1).forEach(function (v, i) {
      svg.appendChild(sv("text", { className: "tick", x: M.l - 7, y: M.t + i * chh + chh / 2 + 4,
        "text-anchor": "end", text: v.toFixed(0) }));
    });
    ticks(d.x_domain[0], d.x_domain[1], nx - 1).forEach(function (v, i) {
      svg.appendChild(sv("text", { className: "tick", x: M.l + i * cw + cw / 2, y: H - M.b + 16,
        "text-anchor": "middle", text: v.toFixed(2) }));
    });
    svg.appendChild(sv("text", { className: "axis-title", x: M.l + cw * nx / 2, y: H - 6,
      "text-anchor": "middle", "font-size": 11.5, text: "rt60 (s)" }));
    svg.appendChild(sv("text", { className: "axis-title", x: -(M.t + chh * ny / 2), y: 12,
      "text-anchor": "middle", transform: "rotate(-90)", "font-size": 11.5, text: "SNR (dB)" }));

    var lx = M.l + cw * nx + 16;
    svg.appendChild(sv("text", { x: lx, y: M.t + 14, "font-size": 12, "font-weight": 600,
      fill: "var(--ink)", text: "GP posterior mean" }));
    [["already evaluated", "var(--surface)", "var(--ink)"],
     ["chosen at this step", "var(--danger)", "var(--danger)"]].forEach(function (L, i) {
      svg.appendChild(sv("circle", { cx: lx + 7, cy: M.t + 34 + i * 20, r: 5, fill: L[1], stroke: L[2], "stroke-width": 2 }));
      svg.appendChild(sv("text", { x: lx + 20, y: M.t + 38 + i * 20, "font-size": 11.5, text: L[0] }));
    });
    if (num(fr.chosen_wer)) {
      svg.appendChild(sv("text", { x: lx, y: M.t + 92, "font-size": 11.5,
        text: "oracle WER here: " + f2(fr.chosen_wer) }));
    }
    if (num(d.threshold)) {
      svg.appendChild(sv("text", { x: lx, y: M.t + 112, "font-size": 11.5,
        text: "hunting the WER = " + f2(d.threshold) + " contour" }));
    }
    return el("div", { className: "scroll-x" }, [svg]);
  }

  var ARM_COLOR = { active_boundary: "var(--cat-1)", active_uncertainty: "var(--cat-3)", random: "var(--cat-2)" };

  function alCurveSvg(c) {
    var series = c.series || [];
    var W = 860, H = 320, M = { t: 14, r: 200, b: 46, l: 62 };
    var allx = [], ally = [];
    series.forEach(function (s) {
      s.n_evals.forEach(function (v) { allx.push(v); });
      s.lo.concat(s.hi, s.median).forEach(function (v) { if (num(v)) ally.push(v); });
    });
    if (!allx.length || !ally.length) throw new Error("active-vs-random curves are empty");
    var x = scale(Math.min.apply(null, allx), Math.max.apply(null, allx), M.l, W - M.r);
    var y = scale(0, Math.max.apply(null, ally) * 1.08, H - M.b, M.t);
    var svg = sv("svg", { className: "chart", viewBox: "0 0 " + W + " " + H,
      preserveAspectRatio: "xMidYMid meet", role: "img",
      style: "max-width:" + W + "px",
      "aria-label": "active versus random learning curves" });

    ticks(0, Math.max.apply(null, ally) * 1.08, 5).forEach(function (v) {
      svg.appendChild(sv("line", { className: "gridline", x1: M.l, x2: W - M.r, y1: y(v), y2: y(v) }));
      svg.appendChild(sv("text", { className: "tick", x: M.l - 7, y: y(v) + 4, "text-anchor": "end", text: v.toFixed(2) }));
    });
    series.forEach(function (s, i) {
      var col = ARM_COLOR[s.arm] || "var(--cat-4)";
      var band = "", back = "";
      s.n_evals.forEach(function (n, k) {
        if (!num(s.lo[k]) || !num(s.hi[k])) return;
        band += (band ? "L" : "M") + x(n) + "," + y(s.lo[k]);
      });
      for (var k = s.n_evals.length - 1; k >= 0; k--) {
        if (!num(s.hi[k])) continue;
        back += "L" + x(s.n_evals[k]) + "," + y(s.hi[k]);
      }
      if (band) svg.appendChild(sv("path", { d: band + back + "Z", fill: col, "fill-opacity": 0.14, stroke: "none" }));
      var line = "";
      s.n_evals.forEach(function (n, kk) {
        if (!num(s.median[kk])) return;
        line += (line ? "L" : "M") + x(n) + "," + y(s.median[kk]);
      });
      if (line) svg.appendChild(sv("path", { d: line, fill: "none", stroke: col, "stroke-width": 2.5 }));
      svg.appendChild(sv("text", { x: W - M.r + 12, y: M.t + 16 + i * 19, "font-size": 12,
        "font-weight": 600, fill: col, text: s.arm }));
    });
    if (num(c.target)) {
      svg.appendChild(sv("line", { x1: M.l, x2: W - M.r, y1: y(c.target), y2: y(c.target),
        stroke: "var(--ink-3)", "stroke-dasharray": "5 4" }));
      svg.appendChild(sv("text", { x: W - M.r - 4, y: y(c.target) - 6, "text-anchor": "end",
        "font-size": 11, text: "equal-fidelity target " + f3(c.target) }));
    }
    ticks(Math.min.apply(null, allx), Math.max.apply(null, allx), 5).forEach(function (v) {
      svg.appendChild(sv("text", { className: "tick", x: x(v), y: H - M.b + 17, "text-anchor": "middle", text: v.toFixed(0) }));
    });
    svg.appendChild(sv("line", { className: "axis-line", x1: M.l, x2: W - M.r, y1: H - M.b, y2: H - M.b }));
    svg.appendChild(sv("text", { className: "axis-title", x: (M.l + W - M.r) / 2, y: H - 6,
      "text-anchor": "middle", text: "oracle evaluations" }));
    svg.appendChild(sv("text", { className: "axis-title", x: -(M.t + H - M.b) / 2, y: 14,
      "text-anchor": "middle", transform: "rotate(-90)", text: c.metric || "boundary RMSE" }));
    svg.appendChild(sv("text", { x: W - M.r + 12, y: M.t + 16 + series.length * 19 + 10,
      "font-size": 11, fill: "var(--ink-3)", text: "band = min–max," }));
    svg.appendChild(sv("text", { x: W - M.r + 12, y: M.t + 16 + series.length * 19 + 26,
      "font-size": 11, fill: "var(--ink-3)",
      text: (c.n_seeds || "?") + " seeds (not a CI)" }));
    return el("div", { className: "scroll-x" }, [svg]);
  }

  // ======================================================================
  // PANEL 6 — SIM2REAL
  // ======================================================================

  function renderSim2Real(host, d) {
    var pts = (d.scatter || []).filter(function (p) { return num(p.wer_real) && num(p.wer_sim); });
    if (!pts.length) throw new Error("no measured/simulated condition pairs matched on measured RT60");
    var h = d.headline || {};

    var strip = el("div", { className: "quad-strip" }, [
      quadBox("matched pairs", h.n_pairs),
      statBox("mean sim − real WER", sign3(h.mean_gap),
        (h.ci && num(h.ci[0])) ? "95% CI [" + f3(h.ci[0]) + ", " + f3(h.ci[1]) + "]" : ""),
      statBox("Spearman rho (ordering)", f2(h.spearman), "does sim rank conditions like reality?"),
      statBox("dead-zone Jaccard", f2(h.dead_zone_jaccard), "same cells flagged dangerous?")
    ]);
    host.appendChild(strip);

    var W = 620, H = 420, M = { t: 14, r: 18, b: 46, l: 56 };
    var lim = d.identity || [0, 1];
    var lo = Math.max(0, Math.min(lim[0], 0) - 0.02), hi = Math.min(1.05, lim[1] + 0.04);
    var x = scale(lo, hi, M.l, W - M.r), y = scale(lo, hi, H - M.b, M.t);
    var svg = sv("svg", { className: "chart", viewBox: "0 0 " + W + " " + H,
      preserveAspectRatio: "xMidYMid meet", role: "img", "aria-label": "measured versus simulated WER" });

    ticks(lo, hi, 5).forEach(function (v) {
      svg.appendChild(sv("line", { className: "gridline", x1: M.l, x2: W - M.r, y1: y(v), y2: y(v) }));
      svg.appendChild(sv("text", { className: "tick", x: M.l - 7, y: y(v) + 4, "text-anchor": "end", text: v.toFixed(2) }));
      svg.appendChild(sv("text", { className: "tick", x: x(v), y: H - M.b + 17, "text-anchor": "middle", text: v.toFixed(2) }));
    });
    svg.appendChild(sv("line", { x1: x(lo), y1: y(lo), x2: x(hi), y2: y(hi),
      stroke: "var(--ink-3)", "stroke-dasharray": "6 4", "stroke-width": 1.5 }));
    svg.appendChild(sv("text", { x: x(hi) - 6, y: y(hi) + 18, "text-anchor": "end", "font-size": 11,
      fill: "var(--ink-3)", text: "y = x (perfect sim)" }));

    pts.forEach(function (p) {
      var dz = p.dead_zone_real || p.dead_zone_sim;
      svg.appendChild(sv("circle", { cx: x(p.wer_real), cy: y(p.wer_sim), r: dz ? 6.5 : 5,
        fill: werColor(p.wer_real),
        stroke: dz ? "var(--danger)" : "var(--rule-strong)", "stroke-width": dz ? 2.5 : 1 },
        [sv("title", { text: p.condition + "\nreal " + f3(p.wer_real) + " · sim " + f3(p.wer_sim)
          + "\ngap " + sign3(p.gap) + (dz ? "\nflagged a dead zone" : "") })]));
    });
    svg.appendChild(sv("line", { className: "axis-line", x1: M.l, x2: W - M.r, y1: H - M.b, y2: H - M.b }));
    svg.appendChild(sv("line", { className: "axis-line", x1: M.l, x2: M.l, y1: M.t, y2: H - M.b }));
    svg.appendChild(sv("text", { className: "axis-title", x: (M.l + W - M.r) / 2, y: H - 6,
      "text-anchor": "middle", text: "WER with MEASURED RIRs" }));
    svg.appendChild(sv("text", { className: "axis-title", x: -(M.t + H - M.b) / 2, y: 14,
      "text-anchor": "middle", transform: "rotate(-90)", text: "WER with SIMULATED RIRs" }));

    host.appendChild(el("div", { className: "two-col" }, [
      el("div", { className: "scroll-x" }, [svg]),
      el("div", {}, [
        el("div", { className: "verdict " + s2rClass(h.verdict) }, [
          el("span", { className: "st", text: "SIM2REAL — " + (h.verdict || "see numbers") }),
          el("p", { text: d.verdict_sentence || ("Points below the dashed line mean the simulator is "
            + "KINDER than the measured rooms; the useful question is not whether the level matches "
            + "but whether the ORDERING does.") })
        ]),
        el("p", { className: "prov", text: "Pairs are matched on MEASURED RT60 of both files, never on the "
          + "requested value — a simulator that misses its target would otherwise be compared against "
          + "the wrong room." })
      ])
    ]));
  }

  function s2rClass(v) {
    v = String(v || "");
    if (v.indexOf("SIM TRACKS") === 0) return "confirmed";
    if (v.indexOf("NOT PRESERVED") >= 0 || v.indexOf("INSUFFICIENT") >= 0) return "notconfirmed";
    return "unresolved";
  }

  function statBox(label, value, sub) {
    return el("div", { className: "quad" }, [
      el("div", { className: "n", style: "font-size:19px", text: value }),
      el("div", { className: "k", text: label + (sub ? " · " + sub : "") })
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
      badge.textContent = (meta.is_synthetic ? "SYNTHETIC DATA" : "REAL GRID")
        + " · " + (meta.n_rows || 0) + " rows · " + (meta.n_conditions || 0) + " conditions";
      badge.title = "source: " + (meta.source || "unknown");
    }
    var built = document.getElementById("built-at");
    if (built) built.textContent = "built " + (meta.generated || "?") + " from " + (meta.source || "?");

    var host = document.getElementById("model-toggle");
    if (!host) return;
    clear(host);
    var models = Object.keys(DATA.models || {});
    host.appendChild(el("span", { className: "lbl", text: "model" }));
    if (!models.length) {
      host.appendChild(el("button", { text: "no models in table", disabled: "disabled" }));
      return;
    }
    models.forEach(function (m) {
      host.appendChild(el("button", {
        text: m, "aria-pressed": String(m === STATE.model),
        onclick: function () { STATE.model = m; STATE.pinned = null; STATE.alStep = 0; renderAll(); }
      }));
    });
  }

  function renderAll() {
    renderHeader();
    var m = (DATA.models || {})[STATE.model] || {};
    var missing = { status: "missing", reason: "No model is selected, or the master table held no rows "
      + "for this model. Check the model column of the source table." };

    panel("panel-hero", m.silent_failure || missing, renderHero, "The silent-failure map is not available");
    panel("panel-heatmap", m.silent_failure || missing, renderHeatmap, "The factor-space heatmap is not available");
    panel("panel-fingerprints", m.fingerprints || missing, renderFingerprints, "Failure fingerprints are not available");
    panel("panel-sensitivity", m.sensitivity || missing, renderSensitivity, "The sensitivity panel is not available");
    panel("panel-al", m.active_learning || missing, renderAL, "The active-learning panel is not available");
    panel("panel-sim2real", m.sim2real || missing, renderSim2Real, "The sim-vs-real panel is not available");

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
