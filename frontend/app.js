/**
 * rubin-skymap — dashboard application
 *
 * Canvas equatorial sky-map + WebSocket live feed + detail inspector.
 * Pure vanilla JS. No dependencies, no build step.
 *
 * Rendering pipeline (back → front):
 *   1. Background + clear
 *   2. Milky Way band (offscreen blurred canvas, optional)
 *   3. RA/Dec grid
 *   4. Hipparcos stars (size by magnitude, optional)
 *   5. Constellation lines (optional)
 *   6. Constellation labels (optional)
 *   7. Alert dots (coloured by class)
 *   8. Selection highlight
 */

"use strict";

/* ══════════════════════════════════════════════════════════════
   CONFIG
══════════════════════════════════════════════════════════════ */

const CLASS_COLORS = {
  "SN Ia":    "#ff6b6b",
  "SN II":    "#ffa94d",
  "SN Ibc":   "#ffd43b",
  "SLSN":     "#69db7c",
  "Kilonova": "#4dabf7",
  "AGN":      "#b197fc",
  "RRL":      "#f783ac",
  "Other":    "#868e96",
};

const ALL_CLASSES = Object.keys(CLASS_COLORS);

/* Map padding (pixels) */
const PAD = { top: 28, right: 18, bottom: 30, left: 36 };

/* ══════════════════════════════════════════════════════════════
   STATE
══════════════════════════════════════════════════════════════ */

let allAlerts      = [];            // full alert store
let visibleClasses = new Set(ALL_CLASSES);
let selectedOid    = null;
let searchQuery    = "";
let classCounts    = {};            // live counts per class
let alertsPerMin   = 0;
let _rateBuffer    = [];            // timestamps for rate calculation

let wsReconnectDelay = 1000;
const WS_MAX_DELAY   = 30000;
let _ws = null;

/* Canvas refs */
let skyCanvas = null;
let skyCtx    = null;

/* Tooltip element */
let tooltip   = null;

/* ── Sky asset data ───────────────────────────────────────────── */
let skyStars          = [];   // [{ra_h, dec_deg, mag, name}]
let skyConstellations = [];   // [{name, segments:[ra1,dec1,ra2,dec2]}]
let skyMilkyway       = null; // {points:[[ra_h,dec_deg],...], width_deg}

/* ── Sky layer toggle state ───────────────────────────────────── */
const skyToggles = {
  milkyway:      true,
  stars:         true,
  constellations:true,
  labels:        false,
};

/* ══════════════════════════════════════════════════════════════
   COORDINATE HELPERS
══════════════════════════════════════════════════════════════ */

/**
 * Convert equatorial (RA hours, Dec degrees) → canvas pixel (x, y).
 * RA increases right-to-left (east → left). Dec +90 at top.
 *
 * @param {number} ra  - RA in decimal hours [0, 24)
 * @param {number} dec - Dec in degrees [-90, 90]
 * @param {number} W   - canvas width px
 * @param {number} H   - canvas height px
 * @returns {{x:number, y:number}}
 */
function sky2px(ra, dec, W, H) {
  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top  - PAD.bottom;
  const x = PAD.left  + ((24 - ra) / 24) * innerW;
  const y = PAD.top   + ((90 - dec) / 180) * innerH;
  return { x, y };
}

/**
 * Convert canvas pixel → equatorial coords (for mouse hover display).
 * @returns {{ra:number, dec:number}}
 */
function px2sky(px, py, W, H) {
  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top  - PAD.bottom;
  const ra  = 24  - ((px - PAD.left)  / innerW) * 24;
  const dec = 90  - ((py - PAD.top)   / innerH) * 180;
  return { ra, dec };
}

/* ══════════════════════════════════════════════════════════════
   SKY MAP RENDERING
══════════════════════════════════════════════════════════════ */

/**
 * Draw the equatorial grid (RA lines, Dec lines, labels).
 * Also clears and fills the background.
 */
function drawGrid() {
  const W = skyCanvas.width;
  const H = skyCanvas.height;

  skyCtx.clearRect(0, 0, W, H);

  /* Background gradient */
  const grad = skyCtx.createRadialGradient(W / 2, H / 2, 0, W / 2, H / 2, Math.max(W, H) * 0.7);
  grad.addColorStop(0, "#05091000");
  grad.addColorStop(1, "#020408");
  skyCtx.fillStyle = "#03060b";
  skyCtx.fillRect(0, 0, W, H);
  skyCtx.fillStyle = grad;
  skyCtx.fillRect(0, 0, W, H);

  skyCtx.save();

  /* ── Dec grid lines ── */
  const decLines = [-75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75];
  for (const dec of decLines) {
    const { y } = sky2px(0, dec, W, H);
    const isEq   = dec === 0;
    skyCtx.beginPath();
    skyCtx.moveTo(PAD.left, y);
    skyCtx.lineTo(W - PAD.right, y);
    skyCtx.strokeStyle = isEq ? "#1e3a4a" : "#0e1e28";
    skyCtx.lineWidth   = isEq ? 1.0 : 0.5;
    skyCtx.stroke();

    /* Dec label on right edge */
    skyCtx.fillStyle  = isEq ? "#3a6070" : "#1e3040";
    skyCtx.font       = `${isEq ? 10 : 9}px "Courier New", monospace`;
    skyCtx.textAlign  = "left";
    skyCtx.fillText((dec > 0 ? "+" : "") + dec + "°", W - PAD.right + 3, y + 3);
  }

  /* ── RA grid lines ── */
  for (let ra = 0; ra <= 24; ra += 2) {
    const { x } = sky2px(ra, 0, W, H);
    const isMajor = ra % 6 === 0;
    skyCtx.beginPath();
    skyCtx.moveTo(x, PAD.top);
    skyCtx.lineTo(x, H - PAD.bottom);
    skyCtx.strokeStyle = isMajor ? "#0e1e28" : "#080f14";
    skyCtx.lineWidth   = isMajor ? 0.6 : 0.4;
    skyCtx.stroke();

    /* RA label at bottom */
    skyCtx.fillStyle  = isMajor ? "#3a6070" : "#1e3040";
    skyCtx.font       = `${isMajor ? 10 : 9}px "Courier New", monospace`;
    skyCtx.textAlign  = "center";
    skyCtx.fillText(ra + "h", x, H - PAD.bottom + 14);
  }

  /* ── Border rect ── */
  const bx = PAD.left, by = PAD.top;
  const bw = W - PAD.left - PAD.right;
  const bh = H - PAD.top  - PAD.bottom;
  skyCtx.strokeStyle = "#1a2e3a";
  skyCtx.lineWidth   = 1;
  skyCtx.strokeRect(bx, by, bw, bh);

  /* ── Axis captions ── */
  skyCtx.fillStyle = "#2a4a5a";
  skyCtx.font      = "10px 'Courier New', monospace";
  skyCtx.textAlign = "left";
  skyCtx.fillText("← RA (hours) →  24h east", PAD.left, H - 4);
  skyCtx.save();
  skyCtx.translate(12, H / 2);
  skyCtx.rotate(-Math.PI / 2);
  skyCtx.textAlign = "center";
  skyCtx.fillText("Dec (°)", 0, 0);
  skyCtx.restore();

  skyCtx.restore();
}

/* ─────────────────────────────────────────────────────────────
   MILKY WAY
   Three clean layers, all on an offscreen canvas composited at
   low opacity with source-over so it stays a background hint,
   never a dominant feature.

   L1  Wide soft haze  — the broad diffuse band of unresolved stars
   L2  Narrow spine    — the slightly brighter galactic-plane core
   L3  Dust notch      — a hairline destination-out that cuts the
                         very centre, hinting at dust-lane absorption
───────────────────────────────────────────────────────────── */

function drawMilkyWay(ctx) {
  if (!skyToggles.milkyway || !skyMilkyway || !skyMilkyway.points.length) return;

  const W   = skyCanvas.width;
  const H   = skyCanvas.height;
  const dpr = window.devicePixelRatio || 1;

  const pts    = skyMilkyway.points;
  const innerH = (H / dpr) - PAD.top - PAD.bottom;
  const bandPx = (skyMilkyway.width_deg / 180) * innerH * dpr;

  /* ── offscreen canvas ─────────────────────────────────────── */
  const off = document.createElement("canvas");
  off.width  = W;
  off.height = H;
  const oc  = off.getContext("2d");

  /* Convert polyline to canvas coords, split at RA wrap-around */
  const segments = [];
  let seg = [];
  for (let i = 0; i < pts.length; i++) {
    const [ra, dec] = pts[i];
    if (i > 0 && Math.abs(ra - pts[i - 1][0]) > 12) {
      if (seg.length) segments.push(seg);
      seg = [];
    }
    seg.push(sky2px(ra, dec, W, H));
  }
  if (seg.length) segments.push(seg);

  function _stroke() {
    for (const s of segments) {
      oc.beginPath();
      oc.moveTo(s[0].x, s[0].y);
      for (let k = 1; k < s.length; k++) oc.lineTo(s[k].x, s[k].y);
      oc.stroke();
    }
  }

  /* L1 — wide diffuse haze */
  oc.save();
  oc.filter      = "blur(18px)";
  oc.strokeStyle = "rgba(200,210,230,0.22)";
  oc.lineWidth   = Math.max(10, bandPx * 0.90);
  oc.lineCap     = "round";
  oc.lineJoin    = "round";
  _stroke();
  oc.restore();

  /* L2 — brighter inner spine */
  oc.save();
  oc.filter      = "blur(4px)";
  oc.strokeStyle = "rgba(210,225,255,0.38)";
  oc.lineWidth   = Math.max(3, bandPx * 0.18);
  oc.lineCap     = "round";
  oc.lineJoin    = "round";
  _stroke();
  oc.restore();

  /* L3 — dust-lane notch (erase a thin thread from the core) */
  oc.save();
  oc.filter                   = "blur(1px)";
  oc.globalCompositeOperation = "destination-out";
  oc.strokeStyle              = "rgba(0,0,0,0.18)";
  oc.lineWidth                = Math.max(1, bandPx * 0.04);
  oc.lineCap                  = "round";
  oc.lineJoin                 = "round";
  _stroke();
  oc.restore();

  /* Composite onto main canvas — normal blend, kept subtle */
  ctx.save();
  ctx.globalAlpha = 0.55;
  ctx.drawImage(off, 0, 0);
  ctx.globalAlpha = 1.0;
  ctx.restore();
}

/* ─────────────────────────────────────────────────────────────
   STARS
───────────────────────────────────────────────────────────── */

/**
 * Draw Hipparcos stars as small white dots scaled by magnitude.
 * size_px = max(0.5, (7.0 - mag) * 0.6)  — skip stars fainter than mag 7.
 */
function drawStars(ctx) {
  if (!skyToggles.stars || !skyStars.length) return;

  const W   = skyCanvas.width;
  const H   = skyCanvas.height;
  const dpr = window.devicePixelRatio || 1;

  ctx.save();
  for (const star of skyStars) {
    if (star.mag > 6.5) continue;

    /* Radius in logical px, scaled by DPR so dots stay physically tiny.
       mag −1.5 → r≈1.6 px    mag 3 → r≈0.9 px    mag 6.5 → r≈0.5 px  */
    const logicalR = Math.max(0.5, (6.5 - star.mag) * 0.22);
    const r = logicalR * dpr;

    const { x, y } = sky2px(star.ra_h, star.dec_deg, W, H);

    /* Clip to inner map area */
    if (x < PAD.left || x > W - PAD.right || y < PAD.top || y > H - PAD.bottom) continue;

    /* Opacity: brighter stars are more opaque; faint stars near transparent */
    const alpha = star.mag < 2 ? 0.90 : star.mag < 4 ? 0.70 : 0.45;

    /* Colour tint: very bright slightly warm, rest cool white */
    let col;
    if (star.mag < 1.0)      col = `rgba(255,245,210,${alpha})`;
    else if (star.mag < 3.0) col = `rgba(220,230,255,${alpha})`;
    else                     col = `rgba(200,210,240,${alpha})`;

    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = col;
    ctx.fill();
  }
  ctx.restore();
}

/* ─────────────────────────────────────────────────────────────
   CONSTELLATION LINES
───────────────────────────────────────────────────────────── */

/**
 * Draw constellation line segments.
 * Stroke: rgba(140,170,220,0.18), lineWidth 0.5
 */
function drawConstellations(ctx) {
  if (!skyToggles.constellations || !skyConstellations.length) return;

  const W   = skyCanvas.width;
  const H   = skyCanvas.height;
  const dpr = window.devicePixelRatio || 1;

  ctx.save();
  ctx.strokeStyle = "rgba(100,140,200,0.30)";
  ctx.lineWidth   = 0.8 * dpr;
  ctx.lineCap     = "round";

  for (const c of skyConstellations) {
    for (const seg of c.segments) {
      const [ra1, dec1, ra2, dec2] = seg;

      /* Skip segments that cross the RA wrap-around (>12 h span) */
      if (Math.abs(ra1 - ra2) > 12) continue;

      const p1 = sky2px(ra1, dec1, W, H);
      const p2 = sky2px(ra2, dec2, W, H);

      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.stroke();
    }
  }
  ctx.restore();
}

/* ─────────────────────────────────────────────────────────────
   CONSTELLATION LABELS
───────────────────────────────────────────────────────────── */

/**
 * Draw constellation name labels at their centroid,
 * only when the centroid falls inside the visible map area.
 */
function drawLabels(ctx) {
  if (!skyToggles.labels || !skyConstellations.length) return;

  const W = skyCanvas.width;
  const H = skyCanvas.height;

  ctx.save();
  ctx.font      = "9px 'Courier New', monospace";
  ctx.fillStyle = "rgba(180,200,230,0.50)";
  ctx.textAlign = "center";

  for (const c of skyConstellations) {
    if (!c.segments.length) continue;

    /* Compute centroid from all segment endpoints */
    let raSum = 0, decSum = 0, n = 0;
    for (const [ra1, dec1, ra2, dec2] of c.segments) {
      raSum  += ra1 + ra2;
      decSum += dec1 + dec2;
      n += 2;
    }
    const raMean  = raSum / n;
    const decMean = decSum / n;

    const { x, y } = sky2px(raMean, decMean, W, H);

    /* Only draw if centroid is inside the inner map area */
    if (x < PAD.left || x > W - PAD.right || y < PAD.top || y > H - PAD.bottom) continue;

    ctx.fillText(c.name, x, y);
  }
  ctx.restore();
}

/* ─────────────────────────────────────────────────────────────
   ALERT DOTS
───────────────────────────────────────────────────────────── */

/**
 * Render one alert dot.
 * @param {Object}  alert     - alert object
 * @param {boolean} highlight - draw ring if selected
 */
function drawDot(alert, highlight) {
  if (!visibleClasses.has(alert.predicted_class)) return;
  const W   = skyCanvas.width, H = skyCanvas.height;
  const dpr = window.devicePixelRatio || 1;
  const { x, y } = sky2px(alert.ra, alert.dec, W, H);

  /* Clip to inner map area */
  if (x < PAD.left || x > W - PAD.right || y < PAD.top || y > H - PAD.bottom) return;

  const col = CLASS_COLORS[alert.predicted_class] || CLASS_COLORS["Other"];
  /* Scale dot radius with DPR so they stay visually large on HiDPI */
  const r   = (highlight ? 6 : 4) * dpr;

  /* Soft dark halo so dot stands out against bright stars */
  skyCtx.save();
  skyCtx.beginPath();
  skyCtx.arc(x, y, r + 1.5 * dpr, 0, Math.PI * 2);
  skyCtx.fillStyle   = "rgba(0,0,0,0.55)";
  skyCtx.fill();

  /* Glow for highlighted dot */
  if (highlight) {
    skyCtx.shadowColor = col;
    skyCtx.shadowBlur  = 16 * dpr;
  }

  skyCtx.beginPath();
  skyCtx.arc(x, y, r, 0, Math.PI * 2);
  skyCtx.fillStyle   = col;
  skyCtx.globalAlpha = highlight ? 1.0 : 0.92;
  skyCtx.fill();
  skyCtx.globalAlpha = 1.0;

  if (highlight) {
    skyCtx.strokeStyle = "#ffffff";
    skyCtx.lineWidth   = 1.5 * dpr;
    skyCtx.stroke();

    /* outer pulse ring */
    skyCtx.beginPath();
    skyCtx.arc(x, y, r + 5 * dpr, 0, Math.PI * 2);
    skyCtx.strokeStyle = col + "60";
    skyCtx.lineWidth   = dpr;
    skyCtx.stroke();
  }
  skyCtx.restore();
}

/* ─────────────────────────────────────────────────────────────
   FULL REDRAW PIPELINE
───────────────────────────────────────────────────────────── */

/** Full redraw in layer order: clear → milkyway → grid → stars → constellations → labels → alerts */
function redrawMap() {
  const ctx = skyCtx;
  drawGrid();
  drawMilkyWay(ctx);
  drawStars(ctx);
  drawConstellations(ctx);
  drawLabels(ctx);
  for (const alert of allAlerts) {
    drawDot(alert, alert.object_id === selectedOid);
  }
}

/* ══════════════════════════════════════════════════════════════
   HIT TESTING
══════════════════════════════════════════════════════════════ */

/**
 * Find closest alert within threshold pixels of (cx, cy).
 * @returns {Object|null}
 */
function hitTest(cx, cy) {
  const W = skyCanvas.width, H = skyCanvas.height;
  const THRESH2 = 15 * 15;
  let best = null, bestD = THRESH2;
  for (const a of allAlerts) {
    if (!visibleClasses.has(a.predicted_class)) continue;
    const { x, y } = sky2px(a.ra, a.dec, W, H);
    const d = (x - cx) ** 2 + (y - cy) ** 2;
    if (d < bestD) { bestD = d; best = a; }
  }
  return best;
}

/* ══════════════════════════════════════════════════════════════
   ALERT LIST (feed panel)
══════════════════════════════════════════════════════════════ */

/**
 * Prepend one row to the alert feed list.
 * @param {Object} alert
 * @param {boolean} flash - animate new arrival
 */
function prependListItem(alert, flash) {
  const ul   = document.getElementById("alert-list");
  const col  = CLASS_COLORS[alert.predicted_class] || CLASS_COLORS["Other"];
  const prob = ((alert.predicted_prob || 0) * 100).toFixed(1);

  const li         = document.createElement("li");
  li.className     = "alert-item" + (flash ? " new-flash" : "");
  li.dataset.oid   = alert.object_id;

  /* Check visibility */
  if (!visibleClasses.has(alert.predicted_class)) li.style.display = "none";

  /* Check search */
  const q = searchQuery.toLowerCase();
  if (q && !alert.object_id.toLowerCase().includes(q) && !alert.predicted_class.toLowerCase().includes(q)) {
    li.style.display = "none";
  }

  li.innerHTML = `
    <span class="ai-dot" style="background:${col}"></span>
    <span class="ai-oid" title="${alert.object_id}">${alert.object_id}</span>
    <span class="ai-cls" style="color:${col}">${alert.predicted_class}</span>
    <span class="ai-prob">${prob}%</span>
  `;
  li.addEventListener("click", () => selectAlert(alert));

  /* Cap list at 400 rows */
  if (ul.children.length >= 400) ul.removeChild(ul.lastChild);
  ul.insertBefore(li, ul.firstChild);
}

/** Apply search + class filter to existing list items. */
function applyListFilter() {
  const q    = searchQuery.toLowerCase();
  const items = document.querySelectorAll("#alert-list .alert-item");
  for (const li of items) {
    const oid   = li.dataset.oid;
    const alert = allAlerts.find(a => a.object_id === oid);
    if (!alert) { li.style.display = "none"; continue; }
    const classOk = visibleClasses.has(alert.predicted_class);
    const searchOk = !q || alert.object_id.toLowerCase().includes(q) || alert.predicted_class.toLowerCase().includes(q);
    li.style.display = (classOk && searchOk) ? "" : "none";
  }
}

/* ══════════════════════════════════════════════════════════════
   DETAIL / INSPECTOR PANEL
══════════════════════════════════════════════════════════════ */

/**
 * Show the inspector panel for *alert*.
 * @param {Object} alert
 */
function selectAlert(alert) {
  selectedOid = alert.object_id;

  /* Highlight selected list row */
  document.querySelectorAll(".alert-item").forEach(el => {
    el.classList.toggle("selected", el.dataset.oid === alert.object_id);
  });

  /* Switch panels */
  document.getElementById("detail-empty").style.display = "none";
  const body = document.getElementById("detail-body");
  body.style.display = "block";

  /* Class badge */
  const col   = CLASS_COLORS[alert.predicted_class] || CLASS_COLORS["Other"];
  const badge = document.getElementById("detail-class-badge");
  badge.textContent   = alert.predicted_class;
  badge.style.background = col + "22";
  badge.style.color      = col;
  badge.style.border     = `1px solid ${col}44`;

  document.getElementById("detail-oid").textContent = alert.object_id;

  /* KV fields */
  document.getElementById("dv-ra").textContent   = typeof alert.ra  === "number" ? alert.ra.toFixed(5) + " h"  : "—";
  document.getElementById("dv-dec").textContent  = typeof alert.dec === "number" ? alert.dec.toFixed(5) + "°"  : "—";
  document.getElementById("dv-prob").textContent = typeof alert.predicted_prob === "number"
    ? (alert.predicted_prob * 100).toFixed(2) + "%" : "—";
  document.getElementById("dv-time").textContent = alert.created_at
    ? new Date(alert.created_at).toLocaleTimeString() : "—";

  /* Probability bars + inline class explain */
  _renderProbBars(alert);
  _renderClassExplain(alert.predicted_class);

  /* SHAP chart */
  let shap = alert.shap || alert.shap_json || [];
  if (typeof shap === "string") { try { shap = JSON.parse(shap); } catch (_) { shap = []; } }
  drawShapChart(shap.slice(0, 5));

  redrawMap();
}

/**
 * Render probability bars for all classes.
 * @param {Object} alert
 */
function _renderProbBars(alert) {
  const container = document.getElementById("prob-bars");
  container.innerHTML = "";

  /* probabilities may come from /api/predict response or be absent on DB rows */
  let probs = alert.probabilities || {};
  if (typeof probs === "string") { try { probs = JSON.parse(probs); } catch (_) { probs = {}; } }

  /* If no probabilities dict, fake it from predicted_class + prob */
  if (Object.keys(probs).length === 0 && alert.predicted_class) {
    probs = { [alert.predicted_class]: alert.predicted_prob || 0 };
  }

  /* Sort classes by probability descending */
  const sorted = Object.entries(probs)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 8);

  for (const [cls, p] of sorted) {
    const col  = CLASS_COLORS[cls] || CLASS_COLORS["Other"];
    const pct  = (p * 100).toFixed(1);
    const w    = Math.max(0, Math.min(100, p * 100));

    const row  = document.createElement("div");
    row.className = "prob-row";
    row.innerHTML = `
      <span class="prob-name" style="color:${col}">${cls}</span>
      <div class="prob-track">
        <div class="prob-fill" style="width:${w}%;background:${col}"></div>
      </div>
      <span class="prob-val">${pct}%</span>
    `;
    container.appendChild(row);
  }
}

/**
 * Draw SHAP horizontal bar chart on #shap-chart canvas.
 * Positive = red, negative = blue, centred at zero.
 * @param {Array<{feature:string, value:number, contribution:number}>} shapData
 */
function drawShapChart(shapData) {
  const canvas = document.getElementById("shap-chart");
  if (!canvas || shapData.length === 0) return;

  const ROW_H = 26;
  const W     = canvas.parentElement ? canvas.parentElement.clientWidth - 20 : 290;
  canvas.width  = W;
  canvas.height = shapData.length * ROW_H + 4;

  const ctx    = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const LABEL_W = 105;
  const BAR_W   = W - LABEL_W - 46;
  const maxAbs  = Math.max(...shapData.map(d => Math.abs(d.contribution)), 1e-9);

  shapData.forEach((d, i) => {
    const y    = i * ROW_H + 2;
    const mid  = LABEL_W + BAR_W / 2;
    const frac = d.contribution / maxAbs;
    const bLen = Math.abs(frac) * (BAR_W / 2);
    const barX = frac >= 0 ? mid : mid - bLen;

    /* Feature name */
    ctx.fillStyle  = "#7d8590";
    ctx.font       = "10px 'Courier New', monospace";
    ctx.textAlign  = "right";
    ctx.fillText(d.feature, LABEL_W - 5, y + 16);

    /* Bar */
    const barCol = frac >= 0 ? "#e05252" : "#3b82d4";
    ctx.fillStyle = barCol + "bb";
    roundRect(ctx, barX, y + 6, bLen || 1, ROW_H - 12, 2);
    ctx.fill();

    /* Center axis */
    ctx.strokeStyle = "#21262d";
    ctx.lineWidth   = 1;
    ctx.beginPath();
    ctx.moveTo(mid, y + 2);
    ctx.lineTo(mid, y + ROW_H - 2);
    ctx.stroke();

    /* Contribution value */
    ctx.fillStyle  = "#484f58";
    ctx.textAlign  = "left";
    ctx.font       = "9px 'Courier New', monospace";
    ctx.fillText(d.contribution.toFixed(3), LABEL_W + BAR_W + 4, y + 16);
  });
}

/** Draw a rounded rectangle path. */
function roundRect(ctx, x, y, w, h, r) {
  if (w < 2 * r) r = w / 2;
  if (h < 2 * r) r = h / 2;
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y,     x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x,     y + h, r);
  ctx.arcTo(x,     y + h, x,     y,     r);
  ctx.arcTo(x,     y,     x + w, y,     r);
  ctx.closePath();
}

/* ══════════════════════════════════════════════════════════════
   STATS BAR
══════════════════════════════════════════════════════════════ */

/** Update header stat chips and legend counts. */
function updateStats() {
  /* Total */
  const total = allAlerts.length;
  const el = document.getElementById("stat-total");
  if (el) el.textContent = total.toLocaleString();

  /* Rate */
  const now = Date.now();
  _rateBuffer.push(now);
  _rateBuffer = _rateBuffer.filter(t => now - t < 60000);
  const rate = _rateBuffer.length;
  const rEl = document.getElementById("stat-rate");
  if (rEl) rEl.textContent = rate + "/min";

  /* Top class */
  const topEntry = Object.entries(classCounts).sort((a, b) => b[1] - a[1])[0];
  const topEl = document.getElementById("stat-top");
  if (topEl) {
    if (topEntry) {
      topEl.textContent = topEntry[0];
      topEl.style.color = CLASS_COLORS[topEntry[0]] || "#e6edf3";
    } else {
      topEl.textContent = "—";
      topEl.style.color = "";
    }
  }

  /* Feed badge */
  const badge = document.getElementById("feed-badge");
  if (badge) badge.textContent = total;

  /* Legend counts */
  for (const cls of ALL_CLASSES) {
    const key = cls.replace(/ /g, "_");
    const el2 = document.getElementById("count-" + key);
    if (el2) el2.textContent = classCounts[cls] || 0;
  }
}

/* ══════════════════════════════════════════════════════════════
   WEBSOCKET
══════════════════════════════════════════════════════════════ */

function setConnStatus(state) {
  const el = document.getElementById("conn-status");
  const lb = document.getElementById("conn-label");
  if (!el) return;
  el.className   = "conn-" + state;
  lb.textContent = state === "open" ? "Live" : state === "pending" ? "Connecting…" : "Disconnected";
}

function connectWS() {
  setConnStatus("pending");
  const proto = location.protocol === "https:" ? "wss" : "ws";
  _ws = new WebSocket(`${proto}://${location.host}/ws/alerts`);

  _ws.onopen = () => {
    setConnStatus("open");
    wsReconnectDelay = 1000;
    console.log("[rubin-skymap] WebSocket connected");
  };

  _ws.onmessage = evt => {
    let alert;
    try { alert = JSON.parse(evt.data); } catch (_) { return; }
    ingestAlert(alert, true);
  };

  _ws.onerror = e => console.warn("[rubin-skymap] WS error", e);

  _ws.onclose = () => {
    setConnStatus("closed");
    setTimeout(connectWS, wsReconnectDelay);
    wsReconnectDelay = Math.min(wsReconnectDelay * 2, WS_MAX_DELAY);
  };
}

/* ══════════════════════════════════════════════════════════════
   INGEST
══════════════════════════════════════════════════════════════ */

/**
 * Accept one alert into the app state.
 * @param {Object}  alert
 * @param {boolean} flash - true for live WS arrivals, false for history load
 */
function ingestAlert(alert, flash) {
  if (allAlerts.some(a => a.object_id === alert.object_id)) return;

  allAlerts.push(alert);
  classCounts[alert.predicted_class] = (classCounts[alert.predicted_class] || 0) + 1;

  prependListItem(alert, flash);
  drawDot(alert, false);
  updateStats();
}

/* ══════════════════════════════════════════════════════════════
   INITIAL LOAD
══════════════════════════════════════════════════════════════ */

async function loadHistory() {
  try {
    const resp = await fetch("/api/alerts?limit=200");
    if (!resp.ok) throw new Error("HTTP " + resp.status);
    const rows = await resp.json();
    const reversed = [...rows].reverse();
    for (const row of reversed) {
      if (typeof row.shap_json === "string") {
        try { row.shap = JSON.parse(row.shap_json); } catch (_) { row.shap = []; }
      }
      ingestAlert(row, false);
    }
    console.log(`[rubin-skymap] Loaded ${rows.length} historical alerts`);
  } catch (err) {
    console.warn("[rubin-skymap] History load failed:", err);
  }
}

/**
 * Fetch the three sky-asset JSON files in parallel.
 * Gracefully degrades — missing files leave the corresponding array empty.
 */
async function loadSkyAssets() {
  const base = "/static/data/";
  const [starsRes, constsRes, mwRes] = await Promise.allSettled([
    fetch(base + "stars.json"),
    fetch(base + "constellations.json"),
    fetch(base + "milkyway.json"),
  ]);

  if (starsRes.status === "fulfilled" && starsRes.value.ok) {
    try {
      skyStars = await starsRes.value.json();
      console.log(`[rubin-skymap] Loaded ${skyStars.length} stars`);
    } catch (e) {
      console.warn("[rubin-skymap] Failed to parse stars.json:", e);
    }
  } else {
    console.warn("[rubin-skymap] stars.json not available — sky will show without background stars");
  }

  if (constsRes.status === "fulfilled" && constsRes.value.ok) {
    try {
      skyConstellations = await constsRes.value.json();
      console.log(`[rubin-skymap] Loaded ${skyConstellations.length} constellations`);
    } catch (e) {
      console.warn("[rubin-skymap] Failed to parse constellations.json:", e);
    }
  } else {
    console.warn("[rubin-skymap] constellations.json not available");
  }

  if (mwRes.status === "fulfilled" && mwRes.value.ok) {
    try {
      skyMilkyway = await mwRes.value.json();
      console.log(`[rubin-skymap] Loaded Milky Way (${skyMilkyway.points.length} pts)`);
    } catch (e) {
      console.warn("[rubin-skymap] Failed to parse milkyway.json:", e);
    }
  } else {
    console.warn("[rubin-skymap] milkyway.json not available");
  }
}

/* ══════════════════════════════════════════════════════════════
   FILTERS (class legend + sky toggles)
══════════════════════════════════════════════════════════════ */

function initFilters() {
  /* Class filter checkboxes */
  document.querySelectorAll("#legend-bar input[type=checkbox][data-class]").forEach(cb => {
    const item = cb.closest(".legend-item");
    cb.addEventListener("change", () => {
      const cls = cb.dataset.class;
      if (cb.checked) {
        visibleClasses.add(cls);
        item.classList.remove("unchecked");
      } else {
        visibleClasses.delete(cls);
        item.classList.add("unchecked");
      }
      applyListFilter();
      redrawMap();
    });
  });

  /* Sky layer toggle checkboxes */
  const toggleMap = {
    "toggle-milkyway":       "milkyway",
    "toggle-stars":          "stars",
    "toggle-constellations": "constellations",
    "toggle-labels":         "labels",
  };
  for (const [id, key] of Object.entries(toggleMap)) {
    const cb = document.getElementById(id);
    if (!cb) continue;
    /* Sync initial state from HTML checked attribute */
    skyToggles[key] = cb.checked;
    cb.addEventListener("change", () => {
      skyToggles[key] = cb.checked;
      redrawMap();
    });
  }

  /* Search box */
  const searchEl = document.getElementById("feed-search");
  if (searchEl) {
    searchEl.addEventListener("input", () => {
      searchQuery = searchEl.value.trim();
      applyListFilter();
    });
  }
}

/* ══════════════════════════════════════════════════════════════
   CANVAS RESIZE & CLICK
══════════════════════════════════════════════════════════════ */

function resizeCanvas() {
  const container = document.getElementById("map-container");
  if (!container) return;
  /* Use devicePixelRatio for crisp rendering on HiDPI displays */
  const dpr = window.devicePixelRatio || 1;
  const cw  = container.clientWidth;
  const ch  = container.clientHeight;
  skyCanvas.width  = cw * dpr;
  skyCanvas.height = ch * dpr;
  skyCanvas.style.width  = cw + "px";
  skyCanvas.style.height = ch + "px";
  skyCtx.scale(dpr, dpr);
  redrawMap();
}

function initMapInteraction() {
  /* Click → select alert */
  skyCanvas.addEventListener("click", evt => {
    const rect = skyCanvas.getBoundingClientRect();
    const cx   = evt.clientX - rect.left;
    const cy   = evt.clientY - rect.top;
    const hit  = hitTest(cx, cy);
    if (hit) selectAlert(hit);
  });

  /* Hover → tooltip + coords */
  const tooltipEl  = document.getElementById("map-tooltip");
  const coordsEl   = document.getElementById("map-coords");

  skyCanvas.addEventListener("mousemove", evt => {
    const rect = skyCanvas.getBoundingClientRect();
    const cx   = evt.clientX - rect.left;
    const cy   = evt.clientY - rect.top;
    const W    = rect.width;
    const H    = rect.height;

    /* Coords readout */
    const { ra, dec } = px2sky(cx, cy, W, H);
    if (ra >= 0 && ra <= 24 && dec >= -90 && dec <= 90) {
      coordsEl.textContent = `RA ${ra.toFixed(2)}h  Dec ${dec.toFixed(2)}°`;
    } else {
      coordsEl.textContent = "";
    }

    /* Tooltip on hover */
    const hit = hitTest(cx, cy);
    if (hit) {
      const col  = CLASS_COLORS[hit.predicted_class] || "#888";
      const prob = ((hit.predicted_prob || 0) * 100).toFixed(1);
      tooltipEl.innerHTML = `
        <span style="color:${col}">&#9679;</span>
        <strong>${hit.object_id}</strong>
        &nbsp;|&nbsp;${hit.predicted_class}
        &nbsp;<span style="color:#7d8590">${prob}%</span>
      `;
      tooltipEl.style.display = "block";
      tooltipEl.style.left    = (cx + 14) + "px";
      tooltipEl.style.top     = (cy - 10) + "px";

      /* Keep tooltip within bounds */
      const tw = tooltipEl.offsetWidth;
      if (cx + 14 + tw > W) tooltipEl.style.left = (cx - tw - 8) + "px";
    } else {
      tooltipEl.style.display = "none";
    }
  });

  skyCanvas.addEventListener("mouseleave", () => {
    if (tooltipEl) tooltipEl.style.display = "none";
    if (coordsEl)  coordsEl.textContent = "";
  });
}

/* ══════════════════════════════════════════════════════════════
   REAL-TIME UTC CLOCK
══════════════════════════════════════════════════════════════ */

/**
 * Start the real-time UTC clock in the header.
 * Ticks every 500 ms for smooth colon blink; updates DOM strings once per second.
 */
function initClock() {
  const dateEl = document.getElementById("clock-date");
  const timeEl = document.getElementById("clock-time");
  if (!dateEl || !timeEl) return;

  /* Pad to two digits */
  const p = n => String(n).padStart(2, "0");

  let lastSec = -1;

  function tick() {
    const now = new Date();
    const sec = now.getUTCSeconds();

    /* Update time string once per second */
    if (sec !== lastSec) {
      lastSec = sec;

      /* Date: e.g. "2025-09-16" */
      const dateStr = `${now.getUTCFullYear()}-${p(now.getUTCMonth() + 1)}-${p(now.getUTCDate())}`;
      dateEl.textContent = dateStr;

      /* Time: HH:MM:SS */
      timeEl.textContent = `${p(now.getUTCHours())}:${p(now.getUTCMinutes())}:${p(sec)}`;

      /* brief accent colour flash each second */
      timeEl.classList.add("tick");
      setTimeout(() => timeEl.classList.remove("tick"), 120);
    }
  }

  tick();
  setInterval(tick, 500);
}

/* ══════════════════════════════════════════════════════════════
   BOOT
══════════════════════════════════════════════════════════════ */

document.addEventListener("DOMContentLoaded", () => {
  skyCanvas = document.getElementById("skymap");
  skyCtx    = skyCanvas.getContext("2d");
  tooltip   = document.getElementById("map-tooltip");

  initClock();
  initFilters();
  resizeCanvas();
  initMapInteraction();

  window.addEventListener("resize", () => {
    /* Re-scale without losing DPR transform */
    const dpr = window.devicePixelRatio || 1;
    skyCtx.setTransform(1, 0, 0, 1, 0, 0);
    resizeCanvas();
  });

  /* Load sky assets, draw immediately, then load history and connect WS */
  loadSkyAssets().then(() => {
    redrawMap();
    return loadHistory();
  }).then(() => connectWS());
});

/* ══════════════════════════════════════════════════════════════
   CLASS EXPLAINABILITY
══════════════════════════════════════════════════════════════ */

/**
 * Plain-English one-liner + key observational signatures for each class.
 * Rendered inline in the inspector panel when an alert is selected.
 */
const CLASS_EXPLAIN = {
  "SN Ia": {
    tagline: "Type Ia Supernova — thermonuclear white-dwarf explosion",
    detail:  "Used as a standard candle for measuring cosmic distances and dark energy.",
    bullets: ["Fast rise ~15–20 days, slow 60-day decline", "No hydrogen in spectrum", "Peak ~19–20 mag at cosmological distances"],
  },
  "SN II": {
    tagline: "Type II Supernova — core-collapse of a massive star",
    detail:  "Star retains its hydrogen envelope, producing a distinctive 80-day plateau.",
    bullets: ["Light-curve plateau powered by hydrogen recombination", "Slower rise and dimmer than SN Ia", "Progenitor mass > 8 M\u2609"],
  },
  "SN Ibc": {
    tagline: "Type Ib/c Supernova — stripped-envelope core collapse",
    detail:  "Star shed its outer layers before exploding; related to gamma-ray bursts.",
    bullets: ["No hydrogen (Ib) or helium (Ic) in spectrum", "Faster rise and decline than SN II", "Intermediate peak brightness"],
  },
  "SLSN": {
    tagline: "Superluminous Supernova — up to 100\u00d7 brighter than ordinary SNe",
    detail:  "Powered by a magnetar or circumstellar interaction. Very rare, very bright.",
    bullets: ["Slow broad light curve spanning months", "Peak ~17–18 mag", "Traces extreme stellar physics at high redshift"],
  },
  "Kilonova": {
    tagline: "Kilonova — neutron-star merger, forge of heavy elements",
    detail:  "r-process nucleosynthesis produces gold, platinum, and lanthanides.",
    bullets: ["Extremely fast: rises and fades in 2–5 days", "Very red colour due to lanthanide opacity", "First confirmed: GW170817"],
  },
  "AGN": {
    tagline: "Active Galactic Nucleus — accreting supermassive black hole",
    detail:  "Persistent, stochastically variable over years to decades. Key survey contaminant.",
    bullets: ["Never fully fades — distinguishes it from supernovae", "Long time-span and high n_detections", "Broad emission lines in spectrum"],
  },
  "RRL": {
    tagline: "RR Lyrae Variable Star — pulsating distance indicator",
    detail:  "Old, low-mass star that expands and contracts with a very regular period.",
    bullets: ["Period 0.2–1 day, amplitude ~0.5 mag", "Sinusoidal, repeating light curve", "Used to map Milky Way structure"],
  },
  "Other": {
    tagline: "Other / Uncertain — not confidently matched to a known class",
    detail:  "May be a TDE, microlensing event, cataclysmic variable, eclipsing binary, or artefact.",
    bullets: ["Model confidence below discrimination threshold", "Consider spectroscopic follow-up", "Could be a rare or novel transient type"],
  },
};

/**
 * Render a small inline explainability card in #class-explain-box.
 * @param {string} cls - predicted class name
 */
function _renderClassExplain(cls) {
  const box = document.getElementById("class-explain-box");
  if (!box) return;

  const info = CLASS_EXPLAIN[cls] || CLASS_EXPLAIN["Other"];
  const col  = CLASS_COLORS[cls]  || CLASS_COLORS["Other"];

  box.innerHTML = `
    <div class="ceb-card" style="border-left-color:${col}">
      <div class="ceb-tagline" style="color:${col}">${info.tagline}</div>
      <div class="ceb-detail">${info.detail}</div>
      <ul class="ceb-bullets">
        ${info.bullets.map(b => `<li>${b}</li>`).join("")}
      </ul>
    </div>
  `;
}
