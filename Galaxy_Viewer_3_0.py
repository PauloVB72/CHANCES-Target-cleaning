#!/usr/bin/env python3
# =============================================================================
# Galaxy Classifier — CHANCES Project
# Copyright (c) 2025 CHANCES Collaboration
# License: MIT
#
# Galaxy Viewer 3.0 — Standalone data exploration tool.
# NOT part of the training pipeline. Run with:
#   streamlit run Galaxy_Viewer_3_0.py
# =============================================================================
"""
Galaxy Viewer 3.0
=================
Interactive visualisation tool for Galaxy Classifier predictions.

Features
--------
• Image Viewer   — Browse & filter predictions with probability bars.
• Group Analysis — Friends-of-friends clustering with an interactive
                   RA/Dec scatter plot: hover a point → see the galaxy
                   image + probability breakdown in a floating panel.
• Filter & Match — Build custom multi-condition filters and export results.

Usage
-----
    streamlit run Galaxy_Viewer_3_0.py
"""

import base64
import json
import os
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageDraw
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

# ── Page config (must be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="Galaxy Viewer · CHANCES",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS — space observatory aesthetic ───────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;600;800&display=swap');

/* Root palette */
:root {
    --bg-deep:    #060b14;
    --bg-card:    #0d1525;
    --bg-panel:   #111e33;
    --accent-c:   #00d4ff;   /* cyan  */
    --accent-g:   #ffd166;   /* gold  */
    --accent-r:   #ff5e84;   /* rose  */
    --text-pri:   #e8f4f8;
    --text-sec:   #7d9ab5;
    --border:     rgba(0,212,255,.18);
}

/* App background */
.stApp { background: var(--bg-deep); }
section[data-testid="stSidebar"] {
    background: var(--bg-card);
    border-right: 1px solid var(--border);
}

/* Typography */
h1, h2, h3, h4 { font-family: 'Syne', sans-serif !important; color: var(--text-pri) !important; }
p, span, label, div { font-family: 'DM Mono', monospace !important; color: var(--text-pri) !important; }
.stMarkdown p { color: var(--text-sec) !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card);
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
    border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 7px;
    color: var(--text-sec) !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600;
    letter-spacing: .03em;
    padding: 6px 18px;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #0a2a4a, #0d3d6b) !important;
    color: var(--accent-c) !important;
    border: 1px solid var(--border) !important;
}

/* Metrics */
[data-testid="metric-container"] {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 18px !important;
}
[data-testid="stMetricValue"] { color: var(--accent-c) !important; font-size: 1.6rem !important; }
[data-testid="stMetricLabel"] { color: var(--text-sec) !important; font-size: .8rem !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #0a2a4a, #0d3d6b);
    border: 1px solid var(--border);
    color: var(--accent-c) !important;
    border-radius: 8px;
    font-family: 'DM Mono', monospace;
    font-size: .82rem;
    transition: all .2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #0d3d6b, #0f4f87);
    border-color: var(--accent-c);
    box-shadow: 0 0 12px rgba(0,212,255,.3);
}
.stButton [data-testid="baseButton-primary"] > button {
    background: linear-gradient(135deg, #00b4d8, #0096c7);
    color: #060b14 !important;
}

/* Image containers */
.img-card {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 10px;
    margin-bottom: 14px;
    transition: border-color .2s;
}
.img-card:hover { border-color: var(--accent-c); }

/* Probability badge */
.prob-badge {
    display:inline-block;
    background: rgba(0,212,255,.12);
    border: 1px solid var(--accent-c);
    border-radius: 6px;
    padding: 2px 8px;
    font-size: .78rem;
    color: var(--accent-c) !important;
    margin: 2px 0;
}

/* Galaxy ID badge */
.gal-id {
    font-family: 'DM Mono', monospace;
    font-size: .75rem;
    color: var(--accent-g) !important;
    background: rgba(255,209,102,.08);
    border: 1px solid rgba(255,209,102,.3);
    border-radius: 5px;
    padding: 2px 7px;
    display: inline-block;
    margin-bottom: 4px;
}

/* DataFrames */
[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 10px; }

/* Sidebar headers */
.sidebar-section {
    border-top: 1px solid var(--border);
    margin: 12px 0 6px;
    padding-top: 10px;
    color: var(--accent-c);
    font-family: 'Syne', sans-serif;
    font-size: .85rem;
    letter-spacing: .06em;
    text-transform: uppercase;
}

/* Info / warning boxes */
.stAlert { border-radius: 10px !important; border-left: 3px solid var(--accent-c) !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

PLACEHOLDER_COLOR = "#0d1525"


def load_image(path: str, size: int | None = None) -> Image.Image:
    """Load an image from disk; return a dark placeholder on failure."""
    try:
        if path and os.path.exists(path):
            img = Image.open(path).convert("RGB")
            if size:
                img = img.resize((size, size), Image.LANCZOS)
            return img
    except Exception:
        pass
    ph = Image.new("RGB", (size or 256, size or 256), color=PLACEHOLDER_COLOR)
    draw = ImageDraw.Draw(ph)
    cx, cy = (size or 256) // 2, (size or 256) // 2
    r = (size or 256) // 4
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline="#1e3a5f", width=2)
    return ph


def image_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    """Convert a PIL image to a base64 data-URI string."""
    buf = BytesIO()
    img.save(buf, format=fmt)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def draw_red_circle(img: Image.Image, radius: int = 20) -> Image.Image:
    img = img.copy()
    draw = ImageDraw.Draw(img)
    cx, cy = img.width // 2, img.height // 2
    draw.ellipse(
        (cx - radius, cy - radius, cx + radius, cy + radius),
        outline="#ff5e84",
        width=3,
    )
    return img


def plot_probability_bars(probs, class_names, highlight_index=None):
    """Horizontal probability bar chart (dark theme)."""
    fig, ax = plt.subplots(figsize=(4, 2.2), facecolor="#0d1525")
    ax.set_facecolor("#0d1525")
    y = np.arange(len(class_names))
    colors = ["#1e3a5f"] * len(class_names)
    if highlight_index is not None:
        colors[highlight_index] = "#00d4ff"
    bars = ax.barh(y, probs, color=colors, alpha=0.9, edgecolor="none", height=0.6)
    for bar, prob in zip(bars, probs):
        ax.text(
            min(prob + 0.02, 0.98), bar.get_y() + bar.get_height() / 2,
            f"{prob:.1%}", va="center", ha="left", fontsize=7.5, color="#7d9ab5"
        )
    ax.set_yticks(y)
    ax.set_yticklabels(class_names, fontsize=8, color="#e8f4f8")
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability", fontsize=8, color="#7d9ab5")
    ax.tick_params(axis="x", labelsize=7.5, colors="#7d9ab5")
    ax.tick_params(axis="y", colors="#e8f4f8")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.xaxis.label.set_color("#7d9ab5")
    ax.grid(axis="x", color="#1e3a5f", alpha=0.5, linewidth=0.5)
    fig.tight_layout(pad=0.5)
    return fig


def plot_global_distribution(df, class_name, threshold):
    fig, ax = plt.subplots(figsize=(10, 3.5), facecolor="#0d1525")
    ax.set_facecolor("#0d1525")
    ax.hist(df[class_name], bins=50, alpha=0.8, color="#1e6091", edgecolor="none")
    ax.axvline(threshold, color="#ff5e84", linestyle="--", linewidth=2,
               label=f"Threshold: {threshold:.2f}")
    ax.set_xlabel(f"P({class_name})", color="#7d9ab5", fontsize=10)
    ax.set_ylabel("Count", color="#7d9ab5", fontsize=10)
    ax.set_title(f"Probability distribution — {class_name}", color="#e8f4f8", fontsize=12)
    ax.tick_params(colors="#7d9ab5")
    for spine in ax.spines.values():
        spine.set_color("#1e3a5f")
    ax.legend(framealpha=0.2, facecolor="#0d1525", labelcolor="#e8f4f8")
    ax.grid(axis="y", color="#1e3a5f", alpha=0.4, linewidth=0.5)
    fig.tight_layout()
    return fig


# ── Friends-of-friends utilities ──────────────────────────────────────────

def radec_to_cartesian(ra_deg, dec_deg):
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    c = np.cos(dec)
    return np.column_stack([c * np.cos(ra), c * np.sin(ra), np.sin(dec)])


def find_groups(ra, dec, linking_radius_arcsec):
    chord = 2.0 * np.sin(np.radians(linking_radius_arcsec / 3600.0) / 2.0)
    coords = radec_to_cartesian(ra, dec)
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=chord, output_type="ndarray")
    n = len(ra)
    if len(pairs):
        r, c = pairs[:, 0], pairs[:, 1]
        adj = csr_matrix((np.ones(len(pairs)), (r, c)), shape=(n, n))
        adj = adj + adj.T
    else:
        adj = csr_matrix((n, n))
    _, labels = connected_components(csgraph=adj, directed=False, return_labels=True)
    degrees = np.array(adj.sum(axis=1)).flatten()
    labels[degrees == 0] = -1
    unique = np.unique(labels)
    mapping = {old: i for i, old in enumerate(unique) if old != -1}
    new_labels = np.full_like(labels, -1)
    for old, new in mapping.items():
        new_labels[labels == old] = new
    return new_labels


def assign_center_flags(df, group_col="group_id", aper_col="aper_radius", ratio_thresh=1.2):
    df = df.copy()
    df["center_flag"] = 0
    centers = {}
    for gid in df[group_col].unique():
        if gid == -1:
            continue
        grp = df[df[group_col] == gid]
        if len(grp) < 2:
            continue
        max_idx = grp[aper_col].idxmax()
        others = grp[grp.index != max_idx][aper_col]
        med = np.median(others) if len(others) else 0
        if med > 0 and (grp.loc[max_idx, aper_col] / med) >= ratio_thresh:
            df.loc[max_idx, "center_flag"] = 1
            centers[gid] = max_idx
        else:
            centers[gid] = None
    return df, centers


# ── Interactive group scatter (hover shows galaxy image) ──────────────────

def build_interactive_scatter(
    group_df, ra_col, dec_col, selected_prob_col, threshold,
    prob_cols, center_idx, aper_col, image_root="",
):
    """
    Build a self-contained HTML component:
      - Plotly scatter of RA/Dec positions
      - On hover: floating panel with galaxy thumbnail + probability bars
    Returns the HTML string to pass to st.components.v1.html().
    """
    points = []
    for idx, row in group_df.iterrows():
        # Resolve image path
        path = str(row.get("file_loc", ""))
        if image_root and path and not os.path.isabs(path):
            path = os.path.join(image_root, path)
        img = load_image(path, size=128)
        img_b64 = image_to_base64(img)

        probs = {col: float(row[col]) if col in row else 0.0 for col in prob_cols}
        prob_target = float(row.get(selected_prob_col, 0.0))
        above = prob_target >= threshold
        is_center = center_idx is not None and idx == center_idx
        galaxy_id = str(row.get("id_merge", f"idx_{idx}"))

        points.append({
            "ra": float(row[ra_col]),
            "dec": float(row[dec_col]),
            "id": galaxy_id,
            "prob_target": prob_target,
            "probs": probs,
            "img": img_b64,
            "above": above,
            "center": is_center,
            "aper_deg": float(row.get(aper_col, 0)) / 3600.0,
        })

    data_json = json.dumps(points)
    prob_cols_json = json.dumps(prob_cols)
    threshold_f = float(threshold)
    sel_col = str(selected_prob_col)

    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  body {{
    margin: 0; padding: 0;
    background: #060b14;
    font-family: 'DM Mono', 'Courier New', monospace;
    color: #e8f4f8;
  }}
  #plot {{ width: 100%; height: 520px; }}

  /* Hover tooltip panel */
  #hover-panel {{
    position: fixed;
    display: none;
    z-index: 9999;
    background: rgba(13,21,37,.97);
    border: 1px solid rgba(0,212,255,.5);
    border-radius: 12px;
    padding: 12px;
    min-width: 220px;
    max-width: 260px;
    box-shadow: 0 0 30px rgba(0,212,255,.2);
    pointer-events: none;
  }}
  #hover-panel img {{
    width: 100%;
    border-radius: 8px;
    border: 1px solid rgba(0,212,255,.3);
    margin-bottom: 8px;
  }}
  .hp-id {{
    font-size: 11px;
    color: #ffd166;
    border: 1px solid rgba(255,209,102,.3);
    background: rgba(255,209,102,.07);
    border-radius: 5px;
    padding: 2px 7px;
    display: inline-block;
    margin-bottom: 8px;
  }}
  .hp-row {{
    display: flex;
    align-items: center;
    margin-bottom: 4px;
    gap: 6px;
  }}
  .hp-label {{
    font-size: 10px;
    color: #7d9ab5;
    width: 72px;
    flex-shrink: 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }}
  .hp-bar-bg {{
    flex: 1;
    height: 8px;
    background: #1e3a5f;
    border-radius: 4px;
    overflow: hidden;
  }}
  .hp-bar-fill {{
    height: 100%;
    border-radius: 4px;
    background: #1e6091;
    transition: width .2s;
  }}
  .hp-bar-fill.active {{ background: #00d4ff; }}
  .hp-val {{
    font-size: 10px;
    color: #e8f4f8;
    width: 38px;
    text-align: right;
  }}
  .hp-divider {{ border: none; border-top: 1px solid rgba(0,212,255,.15); margin: 7px 0; }}
</style>
</head>
<body>
<div id="plot"></div>
<div id="hover-panel">
  <div class="hp-id" id="hp-id">—</div>
  <img id="hp-img" src="" alt="galaxy">
  <hr class="hp-divider">
  <div id="hp-probs"></div>
</div>

<script>
const POINTS      = {data_json};
const PROB_COLS   = {prob_cols_json};
const THRESHOLD   = {threshold_f};
const SEL_COL     = "{sel_col}";

// ── Build Plotly traces ──────────────────────────────────────────────────
const normalX=[], normalY=[], normalCustom=[], normalText=[];
const aboveX=[],  aboveY=[],  aboveCustom=[],  aboveText=[];
const centerX=[], centerY=[], centerCustom=[], centerText=[];

// Aperture circles
const circleShapes = [];

POINTS.forEach((p, i) => {{
  const tip = `<b>${{p.id}}</b><br>P(${{SEL_COL}}): ${{(p.prob_target*100).toFixed(1)}}%`;
  if (p.center) {{
    centerX.push(p.ra); centerY.push(p.dec);
    centerCustom.push(i); centerText.push(tip);
  }} else if (p.above) {{
    aboveX.push(p.ra); aboveY.push(p.dec);
    aboveCustom.push(i); aboveText.push(tip);
  }} else {{
    normalX.push(p.ra); normalY.push(p.dec);
    normalCustom.push(i); normalText.push(tip);
  }}

  if (p.aper_deg > 0) {{
    circleShapes.push({{
      type: 'circle',
      xref: 'x', yref: 'y',
      x0: p.ra  - p.aper_deg,
      y0: p.dec - p.aper_deg,
      x1: p.ra  + p.aper_deg,
      y1: p.dec + p.aper_deg,
      line: {{ color: 'rgba(126,166,196,.35)', width: 1, dash: 'dot' }},
      fillcolor: 'rgba(0,0,0,0)',
    }});
  }}
}});

const traceNormal = {{
  x: normalX, y: normalY, customdata: normalCustom,
  mode: 'markers',
  name: 'Below threshold',
  text: normalText, hovertemplate: '%{{text}}<extra></extra>',
  marker: {{ color: '#2d6a9f', size: 9, symbol: 'circle',
             line: {{ color: '#3d8bc4', width: 1 }} }},
}};
const traceAbove = {{
  x: aboveX, y: aboveY, customdata: aboveCustom,
  mode: 'markers',
  name: `P ≥ ${{THRESHOLD.toFixed(2)}}`,
  text: aboveText, hovertemplate: '%{{text}}<extra></extra>',
  marker: {{ color: '#ff5e84', size: 13, symbol: 'star',
             line: {{ color: '#c0392b', width: 1.5 }} }},
}};
const traceCenter = {{
  x: centerX, y: centerY, customdata: centerCustom,
  mode: 'markers',
  name: 'Center',
  text: centerText, hovertemplate: '%{{text}}<extra></extra>',
  marker: {{ color: '#ffd166', size: 18, symbol: 'star',
             line: {{ color: '#f0a500', width: 2 }} }},
}};

const layout = {{
  paper_bgcolor: '#060b14',
  plot_bgcolor:  '#0a1628',
  font: {{ family: 'DM Mono, Courier New', color: '#7d9ab5', size: 11 }},
  xaxis: {{
    title: {{ text: 'RA (deg)', font: {{ color: '#7d9ab5' }} }},
    autorange: 'reversed',
    gridcolor: '#1e3a5f', zerolinecolor: '#1e3a5f',
    tickfont: {{ color: '#7d9ab5' }},
  }},
  yaxis: {{
    title: {{ text: 'Dec (deg)', font: {{ color: '#7d9ab5' }} }},
    gridcolor: '#1e3a5f', zerolinecolor: '#1e3a5f',
    tickfont: {{ color: '#7d9ab5' }},
    scaleanchor: 'x', scaleratio: 1,
  }},
  legend: {{
    bgcolor: 'rgba(13,21,37,.8)',
    bordercolor: 'rgba(0,212,255,.3)', borderwidth: 1,
    font: {{ color: '#e8f4f8' }},
  }},
  shapes: circleShapes,
  margin: {{ l: 60, r: 20, t: 30, b: 60 }},
  hovermode: 'closest',
}};

const config = {{
  responsive: true,
  displayModeBar: true,
  modeBarButtonsToRemove: ['lasso2d','select2d'],
  displaylogo: false,
  toImageButtonOptions: {{
    format: 'png', filename: 'group_position', scale: 2,
  }},
}};

Plotly.newPlot('plot', [traceNormal, traceAbove, traceCenter], layout, config);

// ── Hover panel logic ────────────────────────────────────────────────────
const panel  = document.getElementById('hover-panel');
const hpImg  = document.getElementById('hp-img');
const hpId   = document.getElementById('hp-id');
const hpProbs= document.getElementById('hp-probs');

function showPanel(ptIdx, mouseX, mouseY) {{
  const p = POINTS[ptIdx];
  hpId.textContent = p.id;
  hpImg.src = p.img;

  // Build probability rows
  let html = '';
  PROB_COLS.forEach(col => {{
    const val = p.probs[col] || 0;
    const pct = (val * 100).toFixed(1);
    const isActive = (col === SEL_COL) ? 'active' : '';
    const shortCol = col.length > 11 ? col.slice(0,9) + '…' : col;
    html += `
      <div class="hp-row">
        <span class="hp-label" title="${{col}}">${{shortCol}}</span>
        <div class="hp-bar-bg">
          <div class="hp-bar-fill ${{isActive}}" style="width:${{pct}}%"></div>
        </div>
        <span class="hp-val">${{pct}}%</span>
      </div>`;
  }});
  hpProbs.innerHTML = html;

  // Position panel (avoid going off screen)
  const pw = 260, ph = 320;
  const vw = window.innerWidth, vh = window.innerHeight;
  let left = mouseX + 16, top = mouseY - 60;
  if (left + pw > vw - 10) left = mouseX - pw - 16;
  if (top + ph > vh - 10) top = vh - ph - 10;
  if (top < 10) top = 10;
  panel.style.left = left + 'px';
  panel.style.top  = top  + 'px';
  panel.style.display = 'block';
}}

function hidePanel() {{
  panel.style.display = 'none';
}}

document.getElementById('plot').on('plotly_hover', function(data) {{
  if (!data.points.length) return;
  const pt = data.points[0];
  const ptIdx = pt.customdata;
  if (ptIdx === undefined) return;
  const evt = data.event;
  showPanel(ptIdx, evt.clientX, evt.clientY);
}});

document.getElementById('plot').on('plotly_unhover', hidePanel);

// Keep panel following mouse while hovering
document.getElementById('plot').addEventListener('mousemove', function(evt) {{
  if (panel.style.display === 'block') {{
    const pw = 260, ph = 320;
    const vw = window.innerWidth, vh = window.innerHeight;
    let left = evt.clientX + 16, top = evt.clientY - 60;
    if (left + pw > vw - 10) left = evt.clientX - pw - 16;
    if (top + ph > vh - 10) top = vh - ph - 10;
    if (top < 10) top = 10;
    panel.style.left = left + 'px';
    panel.style.top  = top  + 'px';
  }}
}});
</script>
</body>
</html>
"""
    return html


# ═══════════════════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ═══════════════════════════════════════════════════════════════════════════

defaults = {
    "marked_table":       pd.DataFrame(columns=["id", "type"]),
    "selected_gid":       None,
    "threshold":          0.5,
    "selected_prob_col":  None,
    "df_merged":          None,
    "page_number":        1,
    "df_pred":            None,
    "df_coord":           None,
    "prob_cols":          [],
    "valid_groups":       [],
    "group_sizes":        {},
    "centers":            {},
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ═══════════════════════════════════════════════════════════════════════════
# MARKED TABLE HELPER
# ═══════════════════════════════════════════════════════════════════════════

def mark(img_id, action_type):
    df = st.session_state.marked_table
    if img_id in df["id"].values:
        df.loc[df["id"] == img_id, "type"] = action_type
    else:
        st.session_state.marked_table = pd.concat(
            [df, pd.DataFrame({"id": [img_id], "type": [action_type]})],
            ignore_index=True,
        )


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR — DATA UPLOAD
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 8px 0 16px;'>
      <div style='font-family:Syne,sans-serif; font-size:1.3rem;
                  color:#00d4ff; letter-spacing:.08em; font-weight:800;'>
        🌌 GALAXY VIEWER
      </div>
      <div style='font-size:.7rem; color:#7d9ab5; letter-spacing:.12em;
                  margin-top:3px;'>
        CHANCES PROJECT · v3.0
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">📂 Data Upload</div>', unsafe_allow_html=True)
    pred_file  = st.file_uploader("Predictions CSV", type=["csv"], key="pred_up")
    coord_file = st.file_uploader("Coordinates CSV (optional)", type=["csv"], key="coord_up")
    image_root = st.text_input(
        "Image root directory",
        value="",
        help="If file_loc paths are relative, specify the base folder here.",
        placeholder="/path/to/images/",
    )

    # Load predictions
    if pred_file:
        if (st.session_state.df_pred is None
                or st.session_state.get("_last_pred") != pred_file.name):
            try:
                df = pd.read_csv(pred_file)
                excl = {"id_str", "file_loc", "filename", "label", "id"}
                pcols = [c for c in df.columns if c not in excl]
                if not pcols:
                    pcols = [c for c in df.select_dtypes(include=[np.number]).columns
                             if c not in excl]
                st.session_state.df_pred   = df
                st.session_state.prob_cols = pcols
                st.session_state._last_pred = pred_file.name
                st.success(f"Loaded {len(df):,} rows · {len(pcols)} class columns")
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.session_state.df_pred   = None
        st.session_state.prob_cols = []

    # Load coordinates
    if coord_file:
        if (st.session_state.df_coord is None
                or st.session_state.get("_last_coord") != coord_file.name):
            try:
                st.session_state.df_coord  = pd.read_csv(coord_file)
                st.session_state._last_coord = coord_file.name
                st.success(f"Coordinates: {len(st.session_state.df_coord):,} rows")
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.session_state.df_coord = None


# ═══════════════════════════════════════════════════════════════════════════
# MAIN TABS
# ═══════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3 = st.tabs([
    "🖼️  Image Viewer",
    "🌠  Group Analysis",
    "📋  Filter & Match",
])


# ───────────────────────────────────────────────────────────────────────────
# TAB 1 — IMAGE VIEWER
# ───────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("## Prediction Explorer")

    if st.session_state.df_pred is None:
        st.info("Upload a predictions CSV in the sidebar to begin.")
    else:
        df_pred   = st.session_state.df_pred
        prob_cols = st.session_state.prob_cols

        if not prob_cols:
            st.error("No probability columns detected in the uploaded file.")
            st.stop()

        # ── Sidebar controls ────────────────────────────────────────────
        with st.sidebar:
            st.markdown('<div class="sidebar-section">🖼️ Viewer Settings</div>',
                        unsafe_allow_html=True)

            selected_class = st.selectbox("Class of interest", prob_cols)
            sel_idx        = prob_cols.index(selected_class)

            threshold_img = st.slider(
                f"Min P({selected_class})", 0.0, 1.0, 0.5, 0.01, key="thr_img"
            )

            total_imgs   = int((df_pred[selected_class] >= threshold_img).sum())
            max_pp       = max(total_imgs, 1)
            imgs_per_page = st.number_input(
                "Images per page", 1, max_pp, min(12, max_pp), 1, key="ipp"
            )

            total_pages = max(1, (total_imgs - 1) // imgs_per_page + 1)
            if st.session_state.page_number > total_pages:
                st.session_state.page_number = max(1, total_pages)

            pc, ppage, nc = st.columns([1, 2, 1])
            with pc:
                if st.button("◀", key="prev") and st.session_state.page_number > 1:
                    st.session_state.page_number -= 1
                    st.rerun()
            with ppage:
                st.markdown(
                    f"<div style='text-align:center;font-size:.8rem;color:#7d9ab5;'>"
                    f"Page {st.session_state.page_number} / {total_pages}</div>",
                    unsafe_allow_html=True,
                )
            with nc:
                if st.button("▶", key="next") and st.session_state.page_number < total_pages:
                    st.session_state.page_number += 1
                    st.rerun()

            sort_order = st.radio("Sort", ["↓ Highest", "↑ Lowest"], key="sort")
            cols_per_row = st.slider("Columns", 2, 5, 3, key="ncols")

        # ── Metrics ─────────────────────────────────────────────────────
        m1, m2, m3 = st.columns(3)
        m1.metric("Total images", f"{len(df_pred):,}")
        m2.metric(f"P({selected_class}) ≥ {threshold_img:.2f}", f"{total_imgs:,}")
        if "label" in df_pred.columns:
            acc = (df_pred[prob_cols].values.argmax(axis=1) == df_pred["label"].values).mean()
            m3.metric("Global accuracy", f"{acc:.2%}")

        # ── Probability histogram ────────────────────────────────────────
        with st.expander("📊 Global probability distribution", expanded=False):
            st.pyplot(plot_global_distribution(df_pred, selected_class, threshold_img))
            plt.close()

        # ── Filter & paginate ────────────────────────────────────────────
        df_f = df_pred[df_pred[selected_class] >= threshold_img].copy()
        df_f = df_f.sort_values(selected_class, ascending=(sort_order == "↑ Lowest"))

        s = (st.session_state.page_number - 1) * imgs_per_page
        df_page = df_f.iloc[s : s + imgs_per_page]

        if df_page.empty:
            st.warning("No images match the current filter.")
        else:
            if "file_loc" in df_page.columns:
                df_page = df_page.copy()
                df_page["full_path"] = df_page["file_loc"].apply(
                    lambda x: os.path.join(image_root, x)
                    if image_root and not os.path.isabs(str(x))
                    else x
                )
            else:
                df_page["full_path"] = None

            grid = st.columns(cols_per_row)
            for i, (idx, row) in enumerate(df_page.iterrows()):
                with grid[i % cols_per_row]:
                    img = load_image(str(row.get("full_path", "")))
                    gal_id = str(row.get("id_str", f"img_{idx}"))

                    # Marking status
                    mt = None
                    if gal_id in st.session_state.marked_table["id"].values:
                        mt = st.session_state.marked_table.loc[
                            st.session_state.marked_table["id"] == gal_id, "type"
                        ].values[0]

                    # Galaxy ID badge + image
                    st.markdown(
                        f'<div class="gal-id">🆔 {gal_id}</div>',
                        unsafe_allow_html=True,
                    )
                    st.image(img, use_container_width=True)

                    # Probability badge
                    st.markdown(
                        f'<div class="prob-badge">P({selected_class}) = {row[selected_class]:.2%}</div>',
                        unsafe_allow_html=True,
                    )

                    # True label
                    if "label" in row and pd.notna(row["label"]):
                        true_cls = (
                            prob_cols[int(row["label"])]
                            if int(row["label"]) < len(prob_cols)
                            else f"class_{int(row['label'])}"
                        )
                        st.caption(f"True: {true_cls}")

                    # Probability bars
                    fig = plot_probability_bars(
                        row[prob_cols].values.astype(float), prob_cols, sel_idx
                    )
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

                    # Action buttons
                    ba, br, bc = st.columns(3)
                    with ba:
                        if st.button("➕", key=f"add_{gal_id}", help="Add"):
                            mark(gal_id, "add"); st.rerun()
                    with br:
                        if st.button("➖", key=f"rem_{gal_id}", help="Remove"):
                            mark(gal_id, "remove"); st.rerun()
                    with bc:
                        if st.button("🔴", key=f"ctr_{gal_id}", help="Mark as center"):
                            mark(gal_id, "centro"); st.rerun()

                    if mt:
                        colour = {"add": "#00d4ff", "remove": "#ff5e84", "centro": "#ffd166"}.get(mt, "#7d9ab5")
                        st.markdown(
                            f'<div style="font-size:.75rem;color:{colour};">● {mt}</div>',
                            unsafe_allow_html=True,
                        )
                    st.markdown("---")

        # ── Marked table ─────────────────────────────────────────────────
        with st.expander("📝 Marked table", expanded=False):
            st.dataframe(st.session_state.marked_table, use_container_width=True)
            if st.button("🗑️ Clear marks"):
                st.session_state.marked_table = pd.DataFrame(columns=["id", "type"])
                st.rerun()


# ───────────────────────────────────────────────────────────────────────────
# TAB 2 — GROUP ANALYSIS
# ───────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("## Friends-of-Friends Group Analysis")

    if st.session_state.df_pred is None or st.session_state.df_coord is None:
        st.info("Upload **both** a predictions CSV and a coordinates CSV in the sidebar.")
    else:
        df_pred   = st.session_state.df_pred
        df_coord  = st.session_state.df_coord
        prob_cols = st.session_state.prob_cols

        with st.sidebar:
            st.markdown('<div class="sidebar-section">🌠 Group Settings</div>',
                        unsafe_allow_html=True)
            id_pred  = st.text_input("ID column — predictions", "id_str",  key="gid_pred")
            id_coord = st.text_input("ID column — coordinates", "OBJID",   key="gid_coord")
            ra_col   = st.text_input("RA column",      "RA_J2000",  key="gra")
            dec_col  = st.text_input("Dec column",     "Dec_J2000", key="gdec")
            aper_col = st.text_input("Aperture column","aper_radius",key="gaper")
            default_aper   = st.number_input("Default aperture (arcsec)", 0.1, 100.0, 1.0, 0.1)
            linking_radius = st.slider("Linking radius (arcsec)", 1.0, 120.0, 30.0, 1.0)
            min_size       = st.slider("Min group size", 2, 20, 5, 1)
            ratio_thr      = st.slider("Center ratio threshold", 1.0, 3.0, 1.2, 0.05)
            show_thumbs    = st.checkbox("Show thumbnails", value=True)

            if st.button("🚀 Process groups", type="primary"):
                dfp = df_pred.rename(columns={id_pred: "id_merge"})
                dfc = df_coord.rename(columns={id_coord: "id_merge"})
                merged = pd.merge(dfp, dfc, on="id_merge", how="inner")
                if merged.empty:
                    st.error("Merge produced no results — check ID column names.")
                else:
                    if aper_col not in merged.columns:
                        merged[aper_col] = default_aper
                    else:
                        merged[aper_col] = pd.to_numeric(
                            merged[aper_col], errors="coerce"
                        ).fillna(default_aper)
                    merged = merged.dropna(subset=[ra_col, dec_col, aper_col])

                    with st.spinner("Clustering sources …"):
                        merged["group_id"] = find_groups(
                            merged[ra_col].values,
                            merged[dec_col].values,
                            linking_radius,
                        )

                    merged, centers = assign_center_flags(
                        merged, aper_col=aper_col, ratio_thresh=ratio_thr
                    )

                    valid = []
                    for gid in merged["group_id"].unique():
                        if gid == -1:
                            continue
                        sz = (merged["group_id"] == gid).sum()
                        if sz < min_size:
                            merged.loc[merged["group_id"] == gid, "group_id"] = -1
                        else:
                            valid.append(gid)

                    gsizes = {g: (merged["group_id"] == g).sum() for g in valid}
                    valid  = sorted(valid, key=lambda x: gsizes[x], reverse=True)

                    st.session_state.df_merged   = merged
                    st.session_state.centers     = centers
                    st.session_state.valid_groups = valid
                    st.session_state.group_sizes  = gsizes
                    st.success(f"Found {len(valid)} groups (≥ {min_size} members)")

        if st.session_state.df_merged is not None:
            df_merged   = st.session_state.df_merged
            centers     = st.session_state.centers
            valid_groups = st.session_state.valid_groups
            prob_cols   = st.session_state.prob_cols

            with st.sidebar:
                st.markdown('<div class="sidebar-section">🎚️ Probability Filter</div>',
                            unsafe_allow_html=True)
                sel_prob_col = st.selectbox(
                    "Probability column", prob_cols,
                    index=prob_cols.index(st.session_state.selected_prob_col)
                    if st.session_state.selected_prob_col in prob_cols else 0,
                    key="grp_pcol",
                )
                st.session_state.selected_prob_col = sel_prob_col
                threshold = st.slider(
                    f"Min P({sel_prob_col})", 0.0, 1.0,
                    st.session_state.threshold, 0.01, key="grp_thr"
                )
                st.session_state.threshold = threshold

            # Compute group stats
            grp_stats = []
            for gid in valid_groups:
                gdf = df_merged[df_merged["group_id"] == gid]
                cmask = gdf["center_flag"] == 1
                ra_c  = gdf.loc[cmask.idxmax(), ra_col]  if cmask.any() else gdf[ra_col].median()
                dec_c = gdf.loc[cmask.idxmax(), dec_col] if cmask.any() else gdf[dec_col].median()
                n_above = (gdf[sel_prob_col] >= threshold).sum()
                grp_stats.append({
                    "Group ID": gid,
                    "Size": len(gdf),
                    f"N ≥ {threshold:.2f}": n_above,
                    "Below": len(gdf) - n_above,
                    "Fraction Above": n_above / len(gdf),
                    "RA_center": ra_c,
                    "Dec_center": dec_c,
                    "Has Center": cmask.any(),
                })
            df_groups = pd.DataFrame(grp_stats)

            smap, sindiv, sstats = st.tabs([
                "🌍 Group Map",
                "🔍 Individual Explorer",
                "📊 Statistics",
            ])

            # ── GROUP MAP ──────────────────────────────────────────────
            with smap:
                if df_groups.empty:
                    st.warning("No groups to display.")
                else:
                    col_name = f"N ≥ {threshold:.2f}"
                    fig_map = px.scatter(
                        df_groups,
                        x="RA_center", y="Dec_center",
                        size="Size",
                        color=col_name,
                        hover_data={c: True for c in df_groups.columns
                                    if c not in ["RA_center", "Dec_center"]},
                        color_continuous_scale="Plasma",
                        title=f"Groups — P({sel_prob_col}) ≥ {threshold:.2f}",
                        labels={"RA_center": "RA", "Dec_center": "Dec", col_name: col_name},
                        template="plotly_dark",
                    )
                    fig_map.update_layout(
                        paper_bgcolor="#060b14",
                        plot_bgcolor="#0a1628",
                        font=dict(family="DM Mono", color="#7d9ab5"),
                    )
                    fig_map.update_xaxes(autorange="reversed")
                    st.plotly_chart(fig_map, use_container_width=True)

                    st.markdown("#### Group summary")
                    st.dataframe(
                        df_groups.sort_values(col_name, ascending=False),
                        use_container_width=True,
                    )

            # ── INDIVIDUAL EXPLORER ────────────────────────────────────
            with sindiv:
                if not valid_groups:
                    st.warning("No groups available.")
                else:
                    if st.session_state.selected_gid not in valid_groups:
                        st.session_state.selected_gid = valid_groups[0]

                    gopts = {
                        f"Group {g}  ·  {st.session_state.group_sizes[g]} members": g
                        for g in valid_groups
                    }
                    sel_label = st.selectbox(
                        "Select group",
                        list(gopts.keys()),
                        index=list(gopts.values()).index(st.session_state.selected_gid),
                    )
                    selected_gid = gopts[sel_label]
                    st.session_state.selected_gid = selected_gid

                    group_df   = df_merged[df_merged["group_id"] == selected_gid].copy()
                    center_idx = centers.get(selected_gid)

                    n_above = (group_df[sel_prob_col] >= threshold).sum()
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Members",         len(group_df))
                    c2.metric(f"P ≥ {threshold:.2f}", n_above)
                    c3.metric("Fraction",        f"{n_above/len(group_df):.1%}")
                    c4.metric("Center detected", "✅" if center_idx else "❌")

                    # ── Interactive scatter with hover-image ──────────────
                    st.markdown("#### Position plot")
                    st.caption(
                        "🖱️  Hover over any point to preview the galaxy image "
                        "and probability breakdown."
                    )

                    html_component = build_interactive_scatter(
                        group_df=group_df,
                        ra_col=ra_col,
                        dec_col=dec_col,
                        selected_prob_col=sel_prob_col,
                        threshold=threshold,
                        prob_cols=prob_cols,
                        center_idx=center_idx,
                        aper_col=aper_col,
                        image_root=image_root,
                    )

                    import streamlit.components.v1 as components
                    components.html(html_component, height=540, scrolling=False)

                    # ── Member thumbnails ─────────────────────────────────
                    if show_thumbs and "file_loc" in group_df.columns:
                        st.markdown("#### Thumbnails")
                        group_df = group_df.copy()
                        group_df["full_path"] = group_df["file_loc"].apply(
                            lambda x: os.path.join(image_root, x)
                            if image_root and not os.path.isabs(str(x)) else x
                        )
                        tcols = st.columns(4)
                        for i, (idx, row) in enumerate(group_df.iterrows()):
                            gal_id = str(row.get("id_merge", f"idx_{idx}"))
                            pval   = row[sel_prob_col]
                            marked = gal_id in st.session_state.marked_table["id"].values
                            with tcols[i % 4]:
                                st.markdown(
                                    f'<div class="gal-id">🆔 {gal_id}</div>',
                                    unsafe_allow_html=True,
                                )
                                img = load_image(str(row["full_path"]), size=128)
                                st.image(img, use_container_width=True)
                                st.markdown(
                                    f'<div class="prob-badge">P = {pval:.2%}</div>',
                                    unsafe_allow_html=True,
                                )
                                ta, tr = st.columns(2)
                                with ta:
                                    if st.button("➕", key=f"tadd_{gal_id}"):
                                        mark(gal_id, "add"); st.rerun()
                                with tr:
                                    if st.button("➖", key=f"trem_{gal_id}"):
                                        mark(gal_id, "remove"); st.rerun()
                                if marked:
                                    st.caption("✓ marked")

                    # ── Member table ──────────────────────────────────────
                    st.markdown("#### Member data")
                    disp = ["id_merge", ra_col, dec_col, aper_col, sel_prob_col, "center_flag"]
                    disp = [c for c in disp if c in group_df.columns]
                    st.dataframe(
                        group_df[disp].sort_values(sel_prob_col, ascending=False),
                        use_container_width=True,
                    )

            # ── STATISTICS ────────────────────────────────────────────
            with sstats:
                if df_groups.empty:
                    st.warning("No groups.")
                else:
                    col_name = f"N ≥ {threshold:.2f}"
                    s1, s2 = st.columns(2)

                    def dark_hist(values, xlabel, color):
                        fig, ax = plt.subplots(figsize=(5, 3), facecolor="#0d1525")
                        ax.set_facecolor("#0a1628")
                        ax.hist(values, bins=20, color=color, edgecolor="none", alpha=0.85)
                        ax.set_xlabel(xlabel, color="#7d9ab5", fontsize=9)
                        ax.set_ylabel("Groups", color="#7d9ab5", fontsize=9)
                        ax.tick_params(colors="#7d9ab5")
                        for sp in ax.spines.values():
                            sp.set_color("#1e3a5f")
                        ax.grid(axis="y", color="#1e3a5f", alpha=0.4)
                        fig.tight_layout()
                        return fig

                    with s1:
                        fig = dark_hist(df_groups[col_name], f"N with P ≥ {threshold:.2f}", "#ff5e84")
                        st.pyplot(fig); plt.close(fig)
                    with s2:
                        fig = dark_hist(df_groups["Fraction Above"], "Fraction above threshold", "#2d6a9f")
                        st.pyplot(fig); plt.close(fig)

                    st.markdown("#### Descriptive statistics")
                    st.dataframe(df_groups.describe(), use_container_width=True)


# ───────────────────────────────────────────────────────────────────────────
# TAB 3 — FILTER & MATCH
# ───────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("## Filter & Match Objects")

    if st.session_state.df_pred is None:
        st.info("Upload a predictions CSV in the sidebar to use this tab.")
    else:
        df_pred   = st.session_state.df_pred
        prob_cols = st.session_state.prob_cols

        # Detect ID column
        id_col = next(
            (c for c in ["id_str", "id", "OBJID"] if c in df_pred.columns), None
        )
        if id_col is None:
            st.error("No ID column found ('id_str', 'id', or 'OBJID').")
            st.stop()

        st.markdown("### Define filter conditions")

        combine_mode = st.radio(
            "Combine conditions with",
            ["AND — all must match", "OR — at least one must match"],
            horizontal=True,
        )

        conditions = {}
        for col in prob_cols:
            ca, cb = st.columns([1, 3])
            with ca:
                use = st.checkbox(f"**{col}**", value=True, key=f"use_{col}")
            with cb:
                if use:
                    cond = st.selectbox(
                        "Condition", ["any", ">", ">=", "<", "<=", "="],
                        key=f"cond_{col}"
                    )
                    if cond != "any":
                        thr = st.number_input(
                            "Threshold", 0.0, 1.0, 0.5, 0.01, key=f"thr_{col}"
                        )
                        conditions[col] = (cond, thr)
                    else:
                        conditions[col] = ("any", None)
                else:
                    conditions[col] = ("ignore", None)

        if st.button("🔍 Generate filtered table", type="primary"):
            active = [
                (col, cond, val)
                for col, (cond, val) in conditions.items()
                if cond not in ("ignore", "any")
            ]

            if not active:
                mask = pd.Series([True] * len(df_pred))
            elif combine_mode.startswith("AND"):
                mask = pd.Series([True] * len(df_pred))
                for col, cond, val in active:
                    ops = {">": df_pred[col] > val, ">=": df_pred[col] >= val,
                           "<": df_pred[col] < val, "<=": df_pred[col] <= val,
                           "=": df_pred[col] == val}
                    mask &= ops[cond]
            else:
                mask = pd.Series([False] * len(df_pred))
                for col, cond, val in active:
                    ops = {">": df_pred[col] > val, ">=": df_pred[col] >= val,
                           "<": df_pred[col] < val, "<=": df_pred[col] <= val,
                           "=": df_pred[col] == val}
                    mask |= ops[cond]

            st.session_state.filtered_table = df_pred[mask].copy()
            st.success(f"Filtered table: **{len(st.session_state.filtered_table):,}** rows")

        if "filtered_table" in st.session_state:
            ft = st.session_state.filtered_table

            st.markdown("### Filtered table")
            st.dataframe(ft, use_container_width=True)

            st.markdown("### Marked table")
            st.dataframe(st.session_state.marked_table, use_container_width=True)

            if st.button("🔗 Apply marks to filtered table"):
                result = ft.copy()
                for _, mrow in st.session_state.marked_table.iterrows():
                    oid, act = mrow["id"], mrow["type"]
                    if act == "remove":
                        result = result[result[id_col] != oid]
                    elif act == "add" and oid not in result[id_col].values:
                        orig = df_pred[df_pred[id_col] == oid]
                        if not orig.empty:
                            result = pd.concat([result, orig], ignore_index=True)
                st.session_state.matched_table = result
                st.success(f"Matched table: **{len(result):,}** rows")

            if "matched_table" in st.session_state:
                st.markdown("### Matched table")
                st.dataframe(st.session_state.matched_table, use_container_width=True)

                csv_bytes = st.session_state.matched_table.to_csv(index=False).encode()
                st.download_button(
                    "⬇️  Download matched table",
                    data=csv_bytes,
                    file_name="matched_table.csv",
                    mime="text/csv",
                )

# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; margin-top:40px; padding: 12px;
            border-top: 1px solid rgba(0,212,255,.15);
            font-size:.75rem; color:#2d5a8e; font-family:DM Mono,monospace;'>
  Galaxy Viewer 3.0 · CHANCES Project · Copyright © 2025 CHANCES Collaboration
</div>
""", unsafe_allow_html=True)
