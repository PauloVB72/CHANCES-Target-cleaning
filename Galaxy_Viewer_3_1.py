#!/usr/bin/env python3
# =============================================================================
# Galaxy Classifier — CHANCES Project
# Copyright (c) 2025 CHANCES Collaboration
# License: MIT
#
# Galaxy Viewer 3.0 — Standalone data exploration tool.
# Run with:  streamlit run Galaxy_Viewer_3_0.py
# =============================================================================
"""
Galaxy Viewer 3.0
=================
Interactive visualisation tool for Galaxy Classifier predictions.

New in v3.0 vs v2.0
--------------------
• Galaxy ID displayed in every card and thumbnail.
• Interactive RA/Dec scatter in Individual Explorer: hover any source to see
  its galaxy image and class probabilities in a floating panel.
• Aperture circles always drawn (uses 'default aperture' when column is absent).
• Centre detection correctly skipped when all apertures are identical.

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
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

# ── Page config (must be the very first Streamlit call) ────────────────────
st.set_page_config(
    page_title="Galaxy Groups + Image Viewer",
    page_icon="🌌",
    layout="wide",
)


# ═══════════════════════════════════════════════════════════════════════════
# IMAGE HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def load_image(path):
    """Load an image from disk; return a dark placeholder on failure."""
    try:
        if path and os.path.exists(str(path)):
            return Image.open(path).convert("RGB")
    except Exception:
        pass
    return Image.new("RGB", (256, 256), color="#2b3e4f")


def draw_red_circle(img, radius=20):
    img = img.copy()
    draw = ImageDraw.Draw(img)
    cx, cy = img.width // 2, img.height // 2
    draw.ellipse(
        (cx - radius, cy - radius, cx + radius, cy + radius),
        outline="red", width=3,
    )
    return img


def img_to_b64(img: Image.Image, size: int = 128) -> str:
    """Resize and encode a PIL Image as a base64 PNG data-URI."""
    img = img.copy()
    img.thumbnail((size, size), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def plot_probability_bars(probs, class_names, highlight_index=None):
    fig, ax = plt.subplots(figsize=(4, 2))
    y = np.arange(len(class_names))
    colors = ["#1f77b4"] * len(class_names)
    if highlight_index is not None:
        colors[highlight_index] = "#ff7f0e"
    ax.barh(y, probs, color=colors, alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability", fontsize=8)
    ax.tick_params(axis="x", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


def plot_global_distribution(df, class_name, threshold):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(df[class_name], bins=50, alpha=0.7, color="steelblue", edgecolor="black")
    ax.axvline(threshold, color="red", linestyle="--", linewidth=2,
               label=f"Threshold: {threshold:.2f}")
    ax.set_xlabel(f"Probability of {class_name}")
    ax.set_ylabel("Number of images")
    ax.set_title(f"Probability distribution - {class_name}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# FRIENDS-OF-FRIENDS GROUPING
# ═══════════════════════════════════════════════════════════════════════════

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


def assign_center_flags(df, group_col="group_id", aper_col="aper_radius",
                        ratio_thresh=1.2):
    """
    Mark the source with the largest aperture as centre only when its
    aperture is at least *ratio_thresh* × the median of the others.

    When all apertures are identical (e.g. the default 1-arcsec fallback),
    the ratio = 1.0 < ratio_thresh, so no centre is assigned — which is
    the correct behaviour.
    """
    df = df.copy()
    df["center_flag"] = 0
    centers = {}
    for gid in df[group_col].unique():
        if gid == -1:
            continue
        grp = df[df[group_col] == gid]
        if len(grp) < 2:
            centers[gid] = None
            continue
        max_idx   = grp[aper_col].idxmax()
        max_r     = grp.loc[max_idx, aper_col]
        others    = grp[grp.index != max_idx][aper_col]
        med_other = np.median(others) if len(others) else 0
        if med_other > 0 and (max_r / med_other) >= ratio_thresh:
            df.loc[max_idx, "center_flag"] = 1
            centers[gid] = max_idx
        else:
            centers[gid] = None
    return df, centers


# ═══════════════════════════════════════════════════════════════════════════
# INTERACTIVE SCATTER  (hover → image + probabilities)
# ═══════════════════════════════════════════════════════════════════════════

def build_interactive_scatter(
    group_df, ra_col, dec_col, selected_prob_col, threshold,
    prob_cols, center_idx, aper_col, default_aper_arcsec, image_root="",
):
    """
    Return a self-contained HTML/JS component with a Plotly RA/Dec scatter.

    Hovering over any point opens a floating panel with:
      - the galaxy thumbnail (base64-encoded)
      - a mini bar chart showing all class probabilities

    Aperture circles are always drawn. If *aper_col* is absent from
    *group_df*, *default_aper_arcsec* is used for every source.
    """
    has_aper = aper_col in group_df.columns

    # Cos-dec factor so circles look round on screen
    median_dec = float(group_df[dec_col].median())
    cos_dec    = float(np.cos(np.radians(median_dec))) if abs(median_dec) < 89 else 1.0

    points = []
    for idx, row in group_df.iterrows():
        # Resolve full image path
        path = str(row.get("file_loc", ""))
        if image_root and path and not os.path.isabs(path):
            path = os.path.join(image_root, path)
        img_b64 = img_to_b64(load_image(path))

        # Aperture
        if has_aper and pd.notna(row.get(aper_col)):
            aper_arcsec = float(row[aper_col])
        else:
            aper_arcsec = default_aper_arcsec
        aper_deg = aper_arcsec / 3600.0

        probs        = {col: float(row[col]) if col in row else 0.0 for col in prob_cols}
        prob_target  = float(row.get(selected_prob_col, 0.0))
        galaxy_id    = str(row.get("id_merge", f"idx_{idx}"))

        points.append({
            "ra":          float(row[ra_col]),
            "dec":         float(row[dec_col]),
            "id":          galaxy_id,
            "prob_target": prob_target,
            "probs":       probs,
            "img":         img_b64,
            "above":       prob_target >= threshold,
            "center":      (center_idx is not None and idx == center_idx),
            "aper_deg":    aper_deg,
        })

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  body {{ margin:0; padding:0; background:#fff; font-family:sans-serif; }}
  #plot {{ width:100%; height:520px; }}

  #hover-panel {{
    position: fixed;
    display: none;
    z-index: 9999;
    background: rgba(255,255,255,0.97);
    border: 1px solid #bbb;
    border-radius: 8px;
    padding: 10px;
    min-width: 210px;
    max-width: 250px;
    box-shadow: 0 4px 18px rgba(0,0,0,0.18);
    pointer-events: none;
  }}
  #hover-panel img {{
    width:100%; border-radius:5px;
    border:1px solid #ddd; margin-bottom:7px;
  }}
  .hp-id  {{ font-size:11px; font-weight:bold; color:#222;
             margin-bottom:7px; word-break:break-all; }}
  .hp-row {{ display:flex; align-items:center; margin-bottom:3px; gap:5px; }}
  .hp-label {{ font-size:10px; color:#555; width:68px; flex-shrink:0;
               white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
  .hp-bar-bg   {{ flex:1; height:7px; background:#e0e0e0;
                  border-radius:3px; overflow:hidden; }}
  .hp-bar-fill {{ height:100%; border-radius:3px; background:#1f77b4; }}
  .hp-bar-fill.active {{ background:#ff7f0e; }}
  .hp-val  {{ font-size:10px; color:#333; width:36px; text-align:right; }}
  hr.hpdiv {{ border:none; border-top:1px solid #ddd; margin:6px 0; }}
</style>
</head>
<body>
<div id="plot"></div>
<div id="hover-panel">
  <div class="hp-id"  id="hp-id"></div>
  <img id="hp-img" src="" alt="galaxy">
  <hr class="hpdiv">
  <div id="hp-probs"></div>
</div>
<script>
const POINTS    = {json.dumps(points)};
const PROB_COLS = {json.dumps(prob_cols)};
const THRESHOLD = {float(threshold)};
const SEL_COL   = {json.dumps(str(selected_prob_col))};
const COS_DEC   = {cos_dec};

// ── Build traces ────────────────────────────────────────────────────────
const nX=[],nY=[],nC=[],nT=[];   // normal
const aX=[],aY=[],aC=[],aT=[];   // above threshold
const cX=[],cY=[],cC=[],cT=[];   // centre
const shapes=[];

POINTS.forEach((p,i)=>{{
  const tip=`<b>${{p.id}}</b><br>P(${{SEL_COL}}): ${{(p.prob_target*100).toFixed(1)}}%`;
  if     (p.center){{cX.push(p.ra);cY.push(p.dec);cC.push(i);cT.push(tip);}}
  else if(p.above) {{aX.push(p.ra);aY.push(p.dec);aC.push(i);aT.push(tip);}}
  else             {{nX.push(p.ra);nY.push(p.dec);nC.push(i);nT.push(tip);}}

  // Aperture circle (cos-dec corrected)
  const rRa  = p.aper_deg / COS_DEC;
  const rDec = p.aper_deg;
  shapes.push({{
    type:'circle', xref:'x', yref:'y',
    x0: p.ra  - rRa,  y0: p.dec - rDec,
    x1: p.ra  + rRa,  y1: p.dec + rDec,
    line:{{ color:'rgba(100,100,100,0.45)', width:1, dash:'dot' }},
    fillcolor:'rgba(0,0,0,0)',
  }});
}});

const trN={{x:nX,y:nY,customdata:nC,mode:'markers',name:'Normal',
  text:nT,hovertemplate:'%{{text}}<extra></extra>',
  marker:{{color:'blue',size:8,symbol:'circle',line:{{color:'navy',width:1}}}}}};

const trA={{x:aX,y:aY,customdata:aC,mode:'markers',
  name:`P\u2265${{THRESHOLD.toFixed(2)}}`,
  text:aT,hovertemplate:'%{{text}}<extra></extra>',
  marker:{{color:'red',size:12,symbol:'star',line:{{color:'darkred',width:1.5}}}}}};

const trC={{x:cX,y:cY,customdata:cC,mode:'markers',name:'Center',
  text:cT,hovertemplate:'%{{text}}<extra></extra>',
  marker:{{color:'gold',size:17,symbol:'star',line:{{color:'darkorange',width:2}}}}}};

const layout={{
  xaxis:{{title:'RA (deg)',autorange:'reversed',gridcolor:'#eee',zeroline:false}},
  yaxis:{{title:'Dec (deg)',gridcolor:'#eee',zeroline:false,
          scaleanchor:'x',scaleratio: 1/COS_DEC}},
  legend:{{bgcolor:'rgba(255,255,255,0.85)',bordercolor:'#ccc',borderwidth:1}},
  shapes:shapes,
  margin:{{l:60,r:20,t:30,b:50}},
  hovermode:'closest',
  plot_bgcolor:'#fafafa',
  paper_bgcolor:'#fff',
}};

Plotly.newPlot('plot',[trN,trA,trC],layout,
  {{responsive:true,displayModeBar:true,
    modeBarButtonsToRemove:['lasso2d','select2d'],displaylogo:false}});

// ── Hover panel ──────────────────────────────────────────────────────────
const panel=document.getElementById('hover-panel');
const hpImg=document.getElementById('hp-img');
const hpId =document.getElementById('hp-id');
const hpPrb=document.getElementById('hp-probs');

function posPanel(mx,my){{
  const pw=250,ph=330,vw=window.innerWidth,vh=window.innerHeight;
  let left=mx+16,top=my-60;
  if(left+pw>vw-10) left=mx-pw-16;
  if(top+ph>vh-10)  top=vh-ph-10;
  if(top<10) top=10;
  panel.style.left=left+'px'; panel.style.top=top+'px';
}}

document.getElementById('plot').on('plotly_hover',function(data){{
  if(!data.points.length) return;
  const pt=data.points[0];
  if(pt.customdata===undefined) return;
  const p=POINTS[pt.customdata];
  hpId.textContent='ID: '+p.id;
  hpImg.src=p.img;
  let html='';
  PROB_COLS.forEach(col=>{{
    const val=p.probs[col]||0;
    const pct=(val*100).toFixed(1);
    const act=(col===SEL_COL)?'active':'';
    const lbl=col.length>11?col.slice(0,9)+'\u2026':col;
    html+=`<div class="hp-row">
      <span class="hp-label" title="${{col}}">${{lbl}}</span>
      <div class="hp-bar-bg">
        <div class="hp-bar-fill ${{act}}" style="width:${{pct}}%"></div>
      </div>
      <span class="hp-val">${{pct}}%</span>
    </div>`;
  }});
  hpPrb.innerHTML=html;
  posPanel(data.event.clientX,data.event.clientY);
  panel.style.display='block';
}});
document.getElementById('plot').on('plotly_unhover',()=>{{panel.style.display='none';}});
document.getElementById('plot').addEventListener('mousemove',e=>{{
  if(panel.style.display==='block') posPanel(e.clientX,e.clientY);
}});
</script>
</body>
</html>"""
    return html


# ═══════════════════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ═══════════════════════════════════════════════════════════════════════════

_defaults = {
    "marked_table":      pd.DataFrame(columns=["id", "type"]),
    "selected_gid":      None,
    "threshold":         0.5,
    "selected_prob_col": None,
    "df_merged":         None,
    "page_number":       1,
    "df_pred":           None,
    "df_coord":          None,
    "prob_cols":         [],
    "valid_groups":      [],
    "group_sizes":       {},
    "centers":           {},
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


def add_to_marked_table(img_id, action_type):
    df = st.session_state.marked_table
    if img_id in df["id"].values:
        df.loc[df["id"] == img_id, "type"] = action_type
    else:
        st.session_state.marked_table = pd.concat(
            [df, pd.DataFrame({"id": [img_id], "type": [action_type]})],
            ignore_index=True,
        )


# ═══════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════

logo_path = "chanceslogo.png"
hc1, hc2 = st.columns([1, 5])
with hc1:
    if os.path.exists(logo_path):
        st.image(Image.open(logo_path), width=100)
    else:
        st.markdown("### CHANCES\n4 MOST")
with hc2:
    st.title("Galaxy Explorer: Images and Groups")
    st.markdown("Prediction viewer and friends‑of‑friends grouping in one interface.")


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR — DATA UPLOAD
# ═══════════════════════════════════════════════════════════════════════════

st.sidebar.header("📂 Data upload")
pred_file  = st.sidebar.file_uploader("Predictions file (CSV)",
                                       type=["csv"], key="pred_combined")
coord_file = st.sidebar.file_uploader(
    "Coordinates file (CSV, optional for groups)", type=["csv"], key="coord_combined"
)
image_root = st.sidebar.text_input(
    "Image root directory (optional)", value="",
    help="If paths in 'file_loc' are relative, specify the base folder.",
)

if pred_file is not None:
    if (st.session_state.df_pred is None
            or st.session_state.get("_last_pred_name") != pred_file.name):
        try:
            df = pd.read_csv(pred_file)
            excl  = {"id_str", "file_loc", "filename", "label", "id"}
            pcols = [c for c in df.columns if c not in excl]
            if not pcols:
                pcols = [c for c in df.select_dtypes(include=[np.number]).columns
                         if c not in excl]
            st.session_state.df_pred         = df
            st.session_state.prob_cols       = pcols
            st.session_state._last_pred_name = pred_file.name
        except Exception as e:
            st.error(f"Error reading predictions file: {e}")
            st.session_state.df_pred = None
else:
    st.session_state.df_pred   = None
    st.session_state.prob_cols = []

if coord_file is not None:
    if (st.session_state.df_coord is None
            or st.session_state.get("_last_coord_name") != coord_file.name):
        try:
            st.session_state.df_coord         = pd.read_csv(coord_file)
            st.session_state._last_coord_name = coord_file.name
        except Exception as e:
            st.error(f"Error reading coordinates file: {e}")
            st.session_state.df_coord = None
else:
    st.session_state.df_coord = None


# ═══════════════════════════════════════════════════════════════════════════
# MAIN TABS
# ═══════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3 = st.tabs([
    "🖼️ Image Viewer",
    "🌠 Group Analysis",
    "📋 Filter & Match",
])


# ───────────────────────────────────────────────────────────────────────────
# TAB 1 — IMAGE VIEWER
# ───────────────────────────────────────────────────────────────────────────
with tab1:
    st.header("Prediction explorer")

    if st.session_state.df_pred is None:
        st.info("Upload a predictions file in the sidebar to start.")
    else:
        df_pred   = st.session_state.df_pred
        prob_cols = st.session_state.prob_cols

        if not prob_cols:
            st.error("No probability columns found.")
            st.stop()

        st.sidebar.markdown("---")
        st.sidebar.header("🖼️ Viewer settings")

        selected_class = st.sidebar.selectbox("Class of interest",
                                               options=prob_cols, index=0)
        selected_idx   = prob_cols.index(selected_class)

        threshold_img = st.sidebar.slider(
            f"Min threshold for {selected_class}", 0.0, 1.0, 0.5, 0.01,
            key="threshold_img",
        )

        total_images    = int((df_pred[selected_class] >= threshold_img).sum())
        max_per_page    = max(total_images, 1)
        images_per_page = st.sidebar.number_input(
            "Images per page", 1, max_per_page, min(10, max_per_page), 1,
            key="images_per_page",
        )

        total_pages = max(1, (total_images - 1) // images_per_page + 1)
        if st.session_state.page_number > total_pages:
            st.session_state.page_number = max(1, total_pages)

        # Pagination buttons — text labels so they render correctly in all themes
        col_prev, col_page, col_next = st.sidebar.columns([1, 2, 1])
        with col_prev:
            if st.button("◀ Prev", key="btn_prev") and st.session_state.page_number > 1:
                st.session_state.page_number -= 1
                st.rerun()
        with col_page:
            st.write(f"Page {st.session_state.page_number} of {total_pages}")
        with col_next:
            if st.button("Next ▶", key="btn_next") \
                    and st.session_state.page_number < total_pages:
                st.session_state.page_number += 1
                st.rerun()

        sort_order = st.sidebar.radio(
            "Sort by", ["Highest first", "Lowest first"], key="sort_order"
        )

        # Metrics
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Total images", len(df_pred))
        mc2.metric(f"{selected_class} ≥ {threshold_img:.2f}", total_images)
        if "label" in df_pred.columns:
            acc = (df_pred[prob_cols].values.argmax(axis=1)
                   == df_pred["label"].values).mean()
            mc3.metric("Global accuracy", f"{acc:.2%}")

        # Collapsible distribution plot — plain label so the arrow renders correctly
        with st.expander("Global probability distribution"):
            fig_dist = plot_global_distribution(df_pred, selected_class, threshold_img)
            st.pyplot(fig_dist)
            plt.close(fig_dist)

        # Filter & paginate
        df_filtered = df_pred[df_pred[selected_class] >= threshold_img].copy()
        df_filtered = df_filtered.sort_values(
            selected_class, ascending=(sort_order == "Lowest first")
        )
        s = (st.session_state.page_number - 1) * images_per_page
        df_page = df_filtered.iloc[s : s + images_per_page]

        if df_page.empty:
            st.warning("No images on this page.")
        else:
            if image_root and "file_loc" in df_page.columns:
                df_page = df_page.copy()
                df_page["full_path"] = df_page["file_loc"].apply(
                    lambda x: os.path.join(image_root, x)
                    if not os.path.isabs(str(x)) else x
                )
            else:
                df_page["full_path"] = df_page.get("file_loc", None)

            cols_grid = st.columns(2)
            for i, (idx, row) in enumerate(df_page.iterrows()):
                with cols_grid[i % 2]:
                    img    = load_image(row.get("full_path"))
                    img_id = str(row.get("id_str", f"img_{idx}"))

                    marked_type = None
                    if img_id in st.session_state.marked_table["id"].values:
                        marked_type = st.session_state.marked_table.loc[
                            st.session_state.marked_table["id"] == img_id, "type"
                        ].values[0]

                    # Galaxy ID (new in v3.0)
                    st.markdown(f"**ID:** `{img_id}`")
                    st.image(img, use_container_width=True)

                    colA, colB, colC, colD = st.columns([1, 1, 1, 1])
                    with colA:
                        if st.button("➕ Add", key=f"add_{img_id}"):
                            add_to_marked_table(img_id, "add"); st.rerun()
                    with colB:
                        if st.button("➖ Remove", key=f"rem_{img_id}"):
                            add_to_marked_table(img_id, "remove"); st.rerun()
                    with colC:
                        if st.button("🔴 Centro", key=f"ctr_{img_id}"):
                            add_to_marked_table(img_id, "centro"); st.rerun()
                    with colD:
                        st.caption(
                            f"Marked: {marked_type}" if marked_type else "Not marked"
                        )

                    st.markdown(
                        f"**ID:** {img_id}  |  **{selected_class}:** {row[selected_class]:.2%}"
                    )
                    if "label" in row and pd.notna(row["label"]):
                        true_cls = (
                            prob_cols[int(row["label"])]
                            if int(row["label"]) < len(prob_cols)
                            else f"Class {int(row['label'])}"
                        )
                        st.markdown(f"**True:** {true_cls}")

                    fig_bar = plot_probability_bars(
                        row[prob_cols].values.astype(float), prob_cols, selected_idx
                    )
                    st.pyplot(fig_bar, use_container_width=True)
                    plt.close(fig_bar)
                    st.markdown("---")

        with st.expander("View current marked table"):
            st.dataframe(st.session_state.marked_table)
            if st.button("Clear marked table"):
                st.session_state.marked_table = pd.DataFrame(columns=["id", "type"])
                st.rerun()


# ───────────────────────────────────────────────────────────────────────────
# TAB 2 — GROUP ANALYSIS
# ───────────────────────────────────────────────────────────────────────────
with tab2:
    st.header("Friends‑of‑friends group analysis")

    if st.session_state.df_pred is None or st.session_state.df_coord is None:
        st.info("Upload both predictions and coordinates files in the sidebar.")
    else:
        df_pred   = st.session_state.df_pred
        df_coord  = st.session_state.df_coord
        prob_cols = st.session_state.prob_cols

        st.sidebar.markdown("---")
        st.sidebar.header("🌠 Group settings")

        id_col_pred  = st.sidebar.text_input("ID in predictions", "id_str",
                                              key="group_id_pred")
        id_col_coord = st.sidebar.text_input("ID in coordinates", "OBJID",
                                              key="group_id_coord")
        ra_col   = st.sidebar.text_input("RA column",             "RA_J2000",  key="group_ra")
        dec_col  = st.sidebar.text_input("Dec column",            "Dec_J2000", key="group_dec")
        aper_col = st.sidebar.text_input("Aperture radius column","aper_radius",key="group_aper")
        default_aper = st.sidebar.number_input(
            "Default aperture (arcsec)", 0.1, 100.0, 1.0, 0.1,
            key="group_default_aper",
            help="Used when the aperture column is absent or has NaN values. "
                 "Centre detection requires real aperture variation.",
        )

        linking_radius  = st.sidebar.slider("Linking radius (arcsec)", 1.0, 120.0,
                                             30.0, 1.0, key="group_link")
        min_group_size  = st.sidebar.slider("Minimum group size", 2, 20, 5, 1,
                                             key="group_min")
        ratio_threshold = st.sidebar.slider("Center ratio threshold", 1.0, 3.0,
                                             1.2, 0.05, key="group_ratio")
        show_thumbnails = st.sidebar.checkbox("Show thumbnails for selected group",
                                              key="show_thumbnails")

        if st.sidebar.button("🚀 Process groups", type="primary", key="group_process"):
            dfp = df_pred.rename(columns={id_col_pred:  "id_merge"})
            dfc = df_coord.rename(columns={id_col_coord: "id_merge"})
            df_merged = pd.merge(dfp, dfc, on="id_merge", how="inner")

            if df_merged.empty:
                st.error("Merge produced no results. Check IDs.")
                st.stop()

            # Aperture column — add with default when absent
            if aper_col not in df_merged.columns:
                st.warning(
                    f"Column '{aper_col}' not found in the merged table. "
                    f"Using {default_aper:.1f} arcsec for all sources. "
                    "Centre detection will not be possible (all apertures equal)."
                )
                df_merged[aper_col] = default_aper
            else:
                df_merged[aper_col] = (
                    pd.to_numeric(df_merged[aper_col], errors="coerce")
                    .fillna(default_aper)
                )

            df_merged = df_merged.dropna(subset=[ra_col, dec_col, aper_col])

            with st.spinner("Grouping sources…"):
                df_merged["group_id"] = find_groups(
                    df_merged[ra_col].values,
                    df_merged[dec_col].values,
                    linking_radius,
                )

            n_raw = len(df_merged[df_merged["group_id"] != -1]["group_id"].unique())
            st.info(f"Groups found (unfiltered): {n_raw}")

            df_merged, centers = assign_center_flags(
                df_merged, aper_col=aper_col, ratio_thresh=ratio_threshold,
            )

            valid = []
            for gid in df_merged["group_id"].unique():
                if gid == -1:
                    continue
                sz = (df_merged["group_id"] == gid).sum()
                if sz < min_group_size:
                    df_merged.loc[df_merged["group_id"] == gid, "group_id"] = -1
                else:
                    valid.append(gid)

            gsizes = {g: (df_merged["group_id"] == g).sum() for g in valid}
            valid  = sorted(valid, key=lambda g: gsizes[g], reverse=True)

            st.session_state.df_merged    = df_merged
            st.session_state.centers      = centers
            st.session_state.valid_groups = valid
            st.session_state.group_sizes  = gsizes
            st.success(f"Groups with ≥ {min_group_size} members: {len(valid)}")

        if st.session_state.df_merged is not None:
            df_merged    = st.session_state.df_merged
            centers      = st.session_state.centers
            valid_groups = st.session_state.valid_groups
            prob_cols    = st.session_state.prob_cols

            st.sidebar.markdown("---")
            st.sidebar.header("🎚️ Probability filter")
            selected_prob_col = st.sidebar.selectbox(
                "Probability column", prob_cols,
                index=(prob_cols.index(st.session_state.selected_prob_col)
                       if st.session_state.selected_prob_col in prob_cols else 0),
                key="group_prob_col",
            )
            st.session_state.selected_prob_col = selected_prob_col
            threshold = st.sidebar.slider(
                f"Min threshold for {selected_prob_col}",
                0.0, 1.0, st.session_state.threshold, 0.01,
                key="group_threshold",
            )
            st.session_state.threshold = threshold

            # Group stats
            group_stats = []
            for gid in valid_groups:
                gdf   = df_merged[df_merged["group_id"] == gid]
                cmask = gdf["center_flag"] == 1
                if cmask.any():
                    ra_c  = gdf[cmask][ra_col].iloc[0]
                    dec_c = gdf[cmask][dec_col].iloc[0]
                else:
                    ra_c, dec_c = gdf[ra_col].median(), gdf[dec_col].median()
                n_above = (gdf[selected_prob_col] >= threshold).sum()
                group_stats.append({
                    "Group ID":             gid,
                    "Size":                 len(gdf),
                    f"Above {threshold:.2f}": n_above,
                    "Below":                len(gdf) - n_above,
                    "Fraction Above":       n_above / len(gdf),
                    "RA_center":            ra_c,
                    "Dec_center":           dec_c,
                    "Has Center":           cmask.any(),
                })
            df_groups = pd.DataFrame(group_stats)

            subtab1, subtab2, subtab3 = st.tabs([
                "🌍 Group map",
                "🔍 Individual explorer",
                "📊 Statistics",
            ])

            # ── GROUP MAP ─────────────────────────────────────────────
            with subtab1:
                if df_groups.empty:
                    st.warning("No groups.")
                else:
                    col_n = f"Above {threshold:.2f}"
                    fig_map = px.scatter(
                        df_groups, x="RA_center", y="Dec_center",
                        size="Size", color=col_n,
                        hover_data={c: True for c in df_groups.columns
                                    if c not in ["RA_center", "Dec_center"]},
                        color_continuous_scale="Reds",
                        title=f"Groups by {selected_prob_col} ≥ {threshold:.2f}",
                        labels={"RA_center": "RA", "Dec_center": "Dec",
                                col_n: f"N≥{threshold}"},
                    )
                    fig_map.update_xaxes(autorange="reversed")
                    st.plotly_chart(fig_map, use_container_width=True)

                    st.subheader("Group summary (sorted by number above threshold)")
                    disp_cols = ["Group ID", "Size", f"Above {threshold:.2f}",
                                 "Below", "Fraction Above", "Has Center"]
                    st.dataframe(
                        df_groups[disp_cols].sort_values(
                            f"Above {threshold:.2f}", ascending=False
                        ),
                        use_container_width=True,
                    )

            # ── INDIVIDUAL EXPLORER ────────────────────────────────────
            with subtab2:
                if not valid_groups:
                    st.warning("No groups.")
                else:
                    if st.session_state.selected_gid not in valid_groups:
                        st.session_state.selected_gid = valid_groups[0]

                    gopts = {
                        f"Group {g} (size: {st.session_state.group_sizes[g]})": g
                        for g in valid_groups
                    }
                    sel_label = st.selectbox(
                        "Select group", list(gopts.keys()),
                        index=list(gopts.values()).index(
                            st.session_state.selected_gid
                        ),
                    )
                    selected_gid = gopts[sel_label]
                    st.session_state.selected_gid = selected_gid

                    group_df   = df_merged[df_merged["group_id"] == selected_gid].copy()
                    center_idx = centers.get(selected_gid)

                    n_above = (group_df[selected_prob_col] >= threshold).sum()
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Size", len(group_df))
                    c2.metric(f"{selected_prob_col} ≥ {threshold:.2f}", n_above)
                    c3.metric("Fraction", f"{n_above/len(group_df):.2%}")
                    c4.metric("Has center", "✅" if center_idx else "❌")

                    # Interactive scatter (v3.0 — hover shows image)
                    st.markdown(
                        "**RA / Dec position plot** — "
                        "hover any point to preview the galaxy image and probabilities."
                    )
                    html_sc = build_interactive_scatter(
                        group_df            = group_df,
                        ra_col              = ra_col,
                        dec_col             = dec_col,
                        selected_prob_col   = selected_prob_col,
                        threshold           = threshold,
                        prob_cols           = prob_cols,
                        center_idx          = center_idx,
                        aper_col            = aper_col,
                        default_aper_arcsec = default_aper,
                        image_root          = image_root,
                    )
                    components.html(html_sc, height=540, scrolling=False)

                    # Thumbnails
                    if show_thumbnails and "file_loc" in group_df.columns:
                        st.subheader("Thumbnails")
                        root = image_root or ""
                        group_df = group_df.copy()
                        group_df["full_path"] = group_df["file_loc"].apply(
                            lambda x: os.path.join(root, x)
                            if not os.path.isabs(str(x)) else x
                        )
                        th_cols = st.columns(4)
                        for i, (idx, row) in enumerate(group_df.iterrows()):
                            img_id   = str(row.get("id_merge", f"img_{idx}"))
                            prob_val = row[selected_prob_col]
                            marked   = img_id in st.session_state.marked_table["id"].values
                            with th_cols[i % 4]:
                                # Galaxy ID (v3.0)
                                st.caption(f"ID: {img_id}")
                                if os.path.exists(str(row["full_path"])):
                                    st.image(Image.open(row["full_path"]),
                                             use_container_width=True)
                                else:
                                    st.image(load_image(None), use_container_width=True)
                                tA, tB = st.columns(2)
                                with tA:
                                    if st.button("➕ Add", key=f"gadd_{img_id}"):
                                        add_to_marked_table(img_id, "add"); st.rerun()
                                with tB:
                                    if st.button("➖ Remove", key=f"grem_{img_id}"):
                                        add_to_marked_table(img_id, "remove"); st.rerun()
                                st.caption(f"Prob: {prob_val:.2%}")
                                if marked:
                                    st.caption("Marked ✓")

                    st.subheader("Members")
                    disp = ["id_merge", ra_col, dec_col, aper_col,
                            selected_prob_col, "center_flag"]
                    disp = [c for c in disp if c in group_df.columns]
                    st.dataframe(
                        group_df[disp].sort_values(
                            selected_prob_col, ascending=False
                        ),
                        use_container_width=True,
                    )

            # ── STATISTICS ────────────────────────────────────────────
            with subtab3:
                if not df_groups.empty:
                    col_n = f"Above {threshold:.2f}"
                    s1, s2 = st.columns(2)
                    with s1:
                        fig, ax = plt.subplots()
                        ax.hist(df_groups[col_n], bins=20,
                                color="coral", edgecolor="black")
                        ax.set_xlabel(f"N with {selected_prob_col} ≥ {threshold:.2f}")
                        ax.set_ylabel("Number of groups")
                        st.pyplot(fig); plt.close(fig)
                    with s2:
                        fig, ax = plt.subplots()
                        ax.hist(df_groups["Fraction Above"], bins=20,
                                color="steelblue", edgecolor="black")
                        ax.set_xlabel("Fraction above threshold")
                        ax.set_ylabel("Number of groups")
                        st.pyplot(fig); plt.close(fig)
                    st.dataframe(df_groups.describe())


# ───────────────────────────────────────────────────────────────────────────
# TAB 3 — FILTER & MATCH
# ───────────────────────────────────────────────────────────────────────────
with tab3:
    st.header("Filter and Match Objects")

    if st.session_state.df_pred is None:
        st.info("Upload a predictions file in the sidebar to use this tab.")
    else:
        df_pred   = st.session_state.df_pred
        prob_cols = st.session_state.prob_cols

        id_col = next(
            (c for c in ["id_str", "id", "OBJID"] if c in df_pred.columns), None
        )
        if id_col is None:
            st.error(
                "No ID column found. "
                "Please ensure your file has 'id_str', 'id', or 'OBJID'."
            )
            st.stop()

        st.subheader("Define filter conditions")
        st.markdown(
            "Specify conditions for each probability column. "
            "Use 'ignore' to exclude a column."
        )

        combine_mode = st.radio(
            "Combine conditions with:",
            ["AND (all must match)", "OR (at least one must match)"],
            horizontal=True,
        )

        conditions = {}
        for col in prob_cols:
            col1, col2 = st.columns([1, 3])
            with col1:
                use_col = st.checkbox(f"Include {col}", value=True, key=f"use_{col}")
            with col2:
                if use_col:
                    cond = st.selectbox(
                        f"Condition for {col}",
                        ["any", ">", ">=", "<", "<=", "="],
                        key=f"cond_{col}",
                    )
                    if cond != "any":
                        th = st.number_input(
                            f"Threshold for {col}", 0.0, 1.0, 0.5, 0.01,
                            key=f"thresh_{col}",
                        )
                        conditions[col] = (cond, th)
                    else:
                        conditions[col] = ("any", None)
                else:
                    conditions[col] = ("ignore", None)

        if st.button("Generate filtered table"):
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
            st.success(
                f"Generated table with {len(st.session_state.filtered_table)} rows."
            )

        if "filtered_table" in st.session_state:
            st.subheader("Filtered Table")
            st.dataframe(st.session_state.filtered_table, use_container_width=True)

            st.subheader("Marked Table (from Add/Remove)")
            st.dataframe(st.session_state.marked_table, use_container_width=True)

            if st.button("Match tables (apply add/remove to filtered table)"):
                result_df = st.session_state.filtered_table.copy()
                for _, mrow in st.session_state.marked_table.iterrows():
                    oid, act = mrow["id"], mrow["type"]
                    if act == "remove":
                        result_df = result_df[result_df[id_col] != oid]
                    elif act == "add" and oid not in result_df[id_col].values:
                        orig = df_pred[df_pred[id_col] == oid]
                        if not orig.empty:
                            result_df = pd.concat([result_df, orig], ignore_index=True)
                st.session_state.matched_table = result_df
                st.success("Match completed. See table below.")

            if "matched_table" in st.session_state:
                st.subheader("Matched Table")
                st.dataframe(st.session_state.matched_table, use_container_width=True)
                st.download_button(
                    label="Download matched table as CSV",
                    data=st.session_state.matched_table.to_csv(index=False),
                    file_name="matched_table.csv",
                    mime="text/csv",
                )
