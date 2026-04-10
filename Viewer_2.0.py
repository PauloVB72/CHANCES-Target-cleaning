#!/usr/bin/env python3
"""
combined_app.py - Versión final con logo, botón centro y tabla de grupos.
Uso:
    streamlit run combined_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import plotly.express as px
from PIL import Image, ImageDraw
import os

# ----------------------------------------------------------------------
# Funciones de imagen (comunes)
# ----------------------------------------------------------------------
def load_image(path):
    try:
        if path and os.path.exists(path):
            return Image.open(path).convert('RGB')
        else:
            return Image.new('RGB', (256, 256), color='#2b3e4f')
    except Exception:
        return Image.new('RGB', (256, 256), color='#2b3e4f')

def draw_red_circle(img, radius=20):
    img = img.copy()
    draw = ImageDraw.Draw(img)
    cx, cy = img.width // 2, img.height // 2
    draw.ellipse(
        (cx - radius, cy - radius, cx + radius, cy + radius),
        outline='red',
        width=3
    )
    return img

def plot_probability_bars(probs, class_names, highlight_index=None):
    fig, ax = plt.subplots(figsize=(4, 2))
    y_pos = np.arange(len(class_names))
    colors = ['#1f77b4'] * len(class_names)
    if highlight_index is not None:
        colors[highlight_index] = '#ff7f0e'
    ax.barh(y_pos, probs, color=colors, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel('Probability', fontsize=8)
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    return fig

def plot_global_distribution(df, class_name, threshold):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(df[class_name], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.2f}')
    ax.set_xlabel(f'Probability of {class_name}')
    ax.set_ylabel('Number of images')
    ax.set_title(f'Probability distribution - {class_name}')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    return fig

# ----------------------------------------------------------------------
# Funciones de agrupamiento
# ----------------------------------------------------------------------
def radec_to_cartesian(ra_deg, dec_deg):
    ra_rad = np.radians(ra_deg)
    dec_rad = np.radians(dec_deg)
    cos_dec = np.cos(dec_rad)
    x = cos_dec * np.cos(ra_rad)
    y = cos_dec * np.sin(ra_rad)
    z = np.sin(dec_rad)
    return np.column_stack([x, y, z])

def find_groups(ra, dec, linking_radius_arcsec):
    theta = np.radians(linking_radius_arcsec / 3600.0)
    chord = 2.0 * np.sin(theta / 2.0)
    coords = radec_to_cartesian(ra, dec)
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=chord, output_type='ndarray')
    n_src = len(ra)
    if len(pairs) > 0:
        row, col = pairs[:, 0], pairs[:, 1]
        data = np.ones(len(pairs))
        adj = csr_matrix((data, (row, col)), shape=(n_src, n_src))
        adj = adj + adj.T
    else:
        adj = csr_matrix((n_src, n_src))
    n_components, labels = connected_components(csgraph=adj, directed=False, return_labels=True)
    degrees = np.array(adj.sum(axis=1)).flatten()
    labels[degrees == 0] = -1
    unique_labels = np.unique(labels)
    mapping = {old: i for i, old in enumerate(unique_labels) if old != -1}
    new_labels = np.full_like(labels, -1)
    for old, new in mapping.items():
        new_labels[labels == old] = new
    return new_labels

def assign_center_flags(df, group_col='group_id', aper_col='aper_radius', ratio_thresh=1.2):
    df = df.copy()
    df['center_flag'] = 0
    centers = {}
    group_ids = df[group_col].unique()
    group_ids = group_ids[group_ids != -1]
    for gid in group_ids:
        mask = df[group_col] == gid
        group = df[mask]
        if len(group) < 2:
            continue
        max_idx = group[aper_col].idxmax()
        max_radius = group.loc[max_idx, aper_col]
        other_radii = group[group.index != max_idx][aper_col]
        med_other = np.median(other_radii) if len(other_radii) > 0 else 0
        if med_other > 0 and (max_radius / med_other) >= ratio_thresh:
            df.loc[max_idx, 'center_flag'] = 1
            centers[gid] = max_idx
        else:
            centers[gid] = None
    return df, centers

# ----------------------------------------------------------------------
# Configuración de página y logo
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Galaxy Groups + Image Viewer",
    page_icon="🌌",
    layout="wide"
)

# Intentar cargar el logo
logo_path = "chanceslogo.png"
col1, col2 = st.columns([1, 5])
with col1:
    if os.path.exists(logo_path):
        logo = Image.open(logo_path)
        st.image(logo, width=100)
    else:
        st.markdown("### CHANCES\n4 MOST")  # Texto alternativo
with col2:
    st.title("Galaxy Explorer: Images and Groups")
    st.markdown("Prediction viewer and friends‑of‑friends grouping in one interface.")

# ----------------------------------------------------------------------
# Inicializar estado de sesión
# ----------------------------------------------------------------------
if 'marked_table' not in st.session_state:
    st.session_state.marked_table = pd.DataFrame(columns=['id', 'type'])

if 'selected_gid' not in st.session_state:
    st.session_state.selected_gid = None
if 'last_valid_groups' not in st.session_state:
    st.session_state.last_valid_groups = []
if 'threshold' not in st.session_state:
    st.session_state.threshold = 0.5
if 'selected_prob_col' not in st.session_state:
    st.session_state.selected_prob_col = None
if 'df_merged' not in st.session_state:
    st.session_state.df_merged = None

if 'page_number' not in st.session_state:
    st.session_state.page_number = 1

# DataFrames almacenados
if 'df_pred' not in st.session_state:
    st.session_state.df_pred = None
if 'df_coord' not in st.session_state:
    st.session_state.df_coord = None
if 'prob_cols' not in st.session_state:
    st.session_state.prob_cols = []

# ----------------------------------------------------------------------
# Funciones auxiliares para la tabla de marcados
# ----------------------------------------------------------------------
def add_to_marked_table(img_id, action_type):
    df = st.session_state.marked_table
    if img_id in df['id'].values:
        df.loc[df['id'] == img_id, 'type'] = action_type
    else:
        new_row = pd.DataFrame({'id': [img_id], 'type': [action_type]})
        st.session_state.marked_table = pd.concat([df, new_row], ignore_index=True)

# ----------------------------------------------------------------------
# Barra lateral: carga de datos
# ----------------------------------------------------------------------
st.sidebar.header("📂 Data upload")

pred_file = st.sidebar.file_uploader("Predictions file (CSV)", type=['csv'], key='pred_combined')
coord_file = st.sidebar.file_uploader("Coordinates file (CSV, optional for groups)", type=['csv'], key='coord_combined')
image_root = st.sidebar.text_input("Image root directory (optional)", value="", help="If paths in 'file_loc' are relative, specify the base folder.")

# --- Leer predicciones una vez y guardar en sesión ---
if pred_file is not None:
    if st.session_state.df_pred is None or st.session_state.get('_last_pred_name') != pred_file.name:
        try:
            df = pd.read_csv(pred_file)
            exclude_cols = {'id_str', 'file_loc', 'filename', 'label', 'id'}
            prob_cols = [col for col in df.columns if col not in exclude_cols]
            if not prob_cols:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                prob_cols = [c for c in numeric_cols if c not in exclude_cols]
            st.session_state.df_pred = df
            st.session_state.prob_cols = prob_cols
            st.session_state._last_pred_name = pred_file.name
        except Exception as e:
            st.error(f"Error reading predictions file: {e}")
            st.session_state.df_pred = None
else:
    st.session_state.df_pred = None
    st.session_state.prob_cols = []

# --- Leer coordenadas una vez y guardar en sesión ---
if coord_file is not None:
    if st.session_state.df_coord is None or st.session_state.get('_last_coord_name') != coord_file.name:
        try:
            st.session_state.df_coord = pd.read_csv(coord_file)
            st.session_state._last_coord_name = coord_file.name
        except Exception as e:
            st.error(f"Error reading coordinates file: {e}")
            st.session_state.df_coord = None
else:
    st.session_state.df_coord = None

# ----------------------------------------------------------------------
# Pestañas principales
# ----------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🖼️ Image Viewer", "🌠 Group Analysis", "📋 Filter & Match"])

# ======================================================================
# TAB 1: VISOR DE IMÁGENES (con botón Centro)
# ======================================================================
with tab1:
    st.header("Prediction explorer")
    if st.session_state.df_pred is None:
        st.info("Upload a predictions file in the sidebar to start.")
    else:
        df_pred = st.session_state.df_pred
        prob_cols = st.session_state.prob_cols

        if not prob_cols:
            st.error("No probability columns found.")
            st.stop()

        st.sidebar.markdown("---")
        st.sidebar.header("🖼️ Viewer settings")

        selected_class = st.sidebar.selectbox("Class of interest", options=prob_cols, index=0)
        selected_idx = prob_cols.index(selected_class)

        threshold_img = st.sidebar.slider(
            f"Min threshold for {selected_class}",
            0.0, 1.0, 0.5, 0.01, key='threshold_img'
        )

        total_images = len(df_pred[df_pred[selected_class] >= threshold_img])
        max_per_page = total_images if total_images > 0 else 1
        images_per_page = st.sidebar.number_input(
            "Images per page", 1, max_per_page, min(10, max_per_page), 1, key='images_per_page'
        )

        total_pages = (total_images - 1) // images_per_page + 1 if total_images > 0 else 1
        if st.session_state.page_number > total_pages:
            st.session_state.page_number = max(1, total_pages)

        col_prev, col_page, col_next = st.sidebar.columns([1, 2, 1])
        with col_prev:
            if st.button("◀ Prev") and st.session_state.page_number > 1:
                st.session_state.page_number -= 1
                st.rerun()
        with col_page:
            st.write(f"Page {st.session_state.page_number} of {total_pages}")
        with col_next:
            if st.button("Next ▶") and st.session_state.page_number < total_pages:
                st.session_state.page_number += 1
                st.rerun()

        sort_order = st.sidebar.radio("Sort by", ["Highest first", "Lowest first"], key='sort_order')

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total images", len(df_pred))
        with col2:
            st.metric(f"{selected_class} ≥ {threshold_img:.2f}", total_images)
        if 'label' in df_pred.columns:
            pred_labels = df_pred[prob_cols].values.argmax(axis=1)
            accuracy = (pred_labels == df_pred['label'].values).mean()
            with col3:
                st.metric("Global accuracy", f"{accuracy:.2%}")

        df_filtered = df_pred[df_pred[selected_class] >= threshold_img].copy()
        if sort_order == "Highest first":
            df_filtered = df_filtered.sort_values(by=selected_class, ascending=False)
        else:
            df_filtered = df_filtered.sort_values(by=selected_class, ascending=True)

        start_idx = (st.session_state.page_number - 1) * images_per_page
        end_idx = start_idx + images_per_page
        df_page = df_filtered.iloc[start_idx:end_idx]

        if len(df_page) == 0:
            st.warning("No images on this page.")
        else:
            if image_root and 'file_loc' in df_filtered.columns:
                df_page = df_page.copy()
                df_page['full_path'] = df_page['file_loc'].apply(
                    lambda x: os.path.join(image_root, x) if not os.path.isabs(x) else x
                )
            else:
                df_page['full_path'] = df_page.get('file_loc', None)

            cols = st.columns(2)
            for i, (idx, row) in enumerate(df_page.iterrows()):
                with cols[i % 2]:
                    img = load_image(row['full_path'])
                    img_id = row.get('id_str', f"img_{idx}")

                    # Obtener tipo de marcado actual
                    marked_type = None
                    if img_id in st.session_state.marked_table['id'].values:
                        marked_type = st.session_state.marked_table[st.session_state.marked_table['id'] == img_id]['type'].values[0]

                    st.image(img, use_container_width=True)

                    # Tres botones: Add, Remove, Centro
                    colA, colB, colC, colD = st.columns([1, 1, 1, 1])
                    with colA:
                        if st.button("➕ Add", key=f"add_{img_id}"):
                            add_to_marked_table(img_id, 'add')
                            st.rerun()
                    with colB:
                        if st.button("➖ Remove", key=f"rem_{img_id}"):
                            add_to_marked_table(img_id, 'remove')
                            st.rerun()
                    with colC:
                        if st.button("🔴 Centro", key=f"centro_{img_id}"):
                            add_to_marked_table(img_id, 'centro')
                            st.rerun()
                    with colD:
                        st.caption(f"Marked: {marked_type}" if marked_type else "Not marked")

                    st.markdown(f"**ID:** {img_id}  |  **{selected_class}:** {row[selected_class]:.2%}")
                    if 'label' in row and pd.notna(row['label']):
                        true_class = prob_cols[int(row['label'])] if int(row['label']) < len(prob_cols) else f"Class {row['label']}"
                        st.markdown(f"**True:** {true_class}")

                    probs = row[prob_cols].values.astype(float)
                    fig = plot_probability_bars(probs, prob_cols, highlight_index=selected_idx)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                    st.markdown("---")

        with st.expander("View current marked table"):
            st.dataframe(st.session_state.marked_table)
            if st.button("Clear marked table"):
                st.session_state.marked_table = pd.DataFrame(columns=['id', 'type'])
                st.rerun()

# ======================================================================
# TAB 2: ANÁLISIS DE GRUPOS (con tabla resumen y probabilidad en miniaturas)
# ======================================================================
with tab2:
    st.header("Friends‑of‑friends group analysis")
    if st.session_state.df_pred is None or st.session_state.df_coord is None:
        st.info("Upload both predictions and coordinates files in the sidebar.")
    else:
        df_pred = st.session_state.df_pred
        df_coord = st.session_state.df_coord
        prob_cols = st.session_state.prob_cols

        st.sidebar.markdown("---")
        st.sidebar.header("🌠 Group settings")

        id_col_pred = st.sidebar.text_input("ID in predictions", "id_str", key='group_id_pred')
        id_col_coord = st.sidebar.text_input("ID in coordinates", "OBJID", key='group_id_coord')
        ra_col = st.sidebar.text_input("RA column", "RA_J2000", key='group_ra')
        dec_col = st.sidebar.text_input("Dec column", "Dec_J2000", key='group_dec')
        aper_col = st.sidebar.text_input("Aperture radius column", "aper_radius", key='group_aper')
        default_aper = st.sidebar.number_input("Default aperture (arcsec)", 0.1, 100.0, 1.0, 0.1, key='group_default_aper')

        linking_radius = st.sidebar.slider("Linking radius (arcsec)", 1.0, 120.0, 30.0, 1.0, key='group_link')
        min_group_size = st.sidebar.slider("Minimum group size", 2, 20, 5, 1, key='group_min')
        ratio_threshold = st.sidebar.slider("Center ratio threshold", 1.0, 3.0, 1.2, 0.05, key='group_ratio')
        show_thumbnails = st.sidebar.checkbox("Show thumbnails for selected group", key='show_thumbnails')

        if st.sidebar.button("🚀 Process groups", type="primary", key='group_process'):
            df_pred_renamed = df_pred.rename(columns={id_col_pred: 'id_merge'})
            df_coord_renamed = df_coord.rename(columns={id_col_coord: 'id_merge'})
            df_merged = pd.merge(df_pred_renamed, df_coord_renamed, on='id_merge', how='inner')
            if df_merged.empty:
                st.error("Merge produced no results. Check IDs.")
                st.stop()

            if aper_col not in df_merged.columns:
                df_merged[aper_col] = default_aper
            else:
                df_merged[aper_col] = pd.to_numeric(df_merged[aper_col], errors='coerce').fillna(default_aper)

            df_merged = df_merged.dropna(subset=[ra_col, dec_col, aper_col])

            with st.spinner("Grouping sources..."):
                df_merged['group_id'] = find_groups(
                    df_merged[ra_col].values,
                    df_merged[dec_col].values,
                    linking_radius
                )

            n_groups_raw = len(df_merged[df_merged['group_id'] != -1]['group_id'].unique())
            st.info(f"Groups found (unfiltered): {n_groups_raw}")

            df_merged, centers = assign_center_flags(
                df_merged,
                group_col='group_id',
                aper_col=aper_col,
                ratio_thresh=ratio_threshold
            )

            valid_groups = []
            for gid in df_merged['group_id'].unique():
                if gid == -1:
                    continue
                size = (df_merged['group_id'] == gid).sum()
                if size < min_group_size:
                    df_merged.loc[df_merged['group_id'] == gid, 'group_id'] = -1
                    df_merged.loc[df_merged['group_id'] == gid, 'center_flag'] = 0
                else:
                    valid_groups.append(gid)

            group_sizes = {gid: (df_merged['group_id'] == gid).sum() for gid in valid_groups}
            valid_groups = sorted(valid_groups, key=lambda x: group_sizes[x], reverse=True)

            st.session_state.df_merged = df_merged
            st.session_state.centers = centers
            st.session_state.valid_groups = valid_groups
            st.session_state.group_sizes = group_sizes
            st.success(f"Groups with ≥ {min_group_size} members: {len(valid_groups)}")

        if st.session_state.df_merged is not None:
            df_merged = st.session_state.df_merged
            centers = st.session_state.centers
            valid_groups = st.session_state.valid_groups
            prob_cols = st.session_state.prob_cols

            st.sidebar.markdown("---")
            st.sidebar.header("🎚️ Probability filter")
            selected_prob_col = st.sidebar.selectbox(
                "Probability column",
                prob_cols,
                index=prob_cols.index(st.session_state.selected_prob_col) if st.session_state.selected_prob_col in prob_cols else 0,
                key='group_prob_col'
            )
            st.session_state.selected_prob_col = selected_prob_col
            threshold = st.sidebar.slider(
                f"Min threshold for {selected_prob_col}",
                0.0, 1.0, st.session_state.threshold, 0.01,
                key='group_threshold'
            )
            st.session_state.threshold = threshold

            # Calcular estadísticas por grupo
            group_stats = []
            for gid in valid_groups:
                mask = df_merged['group_id'] == gid
                group_df = df_merged[mask]
                center_mask = group_df['center_flag'] == 1
                if center_mask.any():
                    center_row = group_df[center_mask].iloc[0]
                    ra_center = center_row[ra_col]
                    dec_center = center_row[dec_col]
                else:
                    ra_center = group_df[ra_col].median()
                    dec_center = group_df[dec_col].median()
                n_above = (group_df[selected_prob_col] >= threshold).sum()
                n_below = len(group_df) - n_above
                group_stats.append({
                    'Group ID': gid,
                    'Size': len(group_df),
                    f'Above {threshold:.2f}': n_above,
                    'Below': n_below,
                    'Fraction Above': n_above / len(group_df) if len(group_df) else 0,
                    'RA_center': ra_center,
                    'Dec_center': dec_center,
                    'Has Center': center_mask.any()
                })
            df_groups = pd.DataFrame(group_stats)

            subtab1, subtab2, subtab3 = st.tabs(["🌍 Group map", "🔍 Individual explorer", "📊 Statistics"])

            with subtab1:
                if df_groups.empty:
                    st.warning("No groups.")
                else:
                    # Mapa de grupos
                    fig = px.scatter(
                        df_groups, x='RA_center', y='Dec_center', size='Size', 
                        color=f'Above {threshold:.2f}',
                        hover_data={c: True for c in df_groups.columns if c not in ['RA_center', 'Dec_center']},
                        color_continuous_scale='Reds',
                        title=f'Groups by {selected_prob_col} ≥ {threshold:.2f}',
                        labels={'RA_center': 'RA', 'Dec_center': 'Dec', f'Above {threshold:.2f}': f'N ≥{threshold}'}
                    )
                    fig.update_xaxes(autorange="reversed")
                    st.plotly_chart(fig, use_container_width=True)

                    # Tabla resumen de grupos (ordenada por número sobre umbral)
                    st.subheader("Group summary (sorted by number above threshold)")
                    display_cols = ['Group ID', 'Size', f'Above {threshold:.2f}', 'Below', 'Fraction Above', 'Has Center']
                    st.dataframe(
                        df_groups[display_cols].sort_values(f'Above {threshold:.2f}', ascending=False),
                        use_container_width=True
                    )

            with subtab2:
                if not valid_groups:
                    st.warning("No groups.")
                else:
                    if st.session_state.selected_gid not in valid_groups:
                        st.session_state.selected_gid = valid_groups[0]
                    group_options = {f"Group {gid} (size: {st.session_state.group_sizes[gid]})": gid for gid in valid_groups}
                    selected_label = st.selectbox(
                        "Select group", list(group_options.keys()),
                        index=list(group_options.values()).index(st.session_state.selected_gid)
                    )
                    selected_gid = group_options[selected_label]
                    st.session_state.selected_gid = selected_gid

                    group_df = df_merged[df_merged['group_id'] == selected_gid].copy()
                    center_idx = centers.get(selected_gid)

                    n_above = (group_df[selected_prob_col] >= threshold).sum()
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Size", len(group_df))
                    col2.metric(f"{selected_prob_col} ≥ {threshold:.2f}", n_above)
                    col3.metric("Fraction", f"{n_above/len(group_df):.2%}")
                    col4.metric("Has center", "✅" if center_idx else "❌")

                    # Gráfico RA/Dec
                    fig, ax = plt.subplots(figsize=(10, 8))
                    spur_mask = group_df[selected_prob_col] >= threshold
                    ax.scatter(group_df[ra_col][~spur_mask], group_df[dec_col][~spur_mask], 
                               c='blue', s=50, label='Normal', edgecolors='k', alpha=0.7)
                    ax.scatter(group_df[ra_col][spur_mask], group_df[dec_col][spur_mask], 
                               c='red', s=120, marker='*', label=f'{selected_prob_col} ≥ {threshold:.2f}',
                               edgecolors='darkred', linewidth=1.5, zorder=5)
                    for _, row in group_df.iterrows():
                        radius_deg = row[aper_col] / 3600.0
                        circle = plt.Circle((row[ra_col], row[dec_col]), radius_deg, 
                                            color='gray', fill=False, linestyle='--', alpha=0.5, linewidth=0.8)
                        ax.add_patch(circle)
                    if center_idx is not None:
                        center_row = group_df.loc[center_idx]
                        ax.scatter(center_row[ra_col], center_row[dec_col], 
                                   s=300, marker='*', color='gold', edgecolors='darkorange',
                                   linewidth=2, label='Center', zorder=6)
                    ax.set_title(f'Group {selected_gid}')
                    ax.set_xlabel('RA (deg)')
                    ax.set_ylabel('Dec (deg)')
                    ax.invert_xaxis()
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    median_dec = group_df[dec_col].median()
                    ax.set_aspect(1.0 / np.cos(np.radians(median_dec)) if abs(median_dec) < 89 else 1.0)
                    st.pyplot(fig)
                    plt.close(fig)

                    # Miniaturas con probabilidad
                    if show_thumbnails and 'file_loc' in group_df.columns:
                        st.subheader("Thumbnails")
                        root = image_root or ""
                        group_df['full_path'] = group_df['file_loc'].apply(
                            lambda x: os.path.join(root, x) if not os.path.isabs(x) else x
                        )
                        cols = st.columns(4)
                        for i, (idx, row) in enumerate(group_df.iterrows()):
                            if os.path.exists(row['full_path']):
                                img = Image.open(row['full_path'])
                                img_id = row.get('id_merge', f"img_{idx}")
                                prob_val = row[selected_prob_col]
                                marked = img_id in st.session_state.marked_table['id'].values
                                with cols[i % 4]:
                                    st.image(img, use_container_width=True)
                                    colA, colB = st.columns(2)
                                    with colA:
                                        if st.button("➕ Add", key=f"gadd_{img_id}"):
                                            add_to_marked_table(img_id, 'add')
                                            st.rerun()
                                    with colB:
                                        if st.button("➖ Remove", key=f"grem_{img_id}"):
                                            add_to_marked_table(img_id, 'remove')
                                            st.rerun()
                                    st.caption(f"Prob: {prob_val:.2%}")  # <-- Probabilidad visible
                                    if marked:
                                        st.caption("Marked")

                    st.subheader("Members")
                    display_cols = ['id_merge', ra_col, dec_col, aper_col, selected_prob_col, 'center_flag']
                    display_cols = [c for c in display_cols if c in group_df.columns]
                    st.dataframe(group_df[display_cols].sort_values(selected_prob_col, ascending=False))

            with subtab3:
                if not df_groups.empty:
                    col1, col2 = st.columns(2)
                    with col1:
                        fig, ax = plt.subplots()
                        ax.hist(df_groups[f'Above {threshold:.2f}'], bins=20, color='coral', edgecolor='black')
                        ax.set_xlabel(f'N with {selected_prob_col} ≥ {threshold:.2f}')
                        ax.set_ylabel('Number of groups')
                        st.pyplot(fig)
                        plt.close(fig)
                    with col2:
                        fig, ax = plt.subplots()
                        ax.hist(df_groups['Fraction Above'], bins=20, color='steelblue', edgecolor='black')
                        ax.set_xlabel('Fraction above threshold')
                        ax.set_ylabel('Number of groups')
                        st.pyplot(fig)
                        plt.close(fig)
                    st.dataframe(df_groups.describe())

# ======================================================================
# TAB 3: FILTRAR Y COMBINAR (con AND/OR)
# ======================================================================
with tab3:
    st.header("Filter and Match Objects")

    if st.session_state.df_pred is None:
        st.info("Upload a predictions file in the sidebar to use this tab.")
    else:
        df_pred = st.session_state.df_pred
        prob_cols = st.session_state.prob_cols

        # Encontrar columna ID
        id_col = None
        for possible in ['id_str', 'id', 'OBJID']:
            if possible in df_pred.columns:
                id_col = possible
                break
        if id_col is None:
            st.error("No ID column found. Please ensure your file has 'id_str', 'id', or 'OBJID'.")
            st.stop()

        st.subheader("Define filter conditions")
        st.markdown("Specify conditions for each probability column. Use 'ignore' to exclude a column.")

        # Selector de modo de combinación
        combine_mode = st.radio(
            "Combine conditions with:",
            options=["AND (all must match)", "OR (at least one must match)"],
            index=0,
            horizontal=True
        )

        conditions = {}
        for col in prob_cols:
            col1, col2 = st.columns([1, 3])
            with col1:
                use_col = st.checkbox(f"Include {col}", value=True, key=f"use_{col}")
            with col2:
                if use_col:
                    cond = st.selectbox(f"Condition for {col}", ["any", ">", ">=", "<", "<=", "="], key=f"cond_{col}")
                    if cond != "any":
                        th = st.number_input(f"Threshold for {col}", 0.0, 1.0, 0.5, 0.01, key=f"thresh_{col}")
                        conditions[col] = (cond, th)
                    else:
                        conditions[col] = ("any", None)
                else:
                    conditions[col] = ("ignore", None)

        if st.button("Generate filtered table"):
            # Recopilar condiciones activas (no ignore y no any)
            active = [(col, cond, val) for col, (cond, val) in conditions.items() if cond not in ('ignore', 'any')]

            if not active:
                # Sin condiciones activas: devolver todas las filas
                mask = pd.Series([True] * len(df_pred))
            else:
                if combine_mode.startswith("AND"):
                    mask = pd.Series([True] * len(df_pred))
                    for col, cond, val in active:
                        if cond == '>':
                            mask &= (df_pred[col] > val)
                        elif cond == '>=':
                            mask &= (df_pred[col] >= val)
                        elif cond == '<':
                            mask &= (df_pred[col] < val)
                        elif cond == '<=':
                            mask &= (df_pred[col] <= val)
                        elif cond == '=':
                            mask &= (df_pred[col] == val)
                else:  # OR
                    mask = pd.Series([False] * len(df_pred))
                    for col, cond, val in active:
                        if cond == '>':
                            mask |= (df_pred[col] > val)
                        elif cond == '>=':
                            mask |= (df_pred[col] >= val)
                        elif cond == '<':
                            mask |= (df_pred[col] < val)
                        elif cond == '<=':
                            mask |= (df_pred[col] <= val)
                        elif cond == '=':
                            mask |= (df_pred[col] == val)

            filtered_df = df_pred[mask].copy()
            st.session_state.filtered_table = filtered_df
            st.success(f"Generated table with {len(filtered_df)} rows.")

        if 'filtered_table' in st.session_state:
            st.subheader("Filtered Table")
            st.dataframe(st.session_state.filtered_table, use_container_width=True)

            st.subheader("Marked Table (from Add/Remove)")
            st.dataframe(st.session_state.marked_table, use_container_width=True)

            if st.button("Match tables (apply add/remove to filtered table)"):
                result_df = st.session_state.filtered_table.copy()
                marked = st.session_state.marked_table

                for _, row in marked.iterrows():
                    obj_id = row['id']
                    action = row['type']
                    if obj_id in result_df[id_col].values:
                        if action == 'remove':
                            result_df = result_df[result_df[id_col] != obj_id]
                    else:
                        if action == 'add':
                            original_row = df_pred[df_pred[id_col] == obj_id]
                            if not original_row.empty:
                                result_df = pd.concat([result_df, original_row], ignore_index=True)
                st.session_state.matched_table = result_df
                st.success("Match completed. See table below.")

            if 'matched_table' in st.session_state:
                st.subheader("Matched Table")
                st.dataframe(st.session_state.matched_table, use_container_width=True)

                csv = st.session_state.matched_table.to_csv(index=False)
                st.download_button(
                    label="Download matched table as CSV",
                    data=csv,
                    file_name="matched_table.csv",
                    mime="text/csv"
                )