# ============================================================
# visualizer.py — Interactive 3D molecular graph visualization using Plotly.
# Replaces/extends matplotlib-based visualization with interactive Plotly graphs.
# Compatible with MolGraph-NCI project structure.
# Requires: plotly>=5.0.0, networkx, numpy
# ============================================================

import numpy as np
import networkx as nx
import os
import sys

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    import warnings
    warnings.warn("Plotly not installed. Install with: pip install 'plotly>=5.0.0'")

# Fallback to matplotlib if Plotly unavailable
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ─────────────────────────────────────────────────────────────
# Element colors (CPK coloring standard)
# ─────────────────────────────────────────────────────────────
ELEMENT_COLORS = {
    'H': '#FFFFFF',   'He': '#D9FFFF',
    'Li': '#CC80FF',  'Be': '#C2FF00',  'B': '#FFB5B5',   'C': '#909090',
    'N': '#3050F8',   'O': '#FF0D0D',   'F': '#90E050',   'Ne': '#B3E3F5',
    'Na': '#AB5CF2',  'Mg': '#8AFF00',  'Al': '#BFA6A6',  'Si': '#F0C8A0',
    'P': '#FF8000',   'S': '#FFFF30',   'Cl': '#1FF01F',  'Ar': '#80D1E3',
    'K': '#8F40D4',   'Ca': '#3DFF00',  'Sc': '#E6E6E6',  'Ti': '#BFC2C7',
    'V': '#A6A6AB',   'Cr': '#8A99C7',  'Mn': '#9C7AC7',  'Fe': '#E06633',
    'Co': '#F090A0',  'Ni': '#50D050',  'Cu': '#C78033',  'Zn': '#7D80B0',
    'Ga': '#C28F8F',  'Ge': '#668F8F',  'As': '#BD80E3',  'Se': '#FFA100',
    'Br': '#A62929',  'Kr': '#5CB8D1',  'Rb': '#702EB0',  'Sr': '#00FF00',
    'Y': '#94FFFF',   'Zr': '#94E0E0',  'Nb': '#73C2C9',  'Mo': '#54B5B5',
    'Tc': '#3B9E9E',  'Ru': '#248F8F',  'Rh': '#0A7D8C',  'Pd': '#006985',
    'Ag': '#C0C0C0',  'Cd': '#FFD98F',  'In': '#A67573',  'Sn': '#668080',
    'Sb': '#9E63B5',  'Te': '#D47A00',  'I': '#940094',   'Xe': '#429EB0',
    'Cs': '#57178F',  'Ba': '#00C900',  'La': '#70D4FF',  'Hf': '#4DC2FF',
    'Ta': '#4DA6FF',  'W': '#2194D6',   'Re': '#267DAB',  'Os': '#266696',
    'Ir': '#175487',  'Pt': '#D0D0E0',  'Au': '#FFD123',  'Hg': '#B8B8D0',
    'Tl': '#A6544D',  'Pb': '#575961',  'Bi': '#9E4FB5',  'Po': '#AB5C00',
    'At': '#754F45',  'Rn': '#428296',  'Fr': '#420066',  'Ra': '#007D00',
    'Ac': '#70ABFA',  'Th': '#00BAFF',  'Pa': '#00A1FF',  'U': '#008FFF',
    'Np': '#0080FF',  'Pu': '#006BFF',  'Am': '#545CF2',  'Cm': '#785CE3',
    'Bk': '#8A4FE3',  'Cf': '#A136D4',  'Es': '#B31FD4',  'Fm': '#B31FBA',
    'Md': '#B30DA6',  'No': '#BD0D87',  'Lr': '#C70066',  'Rf': '#CC0059',
    'Db': '#D1004F',  'Sg': '#D90045',  'Bh': '#E00038',  'Hs': '#E6002E',
    'Mt': '#EB0026',
}

DEFAULT_ELEMENT_COLOR = '#888888'

# ─────────────────────────────────────────────────────────────
# Edge type styling
# ─────────────────────────────────────────────────────────────
EDGE_STYLES = {
    'covalent': {'color': '#808080', 'width': 2, 'dash': 'solid', 'opacity': 0.6},
    'HB_A':     {'color': '#FF6B6B', 'width': 3, 'dash': 'solid', 'opacity': 0.8},
    'HB_B':     {'color': '#4ECDC4', 'width': 4, 'dash': 'solid', 'opacity': 0.9},
    'HB_C':     {'color': '#45B7D1', 'width': 2, 'dash': 'solid', 'opacity': 0.7},
    'sigma_A':  {'color': '#FFA500', 'width': 3, 'dash': 'dot',   'opacity': 0.7},
    'sigma_B':  {'color': '#32CD32', 'width': 4, 'dash': 'dot',   'opacity': 0.8},
    'sigma_C':  {'color': '#8A2BE2', 'width': 2, 'dash': 'dot',   'opacity': 0.7},
}

SUSPICIOUS_STYLE = {'color': '#FF00FF', 'width': 5, 'dash': 'dash', 'opacity': 1.0}


def _get_element_color(element):
    """Return CPK color for element."""
    return ELEMENT_COLORS.get(element, DEFAULT_ELEMENT_COLOR)


def _get_edge_style(edge_type, is_suspicious=False):
    """Return Plotly line style for edge type."""
    if is_suspicious:
        return SUSPICIOUS_STYLE.copy()
    return EDGE_STYLES.get(edge_type, {'color': '#999999', 'width': 2, 'dash': 'solid', 'opacity': 0.5})


def _get_atom_size(element):
    """Return relative atom size based on element."""
    sizes = {
        'H': 8, 'He': 10,
        'Li': 18, 'Be': 14, 'B': 16, 'C': 16, 'N': 16, 'O': 16, 'F': 16, 'Ne': 16,
        'Na': 22, 'Mg': 20, 'Al': 20, 'Si': 20, 'P': 20, 'S': 20, 'Cl': 20, 'Ar': 20,
        'K': 26, 'Ca': 24,
    }
    return sizes.get(element, 18)


# ═════════════════════════════════════════════════════════════
# ATOMIC NUMBER ↔ ELEMENT MAPPING
# ═════════════════════════════════════════════════════════════

_Z_TO_ELEM = {
    1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F',
    10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl',
    18: 'Ar', 19: 'K', 20: 'Ca', 21: 'Sc', 22: 'Ti', 23: 'V', 24: 'Cr', 25: 'Mn',
    26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu', 30: 'Zn', 31: 'Ga', 32: 'Ge', 33: 'As',
    34: 'Se', 35: 'Br', 36: 'Kr', 37: 'Rb', 38: 'Sr', 39: 'Y', 40: 'Zr', 41: 'Nb',
    42: 'Mo', 43: 'Tc', 44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd', 49: 'In',
    50: 'Sn', 51: 'Sb', 52: 'Te', 53: 'I', 54: 'Xe', 55: 'Cs', 56: 'Ba', 57: 'La',
    58: 'Ce', 59: 'Pr', 60: 'Nd', 61: 'Pm', 62: 'Sm', 63: 'Eu', 64: 'Gd', 65: 'Tb',
    66: 'Dy', 67: 'Ho', 68: 'Er', 69: 'Tm', 70: 'Yb', 71: 'Lu', 72: 'Hf', 73: 'Ta',
    74: 'W', 75: 'Re', 76: 'Os', 77: 'Ir', 78: 'Pt', 79: 'Au', 80: 'Hg', 81: 'Tl',
    82: 'Pb', 83: 'Bi', 84: 'Po', 85: 'At', 86: 'Rn',
}

_ELEM_TO_Z = {v: k for k, v in _Z_TO_ELEM.items()}


def _get_element_from_z(atomic_num):
    """Get element symbol from atomic number."""
    return _Z_TO_ELEM.get(int(atomic_num), 'X')


def _get_z_from_element(element):
    """Get atomic number from element symbol."""
    return _ELEM_TO_Z.get(element, 0)


# ═════════════════════════════════════════════════════════════
# GRAPH RECONSTRUCTION FROM JSON
# ═════════════════════════════════════════════════════════════

def reconstruct_graph_from_json(data):
    """
    Reconstruct a NetworkX graph from the JSON output saved by main.py.

    The JSON format from main.py:
    {
        'filename': str,
        'num_atoms': int,
        'rule': str,
        'interaction_type': str,
        'statistics': dict,
        'edge_list': [[i, j, type], ...],
        'edge_index': [[u, v], ...] or [[2, N], ...],
        'node_features': [[atomic_num, fragment_id, x, y, z], ...]
    }
    """
    G = nx.Graph()

    node_features = data.get('node_features', [])
    edge_list = data.get('edge_list', [])

    # Add nodes
    for i, feat in enumerate(node_features):
        if len(feat) >= 5:
            atomic_num = int(feat[0])
            fragment_id = int(feat[1])
            coords = (float(feat[2]), float(feat[3]), float(feat[4]))
            element = _get_element_from_z(atomic_num)
        else:
            atomic_num = 0
            fragment_id = 0
            coords = (0.0, 0.0, 0.0)
            element = 'X'

        G.add_node(i,
                   element=element,
                   atomic_num=atomic_num,
                   fragment_id=fragment_id,
                   coords=coords)

    # Add edges with attributes
    for edge in edge_list:
        if len(edge) >= 3:
            i, j = int(edge[0]), int(edge[1])
            etype = str(edge[2])
            G.add_edge(i, j, type=etype)

            # Try to add distance/angle from statistics if available
            stats = data.get('statistics', {})
            suspicious = stats.get('suspicious_contacts', [])
            for sc in suspicious:
                if sc.get('donor') == i and sc.get('acceptor') == j:
                    G.edges[i, j]['distance'] = sc.get('distance', 0.0)
                    G.edges[i, j]['angle'] = sc.get('angle', 0.0)
                    break

    return G


# ═════════════════════════════════════════════════════════════
# PLOTLY VISUALIZATIONS
# ═════════════════════════════════════════════════════════════

def visualize_structure_plotly(G_full, title="Molecular Graph with NCI",
                                save_path=None, show=True,
                                highlight_suspicious=True,
                                suspicious_threshold_dist=2.8,
                                suspicious_threshold_angle=100,
                                width=1000, height=800):
    """
    Interactive 3D visualization of molecular graph with NCI using Plotly.
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is required. Install: pip install 'plotly>=5.0.0'")

    # Extract node positions and elements
    pos = {}
    elements = {}
    for node, data in G_full.nodes(data=True):
        pos[node] = data.get('coords', (0, 0, 0))
        elements[node] = data.get('element', 'X')

    # ── Build edge traces ────────────────────────────────────
    edge_traces = []
    suspicious_edges = []

    for u, v, d in G_full.edges(data=True):
        etype = d.get('type', 'unknown')
        x0, y0, z0 = pos[u]
        x1, y1, z1 = pos[v]

        # Check suspicious
        is_suspicious = False
        if highlight_suspicious and etype == 'HB_A':
            dist = d.get('distance', 0.0)
            angle = d.get('angle', 0.0)
            if dist > suspicious_threshold_dist or angle < suspicious_threshold_angle:
                is_suspicious = True
                suspicious_edges.append((u, v, dist, angle))

        style = _get_edge_style(etype, is_suspicious)

        trace = go.Scatter3d(
            x=[x0, x1, None],
            y=[y0, y1, None],
            z=[z0, z1, None],
            mode='lines',
            line=dict(
                color=style['color'],
                width=style['width'],
                dash=style['dash']
            ),
            opacity=style['opacity'],
            hoverinfo='text',
            hovertext=(
                f"Edge: {elements.get(u, '?')}{u} — {elements.get(v, '?')}{v}<br>"
                f"Type: {etype}<br>"
                f"Distance: {d.get('distance', 'N/A'):.3f} Å<br>"
                f"Angle: {d.get('angle', 'N/A'):.1f}°"
                if not is_suspicious else
                f"⚠️ SUSPICIOUS<br>"
                f"Edge: {elements.get(u, '?')}{u} — {elements.get(v, '?')}{v}<br>"
                f"Type: {etype}<br>"
                f"Distance: {d.get('distance', 'N/A'):.3f} Å<br>"
                f"Angle: {d.get('angle', 'N/A'):.1f}°"
            ),
            showlegend=False,
            name=etype
        )
        edge_traces.append(trace)

    # ── Build node trace ─────────────────────────────────────
    node_x, node_y, node_z = [], [], []
    node_colors, node_sizes, node_texts = [], [], []

    for node in G_full.nodes():
        x, y, z = pos[node]
        elem = elements[node]
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)
        node_colors.append(_get_element_color(elem))
        node_sizes.append(_get_atom_size(elem))

        # Build hover text
        neighbors = list(G_full.neighbors(node))
        nci_count = sum(1 for n in neighbors if not G_full.edges[node, n].get('type', '').startswith('covalent'))
        text = (
            f"<b>{elem}{node}</b><br>"
            f"Index: {node}<br>"
            f"Coords: ({x:.3f}, {y:.3f}, {z:.3f}) Å<br>"
            f"NCI bonds: {nci_count}<br>"
            f"Fragment: {G_full.nodes[node].get('fragment_id', 'N/A')}"
        )
        node_texts.append(text)

    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers+text',
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(color='black', width=1),
            opacity=0.95,
            sizemode='diameter'
        ),
        text=[f"{elements[n]}{n}" for n in G_full.nodes()],
        textposition='top center',
        textfont=dict(size=10, color='black'),
        hoverinfo='text',
        hovertext=node_texts,
        showlegend=False,
        name='Atoms'
    )

    # ── Build legend traces (invisible, for legend only) ─────
    legend_traces = [
        go.Scatter3d(x=[None], y=[None], z=[None], mode='lines',
                     line=dict(color='#808080', width=2),
                     name='➖ Covalent', showlegend=True, hoverinfo='skip'),
        go.Scatter3d(x=[None], y=[None], z=[None], mode='lines',
                     line=dict(color='#FF6B6B', width=3),
                     name='➖ HB_A (soft)', showlegend=True, hoverinfo='skip'),
        go.Scatter3d(x=[None], y=[None], z=[None], mode='lines',
                     line=dict(color='#4ECDC4', width=4),
                     name='➖ HB_B (sensible)', showlegend=True, hoverinfo='skip'),
        go.Scatter3d(x=[None], y=[None], z=[None], mode='lines',
                     line=dict(color='#45B7D1', width=2),
                     name='➖ HB_C (strict)', showlegend=True, hoverinfo='skip'),
        go.Scatter3d(x=[None], y=[None], z=[None], mode='lines',
                     line=dict(color='#FFA500', width=3, dash='dot'),
                     name='➖ Sigma_A', showlegend=True, hoverinfo='skip'),
        go.Scatter3d(x=[None], y=[None], z=[None], mode='lines',
                     line=dict(color='#32CD32', width=4, dash='dot'),
                     name='➖ Sigma_B', showlegend=True, hoverinfo='skip'),
        go.Scatter3d(x=[None], y=[None], z=[None], mode='lines',
                     line=dict(color='#8A2BE2', width=2, dash='dot'),
                     name='➖ Sigma_C', showlegend=True, hoverinfo='skip'),
    ]
    if highlight_suspicious:
        legend_traces.append(
            go.Scatter3d(x=[None], y=[None], z=[None], mode='lines',
                         line=dict(color='#FF00FF', width=5, dash='dash'),
                         name='⚠️ Suspicious (HB_A)', showlegend=True, hoverinfo='skip')
        )

    # ── Assemble figure ──────────────────────────────────────
    all_traces = legend_traces + edge_traces + [node_trace]

    fig = go.Figure(data=all_traces)

    # Add annotation for suspicious contacts
    annotations = []
    if suspicious_edges:
        info_text = f"Suspicious contacts (HB_A): {len(suspicious_edges)}<br>"
        for u, v, dist, angle in suspicious_edges[:5]:
            info_text += f"  {elements.get(u, '?')}{u}—{elements.get(v, '?')}{v}: d={dist:.2f}Å, a={angle:.1f}°<br>"
        if len(suspicious_edges) > 5:
            info_text += f"  ... and {len(suspicious_edges) - 5} more"
        annotations.append(dict(
            x=0.02, y=0.02,
            xref='paper', yref='paper',
            text=info_text,
            showarrow=False,
            font=dict(size=11, color='#8B0000'),
            bgcolor='rgba(255, 235, 205, 0.8)',
            bordercolor='#8B4513',
            borderwidth=1,
            borderpad=4,
            align='left'
        ))

    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            x=0.5,
            font=dict(size=16)
        ),
        scene=dict(
            xaxis=dict(title='X (Å)', showbackground=True, backgroundcolor='rgb(240,240,240)'),
            yaxis=dict(title='Y (Å)', showbackground=True, backgroundcolor='rgb(240,240,240)'),
            zaxis=dict(title='Z (Å)', showbackground=True, backgroundcolor='rgb(240,240,240)'),
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        width=width,
        height=height,
        legend=dict(
            x=0.01, y=0.99,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='gray',
            borderwidth=1
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        annotations=annotations,
        template='plotly_white'
    )

    if save_path:
        fig.write_html(save_path)
        print(f"Saved interactive visualization to {save_path}")

    if show:
        fig.show()

    return fig


def visualize_compare_rules_plotly(G_A, G_B, G_C, title_prefix="",
                                    save_path=None, show=True,
                                    width=1400, height=600):
    """
    Side-by-side interactive comparison of rules A/B/C using Plotly subplots.
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is required. Install: pip install 'plotly>=5.0.0'")

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[f"{title_prefix} A (soft)", f"{title_prefix} B (sensible)", f"{title_prefix} C (strict)"],
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}]]
    )

    graphs = [
        ('A', G_A, '#FF6B6B'),
        ('B', G_B, '#4ECDC4'),
        ('C', G_C, '#45B7D1')
    ]

    for col_idx, (label, G, color) in enumerate(graphs, 1):
        pos = {node: data.get('coords', (0, 0, 0)) for node, data in G.nodes(data=True)}
        elements = {node: data.get('element', 'X') for node, data in G.nodes(data=True)}

        # Covalent edges
        for u, v, d in G.edges(data=True):
            if d.get('type') == 'covalent':
                x = [pos[u][0], pos[v][0], None]
                y = [pos[u][1], pos[v][1], None]
                z = [pos[u][2], pos[v][2], None]
                fig.add_trace(go.Scatter3d(
                    x=x, y=y, z=z,
                    mode='lines',
                    line=dict(color='#808080', width=1),
                    opacity=0.5,
                    showlegend=False,
                    hoverinfo='skip'
                ), row=1, col=col_idx)

        # NCI edges
        for u, v, d in G.edges(data=True):
            etype = d.get('type', '')
            if etype.startswith('HB_') or etype.startswith('sigma_'):
                x = [pos[u][0], pos[v][0], None]
                y = [pos[u][1], pos[v][1], None]
                z = [pos[u][2], pos[v][2], None]
                fig.add_trace(go.Scatter3d(
                    x=x, y=y, z=z,
                    mode='lines',
                    line=dict(color=color, width=3),
                    opacity=0.8,
                    showlegend=False,
                    hoverinfo='text',
                    hovertext=f"{elements[u]}{u}—{elements[v]}{v}<br>Type: {etype}"
                ), row=1, col=col_idx)

        # Atoms
        node_x, node_y, node_z = [], [], []
        node_colors, node_sizes = [], []
        for node in G.nodes():
            x, y, z = pos[node]
            elem = elements[node]
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)
            node_colors.append(_get_element_color(elem))
            node_sizes.append(_get_atom_size(elem) * 0.6)

        fig.add_trace(go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers',
            marker=dict(size=node_sizes, color=node_colors,
                        line=dict(color='black', width=0.5)),
            showlegend=False,
            hoverinfo='text',
            hovertext=[f"{elements[n]}{n}" for n in G.nodes()]
        ), row=1, col=col_idx)

        # Count HB edges for subtitle
        hb_count = sum(1 for _, _, d in G.edges(data=True) if d.get('type', '').startswith('HB_'))
        sigma_count = sum(1 for _, _, d in G.edges(data=True) if d.get('type', '').startswith('sigma_'))

        fig.layout.annotations[col_idx - 1].text = (
            f"<b>{title_prefix} {label}</b><br>"
            f"<span style='font-size:12px'>HB: {hb_count} | Sigma: {sigma_count}</span>"
        )

    fig.update_layout(
        title=dict(text="<b>Rule Comparison (A/B/C)</b>", x=0.5, font=dict(size=16)),
        scene=dict(aspectmode='data'),
        scene2=dict(aspectmode='data'),
        scene3=dict(aspectmode='data'),
        width=width,
        height=height,
        template='plotly_white'
    )

    if save_path:
        fig.write_html(save_path)
        print(f"Saved comparison to {save_path}")

    if show:
        fig.show()

    return fig


def visualize_hb_network_plotly(G_full, save_path=None, show=True,
                                  width=900, height=700):
    """
    Interactive 2D visualization of H-bond network using Plotly.
    Spring layout for NCI-only subgraph.
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is required. Install: pip install 'plotly>=5.0.0'")

    # Extract HB subgraph
    G_hb = nx.Graph()
    elements = {}
    for node, data in G_full.nodes(data=True):
        elements[node] = data.get('element', 'X')

    hb_edges = []
    for u, v, d in G_full.edges(data=True):
        if d.get('type', '').startswith('HB_'):
            G_hb.add_edge(u, v, **d)

    if G_hb.number_of_edges() == 0:
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5,
            text="No H-bonds found",
            showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(width=width, height=height)
        return fig

    # Spring layout
    pos = nx.spring_layout(G_hb, k=2, iterations=50, seed=42)

    # Edge traces
    edge_traces = []
    for u, v, d in G_hb.edges(data=True):
        etype = d.get('type', 'HB_B')
        if etype == 'HB_A':
            color = '#FF6B6B'
        elif etype == 'HB_B':
            color = '#4ECDC4'
        else:
            color = '#45B7D1'

        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode='lines',
            line=dict(color=color, width=3),
            opacity=0.7,
            hoverinfo='text',
            hovertext=f"{elements[u]}{u} — {elements[v]}{v}<br>Type: {etype}",
            showlegend=False
        ))

    # Node trace
    node_x, node_y = [], []
    node_colors, node_texts = [], []
    for node in G_hb.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_colors.append(_get_element_color(elements[node]))
        node_texts.append(f"{elements[node]}{node}")

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        marker=dict(size=20, color=node_colors,
                    line=dict(color='black', width=1)),
        text=node_texts,
        textposition='top center',
        textfont=dict(size=10),
        hoverinfo='text',
        hovertext=[f"<b>{elements[n]}{n}</b><br>Degree: {G_hb.degree(n)}" for n in G_hb.nodes()],
        showlegend=False
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        title=dict(text="<b>H-bond Network</b>", x=0.5),
        width=width,
        height=height,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        template='plotly_white',
        hovermode='closest'
    )

    if save_path:
        fig.write_html(save_path)
        print(f"Saved HB network to {save_path}")

    if show:
        fig.show()

    return fig


def visualize_interactive_dashboard(G_full, save_path=None, show=True,
                                     width=1200, height=900):
    """
    Create an interactive dashboard with multiple views:
    - 3D molecular graph
    - 2D H-bond network
    - Statistics panel
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is required. Install: pip install 'plotly>=5.0.0'")

    # Try to import metrics, but don't fail if unavailable
    try:
        from metrics import compute_hb_statistics, compute_sigma_statistics
        hb_stats = compute_hb_statistics(G_full)
        sigma_stats = compute_sigma_statistics(G_full)
    except Exception:
        hb_stats = {}
        sigma_stats = {}

    # Build 3D view (simplified)
    pos = {node: data.get('coords', (0, 0, 0)) for node, data in G_full.nodes(data=True)}
    elements = {node: data.get('element', 'X') for node, data in G_full.nodes(data=True)}

    fig = make_subplots(
        rows=2, cols=2,
        specs=[
            [{'type': 'scatter3d', 'colspan': 2}, None],
            [{'type': 'scatter'}, {'type': 'table'}]
        ],
        subplot_titles=['3D Molecular Graph', 'H-bond Network', 'Statistics'],
        row_heights=[0.65, 0.35]
    )

    # 3D plot (top, spans both columns)
    for u, v, d in G_full.edges(data=True):
        etype = d.get('type', 'unknown')
        x = [pos[u][0], pos[v][0], None]
        y = [pos[u][1], pos[v][1], None]
        z = [pos[u][2], pos[v][2], None]
        style = _get_edge_style(etype)
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines',
            line=dict(color=style['color'], width=style['width']),
            opacity=style['opacity'],
            showlegend=False,
            hoverinfo='skip'
        ), row=1, col=1)

    node_x, node_y, node_z = [], [], []
    node_colors, node_sizes = [], []
    for node in G_full.nodes():
        x, y, z = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)
        node_colors.append(_get_element_color(elements[node]))
        node_sizes.append(_get_atom_size(elements[node]))

    fig.add_trace(go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers',
        marker=dict(size=node_sizes, color=node_colors,
                    line=dict(color='black', width=1)),
        showlegend=False,
        hovertext=[f"{elements[n]}{n}" for n in G_full.nodes()],
        hoverinfo='text'
    ), row=1, col=1)

    # 2D network (bottom left)
    G_hb = get_hb_subgraph(G_full)
    if G_hb.number_of_edges() > 0:
        pos_2d = nx.spring_layout(G_hb, k=2, iterations=50, seed=42)
        for u, v, d in G_hb.edges(data=True):
            etype = d.get('type', 'HB_B')
            color = '#FF6B6B' if etype == 'HB_A' else '#4ECDC4' if etype == 'HB_B' else '#45B7D1'
            x = [pos_2d[u][0], pos_2d[v][0], None]
            y = [pos_2d[u][1], pos_2d[v][1], None]
            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode='lines',
                line=dict(color=color, width=2),
                opacity=0.7,
                showlegend=False,
                hoverinfo='skip'
            ), row=2, col=1)

        node_x2, node_y2 = [], []
        for node in G_hb.nodes():
            node_x2.append(pos_2d[node][0])
            node_y2.append(pos_2d[node][1])
        fig.add_trace(go.Scatter(
            x=node_x2, y=node_y2,
            mode='markers+text',
            marker=dict(size=15, color='lightblue',
                        line=dict(color='black', width=1)),
            text=[f"{elements[n]}{n}" for n in G_hb.nodes()],
            textposition='top center',
            textfont=dict(size=8),
            showlegend=False,
            hoverinfo='text',
            hovertext=[f"{elements[n]}{n}<br>Degree: {G_hb.degree(n)}" for n in G_hb.nodes()]
        ), row=2, col=1)

    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, row=2, col=1)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, row=2, col=1)

    # Statistics table (bottom right)
    table_data = {
        'Metric': [
            'H-bonds', 'Avg distance', 'Avg angle', 'Max component',
            'Suspicious', 'Sigma bonds', 'Avg distance (σ)', 'Avg angle 1 (σ)'
        ],
        'Value': [
            hb_stats.get('num_hb', 0),
            f"{hb_stats.get('avg_distance', 0):.3f} Å",
            f"{hb_stats.get('avg_angle', 0):.1f}°",
            hb_stats.get('max_component_size', 0),
            hb_stats.get('num_suspicious', 0),
            sigma_stats.get('num_sigma', 0),
            f"{sigma_stats.get('avg_distance', 0):.3f} Å",
            f"{sigma_stats.get('avg_angle_1', 0):.1f}°",
        ]
    }

    fig.add_trace(go.Table(
        header=dict(
            values=['<b>Metric</b>', '<b>Value</b>'],
            fill_color='lightblue',
            align='left',
            font=dict(size=12)
        ),
        cells=dict(
            values=[table_data['Metric'], table_data['Value']],
            fill_color='white',
            align='left',
            font=dict(size=11),
            height=25
        )
    ), row=2, col=2)

    fig.update_layout(
        title=dict(text="<b>MolGraph-NCI Dashboard</b>", x=0.5, font=dict(size=18)),
        width=width,
        height=height,
        template='plotly_white',
        scene=dict(aspectmode='data')
    )

    if save_path:
        fig.write_html(save_path)
        print(f"Saved dashboard to {save_path}")

    if show:
        fig.show()

    return fig


# ═════════════════════════════════════════════════════════════
# BACKWARD COMPATIBILITY: Matplotlib wrappers
# (keep original function signatures for compatibility with main.py)
# ═════════════════════════════════════════════════════════════

def visualize_structure(G_full, title="Structure with H-bonds", save_path=None,
                        figsize=(12, 10), highlight_suspicious=True,
                        suspicious_threshold_dist=2.8, suspicious_threshold_angle=100):
    """
    Backward-compatible wrapper. Uses Plotly if available, falls back to matplotlib.
    When called from main.py with .png path, saves BOTH .png (matplotlib) and .html (Plotly)
    if Plotly is available.
    """
    # If save_path ends with .png and Plotly is available, also save .html
    html_path = None
    if save_path and PLOTLY_AVAILABLE and save_path.endswith('.png'):
        html_path = save_path[:-4] + '.html'

    if PLOTLY_AVAILABLE and html_path:
        # Save interactive HTML
        try:
            visualize_structure_plotly(
                G_full, title=title, save_path=html_path, show=False,
                highlight_suspicious=highlight_suspicious,
                suspicious_threshold_dist=suspicious_threshold_dist,
                suspicious_threshold_angle=suspicious_threshold_angle,
                width=figsize[0] * 80, height=figsize[1] * 80
            )
        except Exception as e:
            print(f"Plotly HTML save failed: {e}")

    # Always generate matplotlib PNG (for backward compatibility)
    return _visualize_structure_mpl(
        G_full, title=title, save_path=save_path,
        figsize=figsize, highlight_suspicious=highlight_suspicious,
        suspicious_threshold_dist=suspicious_threshold_dist,
        suspicious_threshold_angle=suspicious_threshold_angle
    )


def visualize_compare_rules(G_A, G_B, G_C, title_prefix="",
                             save_path=None, figsize=(18, 5)):
    """
    Backward-compatible wrapper for rule comparison.
    """
    html_path = None
    if save_path and PLOTLY_AVAILABLE and save_path.endswith('.png'):
        html_path = save_path[:-4] + '.html'

    if PLOTLY_AVAILABLE and html_path:
        try:
            visualize_compare_rules_plotly(
                G_A, G_B, G_C, title_prefix=title_prefix,
                save_path=html_path, show=False,
                width=figsize[0] * 80, height=figsize[1] * 80
            )
        except Exception as e:
            print(f"Plotly comparison HTML save failed: {e}")

    return _visualize_compare_rules_mpl(
        G_A, G_B, G_C, title_prefix=title_prefix,
        save_path=save_path, figsize=figsize
    )


def visualize_hb_network(G_full, save_path=None, figsize=(10, 8)):
    """
    Backward-compatible wrapper for HB network visualization.
    """
    html_path = None
    if save_path and PLOTLY_AVAILABLE and save_path.endswith('.png'):
        html_path = save_path[:-4] + '.html'

    if PLOTLY_AVAILABLE and html_path:
        try:
            visualize_hb_network_plotly(
                G_full, save_path=html_path, show=False,
                width=figsize[0] * 80, height=figsize[1] * 80
            )
        except Exception as e:
            print(f"Plotly network HTML save failed: {e}")

    return _visualize_hb_network_mpl(
        G_full, save_path=save_path, figsize=figsize
    )


# ═════════════════════════════════════════════════════════════
# MATPLOTLIB FALLBACK IMPLEMENTATIONS (original code preserved)
# ═════════════════════════════════════════════════════════════

def _visualize_structure_mpl(G_full, title="Structure with H-bonds", save_path=None,
                              figsize=(12, 10), highlight_suspicious=True,
                              suspicious_threshold_dist=2.8, suspicious_threshold_angle=100):
    """Original matplotlib implementation (fallback)."""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    pos = {}
    elements = {}
    for node, data in G_full.nodes(data=True):
        pos[node] = data.get('coords', (0, 0, 0))
        elements[node] = data.get('element', 'X')

    color_map = {
        'H': '#FFFFFF', 'C': '#909090', 'N': '#3050F8', 'O': '#FF0D0D',
        'F': '#90E050', 'P': '#FF8000', 'S': '#FFFF30', 'Cl': '#1FF01F',
        'Br': '#A62929', 'I': '#940094', 'Fe': '#E06633', 'Pd': '#698598'
    }
    default_color = '#888888'

    # Covalent edges
    covalent_edges = [(u, v) for u, v, d in G_full.edges(data=True)
                      if d.get('type') == 'covalent']
    for u, v in covalent_edges:
        x = [pos[u][0], pos[v][0]]
        y = [pos[u][1], pos[v][1]]
        z = [pos[u][2], pos[v][2]]
        ax.plot(x, y, z, color='gray', linewidth=1.0, alpha=0.6)

    # HB edges
    hb_colors = {'HB_A': '#FF6B6B', 'HB_B': '#4ECDC4', 'HB_C': '#45B7D1'}
    suspicious_edges = []
    for u, v, d in G_full.edges(data=True):
        etype = d.get('type', '')
        if etype.startswith('HB_'):
            color = hb_colors.get(etype, '#999999')
            linewidth = 2.5 if etype == 'HB_B' else 1.5
            alpha = 0.9 if etype == 'HB_B' else 0.6

            is_suspicious = False
            if highlight_suspicious and etype == 'HB_A':
                dist = d.get('distance', 0.0)
                angle = d.get('angle', 0.0)
                if dist > suspicious_threshold_dist or angle < suspicious_threshold_angle:
                    is_suspicious = True
                    suspicious_edges.append((u, v, dist, angle))

            x = [pos[u][0], pos[v][0]]
            y = [pos[u][1], pos[v][1]]
            z = [pos[u][2], pos[v][2]]

            if is_suspicious:
                ax.plot(x, y, z, color='#FF00FF', linewidth=3.0, alpha=1.0, linestyle='--')
            else:
                ax.plot(x, y, z, color=color, linewidth=linewidth, alpha=alpha)

    # Sigma edges
    sigma_colors = {'sigma_A': '#FFA500', 'sigma_B': '#32CD32', 'sigma_C': '#8A2BE2'}
    for u, v, d in G_full.edges(data=True):
        etype = d.get('type', '')
        if etype.startswith('sigma_'):
            color = sigma_colors.get(etype, '#999999')
            x = [pos[u][0], pos[v][0]]
            y = [pos[u][1], pos[v][1]]
            z = [pos[u][2], pos[v][2]]
            ax.plot(x, y, z, color=color, linewidth=2.0, linestyle=':', alpha=0.7)

    # Atoms
    for node, coords in pos.items():
        elem = elements[node]
        color = color_map.get(elem, default_color)
        size = 80 if elem == 'H' else 200
        ax.scatter(*coords, c=color, s=size, edgecolors='black', linewidths=0.5, alpha=0.9)

    # Labels
    for node, coords in pos.items():
        ax.text(coords[0], coords[1], coords[2], f' {elements[node]}{node}', fontsize=7, alpha=0.7)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='gray', lw=1.0, label='Covalent'),
        Line2D([0], [0], color='#FF6B6B', lw=2.0, label='HB_A (soft)'),
        Line2D([0], [0], color='#4ECDC4', lw=2.5, label='HB_B (sensible)'),
        Line2D([0], [0], color='#45B7D1', lw=1.5, label='HB_C (strict)'),
        Line2D([0], [0], color='#FF00FF', lw=3.0, linestyle='--', label='Suspicious (A)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=8)

    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    ax.set_title(title)

    if suspicious_edges:
        info_text = f"Suspicious contacts (HB_A): {len(suspicious_edges)}\n"
        for u, v, dist, angle in suspicious_edges[:5]:
            info_text += f" {u}-{v}: d={dist:.2f}Å, a={angle:.1f}°\n"
        ax.text2D(0.02, 0.02, info_text, transform=ax.transAxes, fontsize=8,
                  verticalalignment='bottom',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()

    return fig, ax


def _visualize_compare_rules_mpl(G_A, G_B, G_C, title_prefix="",
                                  save_path=None, figsize=(18, 5)):
    """Original matplotlib comparison implementation (fallback)."""
    fig = plt.figure(figsize=figsize)
    graphs = [('A (soft)', G_A, '#FF6B6B'),
              ('B (sensible)', G_B, '#4ECDC4'),
              ('C (strict)', G_C, '#45B7D1')]

    for idx, (label, G, color) in enumerate(graphs, 1):
        ax = fig.add_subplot(1, 3, idx, projection='3d')
        pos = {node: data.get('coords', (0, 0, 0)) for node, data in G.nodes(data=True)}
        elements = {node: data.get('element', 'X') for node, data in G.nodes(data=True)}

        for u, v, d in G.edges(data=True):
            if d.get('type') == 'covalent':
                x = [pos[u][0], pos[v][0]]
                y = [pos[u][1], pos[v][1]]
                z = [pos[u][2], pos[v][2]]
                ax.plot(x, y, z, color='gray', linewidth=0.8, alpha=0.5)

        for u, v, d in G.edges(data=True):
            if d.get('type', '').startswith('HB_'):
                x = [pos[u][0], pos[v][0]]
                y = [pos[u][1], pos[v][1]]
                z = [pos[u][2], pos[v][2]]
                ax.plot(x, y, z, color=color, linewidth=2.0, alpha=0.8)

        for node, coords in pos.items():
            ax.scatter(*coords, c='black', s=50, alpha=0.8)

        hb_edges = [e for e in G.edges(data=True) if e[2].get('type', '').startswith('HB_')]
        ax.set_title(f"{title_prefix} {label}\n{len(hb_edges)} H-bonds")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    return fig


def _visualize_hb_network_mpl(G_full, save_path=None, figsize=(10, 8)):
    """Original matplotlib HB network implementation (fallback)."""
    fig, ax = plt.subplots(figsize=figsize)
    G_hb = nx.Graph()
    for u, v, d in G_full.edges(data=True):
        if d.get('type', '').startswith('HB_'):
            G_hb.add_edge(u, v, **d)

    if G_hb.number_of_edges() == 0:
        ax.text(0.5, 0.5, "No H-bonds found", ha='center', va='center', fontsize=14)
        if save_path:
            plt.savefig(save_path, dpi=150)
        return fig, ax

    pos = nx.spring_layout(G_hb, k=2, iterations=50)
    edge_colors = []
    for u, v, d in G_hb.edges(data=True):
        etype = d.get('type', 'HB_B')
        if etype == 'HB_A':
            edge_colors.append('#FF6B6B')
        elif etype == 'HB_B':
            edge_colors.append('#4ECDC4')
        else:
            edge_colors.append('#45B7D1')

    nx.draw_networkx_nodes(G_hb, pos, node_color='lightblue', node_size=300, ax=ax)
    nx.draw_networkx_labels(G_hb, pos, font_size=8, ax=ax)
    nx.draw_networkx_edges(G_hb, pos, edge_color=edge_colors, width=2.0, alpha=0.7, ax=ax)

    ax.set_title("H-bond Network")
    ax.axis('off')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    return fig, ax


# ═════════════════════════════════════════════════════════════
# CLI ENTRY POINT for standalone visualization
# ═════════════════════════════════════════════════════════════

def main():
    """CLI entry point for standalone visualization."""
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description='MolGraph-NCI: Interactive Molecular Graph Visualizer (Plotly)'
    )
    parser.add_argument('input', help='JSON result file from MolGraph-NCI or XYZ file')
    parser.add_argument('--format', choices=['json', 'xyz'], default=None,
                        help='Input format (auto-detected if not specified)')
    parser.add_argument('--mode', choices=['structure', 'compare', 'network', 'dashboard'],
                        default='structure',
                        help='Visualization mode')
    parser.add_argument('--output', '-o', default=None,
                        help='Output HTML file path')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not open browser, just save')
    parser.add_argument('--width', type=int, default=1000,
                        help='Figure width in pixels')
    parser.add_argument('--height', type=int, default=800,
                        help='Figure height in pixels')
    parser.add_argument('--rule', choices=['A', 'B', 'C'], default='B',
                        help='Rule for XYZ mode')
    parser.add_argument('--inter', choices=['HB', 'sigma'], default='HB',
                        help='Interaction type for XYZ mode')

    args = parser.parse_args()

    # Auto-detect format if not specified
    input_path = args.input
    if args.format is None:
        if input_path.endswith('.json'):
            args.format = 'json'
        elif input_path.endswith('.xyz'):
            args.format = 'xyz'
        else:
            print("Error: Cannot auto-detect format. Please specify --format json or --format xyz")
            sys.exit(1)

    # Check file exists
    if not os.path.exists(input_path):
        print(f"Error: File not found: {input_path}")
        print(f"Current directory: {os.getcwd()}")
        print(f"Absolute path tried: {os.path.abspath(input_path)}")
        sys.exit(1)

    if args.format == 'json':
        # Load from JSON result
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Reconstruct graph from JSON
        G = reconstruct_graph_from_json(data)

        if args.mode == 'structure':
            title = f"Loaded: {data.get('filename', os.path.basename(input_path))}"
            out_path = args.output or input_path.replace('.json', '.html')
            visualize_structure_plotly(G, title=title,
                                        save_path=out_path, show=not args.no_show,
                                        width=args.width, height=args.height)
        elif args.mode == 'network':
            out_path = args.output or input_path.replace('.json', '_network.html')
            visualize_hb_network_plotly(G, save_path=out_path, show=not args.no_show,
                                          width=args.width, height=args.height)
        elif args.mode == 'dashboard':
            out_path = args.output or input_path.replace('.json', '_dashboard.html')
            visualize_interactive_dashboard(G, save_path=out_path, show=not args.no_show,
                                            width=args.width, height=args.height)
        elif args.mode == 'compare':
            print("Error: --mode compare requires 3 graphs (A/B/C). Use XYZ mode or run main.py --mode compare")
            sys.exit(1)

    elif args.format == 'xyz':
        # Process XYZ and visualize
        # Import here to avoid circular imports when visualizer is imported by main.py
        try:
            from io_utils import read_raw_xyz, load_radii
            from main import build_graph_for_structure
        except ImportError as e:
            print(f"Error: Cannot import project modules: {e}")
            print("Make sure you run this from the project root directory.")
            sys.exit(1)

        atoms, coords = read_raw_xyz(input_path)
        covalent_radii = load_radii('covalent_radii.json')
        vdw_radii = load_radii('vdw_radii.json')

        G_full, _, _, _ = build_graph_for_structure(
            atoms, coords, covalent_radii, vdw_radii,
            rule=args.rule, inter_type=args.inter
        )

        basename = os.path.splitext(os.path.basename(input_path))[0]

        if args.mode == 'structure':
            title = f"{basename} - {args.inter} Rule {args.rule}"
            out_path = args.output or os.path.join('results', f"{basename}_{args.inter}_{args.rule}.html")
            os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else '.', exist_ok=True)
            visualize_structure_plotly(G_full, title=title,
                                        save_path=out_path, show=not args.no_show,
                                        width=args.width, height=args.height)
        elif args.mode == 'network':
            out_path = args.output or os.path.join('results', f"{basename}_{args.inter}_{args.rule}_network.html")
            os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else '.', exist_ok=True)
            visualize_hb_network_plotly(G_full, save_path=out_path, show=not args.no_show,
                                        width=args.width, height=args.height)
        elif args.mode == 'dashboard':
            out_path = args.output or os.path.join('results', f"{basename}_{args.inter}_{args.rule}_dashboard.html")
            os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else '.', exist_ok=True)
            visualize_interactive_dashboard(G_full, save_path=out_path, show=not args.no_show,
                                            width=args.width, height=args.height)
        elif args.mode == 'compare':
            # Build all three rules and compare
            G_A, _, _, _ = build_graph_for_structure(atoms, coords, covalent_radii, vdw_radii, 'A', args.inter)
            G_B, _, _, _ = build_graph_for_structure(atoms, coords, covalent_radii, vdw_radii, 'B', args.inter)
            G_C, _, _, _ = build_graph_for_structure(atoms, coords, covalent_radii, vdw_radii, 'C', args.inter)
            out_path = args.output or os.path.join('results', f"{basename}_{args.inter}_compare.html")
            os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else '.', exist_ok=True)
            visualize_compare_rules_plotly(G_A, G_B, G_C, title_prefix=basename,
                                            save_path=out_path, show=not args.no_show,
                                            width=args.width, height=args.height)


def get_hb_subgraph(G):
    """Returns subgraph with only H-bond edges."""
    edges = [(u, v) for u, v, d in G.edges(data=True)
             if d.get('type', '').startswith('HB_')]
    return G.edge_subgraph(edges).copy() if edges else nx.Graph()


if __name__ == "__main__":
    main()