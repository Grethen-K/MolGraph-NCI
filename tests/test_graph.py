# ============================================================
# 15. tests/test_graph.py
#Unit tests for graph building and formats.
# ============================================================
import pytest
import numpy as np
import networkx as nx
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph_builder import build_full_graph, to_pyg_data, get_hb_subgraph


# Тестовые данные: простая структура
ATOMS = ['O', 'H', 'H']
COORDS = [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]
COVALENT_EDGES = [(0, 1), (0, 2)]
NCI_EDGES = [
    {'i': 1, 'j': 0, 'type': 'HB_B', 'distance': 1.8, 'angle': 150.0}
]


def test_build_full_graph():
    """Проверяем построение полного графа."""
    G, data = build_full_graph(ATOMS, COORDS, COVALENT_EDGES, NCI_EDGES)
    
    assert G.number_of_nodes() == 3
    assert G.number_of_edges() == 3  # 2 ковалентных + 1 HB
    
    # Проверяем атрибуты вершин
    for node in G.nodes():
        assert 'element' in G.nodes[node]
        assert 'atomic_num' in G.nodes[node]
        assert 'coords' in G.nodes[node]
        assert 'fragment_id' in G.nodes[node]
    
    # Проверяем типы рёбер
    edge_types = [d['type'] for u, v, d in G.edges(data=True)]
    assert 'covalent' in edge_types
    assert 'HB_B' in edge_types


def test_edge_index_format():
    """Проверяем формат edge_index."""
    G, data = build_full_graph(ATOMS, COORDS, COVALENT_EDGES, NCI_EDGES)
    
    edge_index = data['edge_index']
    assert edge_index.shape[0] == 2  # [2, num_edges]
    assert edge_index.shape[1] == G.number_of_edges()
    
    # Проверяем, что все индексы в допустимом диапазоне
    assert np.all(edge_index >= 0)
    assert np.all(edge_index < len(ATOMS))


def test_adjacency_matrix():
    """Проверяем матрицу смежности."""
    G, data = build_full_graph(ATOMS, COORDS, COVALENT_EDGES, NCI_EDGES)
    
    adj = data['adjacency_matrix']
    assert adj.shape == (3, 3)
    assert np.allclose(adj, adj.T)  # симметричность
    assert np.allclose(np.diag(adj), 0)  # диагональ = 0
    
    # Проверяем количество связей
    assert np.sum(adj) / 2 == G.number_of_edges()


def test_node_features():
    """Проверяем признаки вершин."""
    G, data = build_full_graph(ATOMS, COORDS, COVALENT_EDGES, NCI_EDGES)
    
    features = data['node_features']
    assert features.shape == (3, 5)  # [atomic_num, fragment_id, x, y, z]
    
    # Проверяем атомные номера
    assert features[0, 0] == 8   # O
    assert features[1, 0] == 1   # H
    assert features[2, 0] == 1   # H


def test_hb_subgraph():
    """Проверяем извлечение подграфа HB."""
    G, data = build_full_graph(ATOMS, COORDS, COVALENT_EDGES, NCI_EDGES)
    
    G_hb = get_hb_subgraph(G)
    
    # В подграфе только HB-рёбра
    for u, v, d in G_hb.edges(data=True):
        assert d.get('type', '').startswith('HB_')


def test_pyg_data():
    """Проверяем конвертацию в PyG формат."""
    G, data = build_full_graph(ATOMS, COORDS, COVALENT_EDGES, NCI_EDGES)
    
    pyg_data = to_pyg_data(G, len(ATOMS))
    
    if pyg_data is not None:
        assert 'x' in pyg_data
        assert 'edge_index' in pyg_data
        assert 'edge_attr' in pyg_data
        assert pyg_data['x'].shape[0] == len(ATOMS)
        assert pyg_data['edge_index'].shape[1] == G.number_of_edges()