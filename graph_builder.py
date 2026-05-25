# ============================================================
# graph_builder.py
# Построение полного графа G=(V,E) с типами рёбер.
# Объединяет ковалентные рёбра и NCI-рёбра (HB, sigma).
# Формирует node features и экспортирует в различные форматы.
# ============================================================
import numpy as np
import networkx as nx

def build_full_graph(atoms, coords, covalent_edges, nci_edges,
                     fragment_ids=None, atomic_nums=None):
    """
    Строит полный граф G=(V,E) с типами рёбер.

    Parameters
    ----------
    atoms : list of str
        Символы атомов.
    coords : list or np.ndarray, shape (N, 3)
        Координаты.
    covalent_edges : list of tuple(int, int)
        Список ковалентных рёбер (i, j).
    nci_edges : list of dict
        Список NCI-рёбер. Каждый элемент — dict:
        {'i': int, 'j': int, 'type': str, 'distance': float, ...}
        type может быть 'HB_A', 'HB_B', 'HB_C', 'sigma_A', 'sigma_B', 'sigma_C'.
    fragment_ids : dict or None
        {node_idx: fragment_id}. Если None — все в фрагменте 0.
    atomic_nums : dict or None
        {element: atomic_number}. Если None — используется встроенный словарь.

    Returns
    -------
    networkx.Graph
        Полный граф с атрибутами вершин и рёбер.
    dict
        Словарь с edge_index, adjacency matrix и другими форматами.
    """
    coords = np.asarray(coords, dtype=float)
    n = len(atoms)

    # FIX #5: Добавлены лантаноиды (Z=58-71)
    if atomic_nums is None:
        atomic_nums = {
            'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
            'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
            'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22,
            'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29,
            'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36,
            'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43,
            'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
            'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57,
            'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63,
            'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69,
            'Yb': 70, 'Lu': 71,
            'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78,
            'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85
        }

    if fragment_ids is None:
        fragment_ids = {i: 0 for i in range(n)}

    G = nx.Graph()

    # Вершины
    for i in range(n):
        G.add_node(
            i,
            element=atoms[i],
            atomic_num=atomic_nums.get(atoms[i], 0),
            coords=tuple(coords[i]),
            fragment_id=fragment_ids.get(i, 0)
        )

    # Ковалентные рёбра
    for i, j in covalent_edges:
        G.add_edge(i, j, type='covalent')

    # NCI-рёбра
    for edge in nci_edges:
        i = edge['i']
        j = edge['j']
        edge_type = edge['type']
        # Добавляем все атрибуты из edge
        attrs = {k: v for k, v in edge.items() if k not in ('i', 'j')}
        G.add_edge(i, j, **attrs)

    # Формируем дополнительные структуры
    result = {
        'graph': G,
        'edge_index': _to_edge_index(G),
        'adjacency_matrix': _to_adjacency_matrix(G, n),
        'edge_list': _to_edge_list(G),
        'node_features': _extract_node_features(G, n)
    }

    return G, result

def _to_edge_index(G):
    """
    Формат edge_index как в PyTorch Geometric: [2, num_edges].
    """
    edges = []
    for u, v, data in G.edges(data=True):
        edges.append([u, v])
    return np.array(edges, dtype=int).T if edges else np.zeros((2, 0), dtype=int)

def _to_adjacency_matrix(G, n):
    """
    Матрица смежности (n x n).
    """
    adj = np.zeros((n, n), dtype=int)
    for u, v in G.edges():
        adj[u, v] = 1
        adj[v, u] = 1
    return adj

def _to_edge_list(G):
    """
    Список рёбер [i, j, type].
    """
    edges = []
    for u, v, data in G.edges(data=True):
        edges.append([u, v, data.get('type', 'unknown')])
    return edges

def _extract_node_features(G, n):
    """
    Извлекает признаки вершин в виде numpy массива.
    """
    features = []
    for i in range(n):
        data = G.nodes[i]
        feat = [
            data.get('atomic_num', 0),
            data.get('fragment_id', 0),
            *data.get('coords', (0, 0, 0))
        ]
        features.append(feat)
    return np.array(features, dtype=float)

def to_pyg_data(G, n):
    """
    Конвертирует граф в формат PyTorch Geometric Data (если torch_geometric доступен).
    Возвращает dict с данными, совместимыми с PyG.
    """
    try:
        import torch
        x = torch.tensor(_extract_node_features(G, n), dtype=torch.float)
        edge_index = torch.tensor(_to_edge_index(G), dtype=torch.long)

        # Edge attributes: one-hot encoding типа рёбер
        edge_types = []
        type_map = {'covalent': 0, 'HB_A': 1, 'HB_B': 2, 'HB_C': 3,
                    'sigma_A': 4, 'sigma_B': 5, 'sigma_C': 6}
        for u, v, data in G.edges(data=True):
            et = data.get('type', 'unknown')
            edge_types.append(type_map.get(et, -1))
        edge_attr = torch.tensor(edge_types, dtype=torch.long)

        return {
            'x': x,
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'num_nodes': n
        }
    except ImportError:
        return None

def get_subgraph_by_type(G, edge_type):
    """
    Возвращает подграф, содержащий только рёбра заданного типа.
    """
    edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == edge_type]
    return G.edge_subgraph(edges).copy()

def get_hb_subgraph(G):
    """
    Возвращает подграф только с водородными связями (все типы HB_*).
    """
    edges = [(u, v) for u, v, d in G.edges(data=True)
             if d.get('type', '').startswith('HB_')]
    return G.edge_subgraph(edges).copy() if edges else nx.Graph()