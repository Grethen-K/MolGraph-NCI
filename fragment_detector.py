# ============================================================
# 3. fragment_detector.py Определение фрагментов/молекул в кластере через BFS/DFS
#по ковалентным рёбрам. Присвоение fragment_id каждому атому.
# ============================================================
import networkx as nx


def detect_fragments(G_covalent):
    """
    Определяет фрагменты (молекулы) в кластере через connected components
    ковалентного графа.

    Parameters
    ----------
    G_covalent : networkx.Graph
        Граф ковалентных связей.

    Returns
    -------
    dict
        {node_idx: fragment_id}
    int
        Общее число фрагментов.
    """
    fragments = {}
    for frag_id, component in enumerate(nx.connected_components(G_covalent)):
        for node in component:
            fragments[node] = frag_id
    num_fragments = len(set(fragments.values())) if fragments else 0
    return fragments, num_fragments


def is_intermolecular(G_covalent, i, j):
    """
    Проверяет, являются ли атомы i и j из разных фрагментов
    (межмолекулярный контакт).

    Parameters
    ----------
    G_covalent : networkx.Graph
    i, j : int
        Индексы атомов.

    Returns
    -------
    bool
        True если межмолекулярный, False если внутримолекулярный.
    """
    fragments, _ = detect_fragments(G_covalent)
    return fragments.get(i, -1) != fragments.get(j, -1)


def get_fragment_atoms(G_covalent, fragment_id):
    """
    Возвращает список атомов, принадлежащих заданному фрагменту.
    """
    fragments, _ = detect_fragments(G_covalent)
    return [node for node, fid in fragments.items() if fid == fragment_id]


def assign_fragment_ids_to_graph(G_covalent):
    """
    Добавляет атрибут 'fragment_id' к каждой вершине графа.

    Returns
    -------
    networkx.Graph
        Обновлённый граф.
    """
    fragments, num_frags = detect_fragments(G_covalent)
    for node, data in G_covalent.nodes(data=True):
        data['fragment_id'] = fragments.get(node, -1)
    return G_covalent, num_frags
