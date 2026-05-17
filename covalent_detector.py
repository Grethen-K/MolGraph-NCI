# ============================================================
# 2. covalent_detector.py Альтернативный способ построения ковалентного скелета
#через ковалентные радиусы с коэффициентом масштабирования.
#Fallback, если xyz2mol не справился.
# ============================================================
import numpy as np
import networkx as nx
from distance_matrix import compute_distance_matrix


def get_covalent_skeleton_radii(atoms, coords, covalent_radii, scale=1.15, tolerance=0.0):
    """
    Строит ковалентный скелет через сумму ковалентных радиусов.

    Parameters
    ----------
    atoms : list of str
        Символы атомов.
    coords : list or np.ndarray, shape (N, 3)
        Координаты.
    covalent_radii : dict
        Словарь {element: radius_in_A}.
    scale : float
        Коэффициент масштабирования (по умолчанию 1.15).
    tolerance : float
        Дополнительный толеранс в Ангстремах.

    Returns
    -------
    networkx.Graph
        Граф ковалентных связей.
    np.ndarray
        Матрица попарных расстояний.
    """
    coords = np.asarray(coords, dtype=float)
    n = len(atoms)
    dist_mat = compute_distance_matrix(coords)

    G = nx.Graph()
    for i in range(n):
        G.add_node(i, element=atoms[i], coords=coords[i])

    for i in range(n):
        for j in range(i + 1, n):
            r_i = covalent_radii.get(atoms[i], 0.7)
            r_j = covalent_radii.get(atoms[j], 0.7)
            threshold = (r_i + r_j) * scale + tolerance
            if dist_mat[i, j] <= threshold:
                G.add_edge(i, j, type='covalent')

    return G, dist_mat


def get_covalent_skeleton_combined(atoms, coords, covalent_radii,
                                    xyz2mol_func=None, charge=0,
                                    scale=1.15, tolerance=0.0):
    """
    Комбинированный подход: пробуем xyz2mol, если не удалось — fallback на радиусы.

    Parameters
    ----------
    xyz2mol_func : callable or None
        Функция get_covalent_skeleton из covalent_engine.py.
        Если None или не удалась — используем радиусы.

    Returns
    -------
    networkx.Graph
    np.ndarray
        Матрица расстояний.
    str
        Использованный метод: 'xyz2mol' или 'radii'.
    """
    if xyz2mol_func is not None:
        try:
            G, coords_out = xyz2mol_func(coords, atoms, charge)
            if G is not None and G.number_of_nodes() > 0:
                dist_mat = compute_distance_matrix(coords_out)
                return G, dist_mat, 'xyz2mol'
        except Exception:
            pass

    G, dist_mat = get_covalent_skeleton_radii(atoms, coords, covalent_radii, scale, tolerance)
    return G, dist_mat, 'radii'