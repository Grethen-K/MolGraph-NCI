# ============================================================
# 1. distance_matrix.py Вычисление матрицы попарных расстояний между атомами.
#Используется ковалентным детектором и детекторами NCI.
# ============================================================
import numpy as np
from scipy.spatial.distance import cdist


def compute_distance_matrix(coords):
    """
    Вычисляет матрицу попарных евклидовых расстояний.

    Parameters
    ----------
    coords : list or np.ndarray, shape (N, 3)
        Координаты атомов.

    Returns
    -------
    np.ndarray, shape (N, N)
        Матрица попарных расстояний (симметричная, диагональ = 0).
    """
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape (N, 3)")
    return cdist(coords, coords, metric='euclidean')


def compute_distance_vector(coords, i, j):
    """
    Вектор расстояния между атомами i и j.

    Returns
    -------
    np.ndarray, shape (3,)
        Вектор r_j - r_i.
    """
    coords = np.asarray(coords, dtype=float)
    return coords[j] - coords[i]


def get_bond_distance_matrix(coords, bond_pairs):
    """
    Возвращает расстояния только для заданных пар (связей).

    Parameters
    ----------
    bond_pairs : list of tuple(int, int)
        Список пар индексов атомов.

    Returns
    -------
    list of float
        Расстояния для каждой пары.
    """
    coords = np.asarray(coords, dtype=float)
    return [float(np.linalg.norm(coords[i] - coords[j])) for i, j in bond_pairs]