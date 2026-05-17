# ============================================================
# 13. tests/test_covalent.py Unit tests for covalent bond detection.
# ============================================================
import pytest
import numpy as np
import networkx as nx
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from covalent_detector import get_covalent_skeleton_radii


# Тестовые данные: молекула воды
WATER_ATOMS = ['O', 'H', 'H']
WATER_COORDS = [
    [0.0, 0.0, 0.0],
    [0.96, 0.0, 0.0],
    [-0.24, 0.93, 0.0]
]

COVALENT_RADII = {
    'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57,
    'P': 1.07, 'S': 1.05, 'Cl': 1.02, 'Br': 1.20, 'I': 1.39
}


def test_water_covalent_bonds():
    """Проверяем, что в воде 2 ковалентные связи O-H."""
    G, dist_mat = get_covalent_skeleton_radii(
        WATER_ATOMS, WATER_COORDS, COVALENT_RADII, scale=1.15
    )
    assert G.number_of_nodes() == 3
    assert G.number_of_edges() == 2
    # Проверяем, что связи O-H
    for u, v in G.edges():
        elems = {WATER_ATOMS[u], WATER_ATOMS[v]}
        assert elems == {'O', 'H'}


def test_distance_matrix_shape():
    """Проверяем форму матрицы расстояний."""
    G, dist_mat = get_covalent_skeleton_radii(
        WATER_ATOMS, WATER_COORDS, COVALENT_RADII
    )
    assert dist_mat.shape == (3, 3)
    assert np.allclose(dist_mat, dist_mat.T)  # симметричность
    assert np.allclose(np.diag(dist_mat), 0)   # диагональ = 0


def test_empty_structure():
    """Проверяем обработку пустой структуры."""
    G, dist_mat = get_covalent_skeleton_radii([], [], COVALENT_RADII)
    assert G.number_of_nodes() == 0
    assert dist_mat.shape == (0, 0)