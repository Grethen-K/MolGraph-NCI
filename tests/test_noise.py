# ============================================================
# 17. tests/test_noise.py
# ============================================================
import pytest
import numpy as np
import sys
import os


from noise_stability import (
    add_coordinate_noise, compute_edge_jaccard, compute_edge_f1,
    evaluate_noise_stability
)
from graph_builder import build_full_graph


# Тестовые данные
ATOMS = ['O', 'H', 'H']
COORDS = [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]


def test_add_noise_shape():
    """Проверяем, что шум сохраняет форму координат."""
    coords = np.array(COORDS, dtype=float)
    noisy = add_coordinate_noise(coords, sigma=0.01, seed=42)
    
    assert noisy.shape == coords.shape
    assert not np.allclose(noisy, coords)  # координаты изменились


def test_add_noise_reproducibility():
    """Проверяем воспроизводимость с одинаковым seed."""
    coords = np.array(COORDS, dtype=float)
    noisy1 = add_coordinate_noise(coords, sigma=0.01, seed=42)
    noisy2 = add_coordinate_noise(coords, sigma=0.01, seed=42)
    
    np.testing.assert_array_equal(noisy1, noisy2)


def test_add_noise_different_seeds():
    """Проверяем, что разные seeds дают разный шум."""
    coords = np.array(COORDS, dtype=float)
    noisy1 = add_coordinate_noise(coords, sigma=0.01, seed=42)
    noisy2 = add_coordinate_noise(coords, sigma=0.01, seed=43)
    
    assert not np.allclose(noisy1, noisy2)


def test_edge_jaccard_identical():
    """Jaccard для одинаковых множеств = 1.0."""
    edges = {tuple(sorted((0, 1))), tuple(sorted((1, 2)))}
    
    assert compute_edge_jaccard(edges, edges) == 1.0


def test_edge_jaccard_disjoint():
    """Jaccard для непересекающихся множеств = 0.0."""
    edges1 = {tuple(sorted((0, 1)))}
    edges2 = {tuple(sorted((1, 2)))}
    
    assert compute_edge_jaccard(edges1, edges2) == 0.0


def test_edge_jaccard_partial():
    """Jaccard для частично пересекающихся множеств."""
    edges1 = {tuple(sorted((0, 1))), tuple(sorted((1, 2)))}
    edges2 = {tuple(sorted((1, 2))), tuple(sorted((2, 0)))}
    
    # intersection = {(1,2)}, union = {(0,1), (1,2), (0,2)}
    assert compute_edge_jaccard(edges1, edges2) == 1.0 / 3.0


def test_edge_f1_perfect():
    """F1 для идеального совпадения = 1.0."""
    edges = {tuple(sorted((0, 1))), tuple(sorted((1, 2)))}
    
    assert compute_edge_f1(edges, edges) == 1.0


def test_edge_f1_empty():
    """F1 при отсутствии совпадений = 0.0."""
    edges1 = {tuple(sorted((0, 1)))}
    edges2 = {tuple(sorted((2, 3)))}
    
    assert compute_edge_f1(edges1, edges2) == 0.0


def test_evaluate_noise_stability():
    """Проверяем полный пайплайн оценки устойчивости."""
    def build_graph(atoms, coords):
        covalent = [(0, 1), (0, 2)]
        nci = [{'i': 1, 'j': 2, 'type': 'HB_B', 'distance': 1.5, 'angle': 140.0}]
        G, _ = build_full_graph(atoms, coords, covalent, nci)
        return G, None
    
    result = evaluate_noise_stability(
        build_graph, ATOMS, COORDS,
        sigma_values=(0.01, 0.02), n_repeats=2, seed_base=42
    )
    
    assert 'original_edges' in result
    assert 'stability_by_sigma' in result
    assert 'overall_stability' in result
    
    # Проверяем, что для каждого sigma есть результаты
    for sigma in (0.01, 0.02):
        assert sigma in result['stability_by_sigma']
        stats = result['stability_by_sigma'][sigma]
        assert 'jaccard_mean' in stats
        assert 'jaccard_std' in stats
        assert 0.0 <= stats['jaccard_mean'] <= 1.0
