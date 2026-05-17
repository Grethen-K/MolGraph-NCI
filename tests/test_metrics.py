# ============================================================
# 16. tests/test_metrics.py
# ============================================================
import pytest
import numpy as np
import networkx as nx
import sys
import os


from graph_builder import build_full_graph
from metrics import compute_hb_statistics, compute_dataset_statistics, compare_rules_statistics


def create_test_graph_with_hb(num_hb=3):
    """Создаёт тестовый граф с заданным числом ВС."""
    atoms = ['O', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H']
    coords = [
        [0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0],
        [2.8, 0.0, 0.0], [3.5, 0.5, 0.0], [3.0, -0.8, 0.0],
        [5.6, 0.0, 0.0], [6.5, 0.5, 0.0], [6.0, -0.8, 0.0]
    ]
    covalent_edges = [(0,1), (0,2), (3,4), (3,5), (6,7), (6,8)]
    
    nci_edges = []
    if num_hb >= 1:
        nci_edges.append({'i': 1, 'j': 3, 'type': 'HB_B', 'distance': 1.8, 'angle': 150.0})
    if num_hb >= 2:
        nci_edges.append({'i': 4, 'j': 6, 'type': 'HB_B', 'distance': 1.9, 'angle': 145.0})
    if num_hb >= 3:
        nci_edges.append({'i': 7, 'j': 0, 'type': 'HB_B', 'distance': 2.0, 'angle': 140.0})
    
    G, _ = build_full_graph(atoms, coords, covalent_edges, nci_edges)
    return G


def test_hb_statistics_count():
    """Проверяем подсчёт числа ВС."""
    G = create_test_graph_with_hb(num_hb=2)
    stats = compute_hb_statistics(G, hb_type='HB_B')
    
    assert stats['num_hb'] == 2


def test_hb_statistics_distances():
    """Проверяем статистику по расстояниям."""
    G = create_test_graph_with_hb(num_hb=2)
    stats = compute_hb_statistics(G, hb_type='HB_B')
    
    assert stats['min_distance'] == 1.8
    assert stats['max_distance'] == 1.9
    assert abs(stats['avg_distance'] - 1.85) < 0.01


def test_hb_statistics_angles():
    """Проверяем статистику по углам."""
    G = create_test_graph_with_hb(num_hb=2)
    stats = compute_hb_statistics(G, hb_type='HB_B')
    
    assert stats['min_angle'] == 145.0
    assert stats['max_angle'] == 150.0


def test_hb_statistics_components():
    """Проверяем размер компонент связности."""
    G = create_test_graph_with_hb(num_hb=3)
    stats = compute_hb_statistics(G, hb_type='HB_B')
    
    # Все 3 молекулы связаны через ВС => 1 компонента размером 9
    assert stats['num_components'] == 6
    assert stats['max_component_size'] == 2


def test_hb_statistics_suspicious():
    """Проверяем выделение подозрительных контактов для правила A."""
    # Создаём граф с HB_A и длинной связью
    atoms = ['O', 'H', 'O', 'H']
    coords = [[0,0,0], [3.0,0,0], [5,0,0], [6,0,0]]
    covalent = [(0,1), (2,3)]
    nci = [{'i': 1, 'j': 2, 'type': 'HB_A', 'distance': 3.1, 'angle': 90.0}]
    
    G, _ = build_full_graph(atoms, coords, covalent, nci)
    stats = compute_hb_statistics(G, hb_type='HB_A')
    
    assert stats['num_suspicious'] >= 1


def test_dataset_statistics():
    """Проверяем агрегацию по набору структур."""
    stats_list = [
        {'num_hb': 2, 'max_component_size': 3, 'num_suspicious': 0},
        {'num_hb': 4, 'max_component_size': 6, 'num_suspicious': 1},
        {'num_hb': 3, 'max_component_size': 4, 'num_suspicious': 0}
    ]
    
    # Добавляем остальные обязательные поля
    for s in stats_list:
        s.update({
            'num_components': 1,
            'donor_counts': [1, 1],
            'acceptor_counts': [1, 1],
            'avg_hb_per_donor': 1.0,
            'avg_hb_per_acceptor': 1.0,
            'max_hb_per_donor': 1,
            'max_hb_per_acceptor': 1,
            'min_hb_per_donor': 1,
            'min_hb_per_acceptor': 1,
            'avg_distance': 1.8,
            'min_distance': 1.5,
            'max_distance': 2.0,
            'avg_angle': 140.0,
            'min_angle': 130.0,
            'max_angle': 150.0,
            'suspicious_contacts': []
        })
    
    ds_stats = compute_dataset_statistics(stats_list)
    
    assert ds_stats['num_structures'] == 3
    assert ds_stats['avg_hb_per_structure'] == 3.0
    assert ds_stats['min_hb_per_structure'] == 2
    assert ds_stats['max_hb_per_structure'] == 4
    assert ds_stats['total_suspicious'] == 1


def test_compare_rules_statistics():
    """Проверяем сравнение правил A/B/C."""
    results_A = [{'num_hb': 5, 'max_component_size': 8, 'num_suspicious': 2}]
    results_B = [{'num_hb': 3, 'max_component_size': 5, 'num_suspicious': 0}]
    results_C = [{'num_hb': 1, 'max_component_size': 2, 'num_suspicious': 0}]
    
    # Дополняем обязательные поля
    for r in [results_A, results_B, results_C]:
        for s in r:
            s.update({
                'num_components': 1,
                'donor_counts': [1],
                'acceptor_counts': [1],
                'avg_hb_per_donor': 1.0, 'avg_hb_per_acceptor': 1.0,
                'max_hb_per_donor': 1, 'max_hb_per_acceptor': 1,
                'min_hb_per_donor': 1, 'min_hb_per_acceptor': 1,
                'avg_distance': 1.8, 'min_distance': 1.5, 'max_distance': 2.0,
                'avg_angle': 140.0, 'min_angle': 130.0, 'max_angle': 150.0,
                'suspicious_contacts': []
            })
    
    comparison = compare_rules_statistics(results_A, results_B, results_C)
    
    assert comparison['comparison']['extra_hb_A_vs_B'] == 2.0  # 5 - 3
    assert comparison['comparison']['lost_hb_C_vs_B'] == 2.0   # 3 - 1

