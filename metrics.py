# ============================================================
# metrics.py  Метрики корректности и статистики по графу.
# ============================================================
import numpy as np
import networkx as nx
from collections import Counter

def compute_hb_statistics(G_full, hb_type='all'):
    """
    Статистика по водородным связям для одной структуры.

    Parameters
    ----------
    G_full : networkx.Graph
        Полный граф.
    hb_type : str
        'all' — все HB, 'HB_A', 'HB_B', 'HB_C' — конкретное правило.

    Returns
    -------
    dict
        Словарь метрик.
    """
    # Фильтруем HB-рёбра
    if hb_type == 'all':
        hb_edges = [(u, v, d) for u, v, d in G_full.edges(data=True)
                    if d.get('type', '').startswith('HB_')]
    else:
        hb_edges = [(u, v, d) for u, v, d in G_full.edges(data=True)
                    if d.get('type') == hb_type]

    num_hb = len(hb_edges)

    # Распределение HB на донор (атом H)
    donor_counts = Counter()
    acceptor_counts = Counter()
    for u, v, d in hb_edges:
        # u — H, v — акцептор (по соглашению из hb_detectors)
        donor_counts[u] += 1
        acceptor_counts[v] += 1

    donor_distribution = list(donor_counts.values()) if donor_counts else [0]
    acceptor_distribution = list(acceptor_counts.values()) if acceptor_counts else [0]

    # Максимальная компонента связности (только по HB-рёбрам)
    if hb_edges:
        G_hb = nx.Graph()
        G_hb.add_nodes_from(G_full.nodes())
        for u, v, d in hb_edges:
            G_hb.add_edge(u, v)
        components = list(nx.connected_components(G_hb))
        max_component_size = max(len(c) for c in components) if components else 0
        num_components = len(components)
    else:
        max_component_size = 0
        num_components = 0

    # Дистанции и углы
    distances = [d.get('distance', 0.0) for u, v, d in hb_edges]
    angles = [d.get('angle', 0.0) for u, v, d in hb_edges]

    # "Подозрительные" контакты (для варианта A)
    suspicious = []
    for u, v, d in hb_edges:
        if d.get('type') == 'HB_A':
            dist = d.get('distance', 0.0)
            angle = d.get('angle', 0.0)
            if dist > 2.8 or angle < 100:
                suspicious.append({
                    'donor': u,
                    'acceptor': v,
                    'distance': dist,
                    'angle': angle,
                    'reason': 'long_distance' if dist > 2.8 else 'low_angle'
                })

    return {
        'num_hb': num_hb,
        'num_components': num_components,
        'max_component_size': max_component_size,
        'donor_counts': donor_distribution,
        'acceptor_counts': acceptor_distribution,
        'avg_hb_per_donor': np.mean(donor_distribution) if donor_distribution else 0.0,
        'avg_hb_per_acceptor': np.mean(acceptor_distribution) if acceptor_distribution else 0.0,
        'max_hb_per_donor': max(donor_distribution) if donor_distribution else 0,
        'max_hb_per_acceptor': max(acceptor_distribution) if acceptor_distribution else 0,
        'min_hb_per_donor': min(donor_distribution) if donor_distribution else 0,
        'min_hb_per_acceptor': min(acceptor_distribution) if acceptor_distribution else 0,
        'avg_distance': np.mean(distances) if distances else 0.0,
        'min_distance': min(distances) if distances else 0.0,
        'max_distance': max(distances) if distances else 0.0,
        'avg_angle': np.mean(angles) if angles else 0.0,
        'min_angle': min(angles) if angles else 0.0,
        'max_angle': max(angles) if angles else 0.0,
        'suspicious_contacts': suspicious,
        'num_suspicious': len(suspicious)
    }

def compute_sigma_statistics(G_full, sigma_type='all'):
    """
    Статистика по сигма-дырочным взаимодействиям.
    """
    if sigma_type == 'all':
        sigma_edges = [(u, v, d) for u, v, d in G_full.edges(data=True)
                       if d.get('type', '').startswith('sigma_')]
    else:
        sigma_edges = [(u, v, d) for u, v, d in G_full.edges(data=True)
                       if d.get('type') == sigma_type]

    num_sigma = len(sigma_edges)

    donor_counts = Counter()
    acceptor_counts = Counter()
    for u, v, d in sigma_edges:
        donor_counts[u] += 1
        acceptor_counts[v] += 1

    distances = [d.get('distance', 0.0) for u, v, d in sigma_edges]
    angles_1 = [d.get('angle_1', 0.0) for u, v, d in sigma_edges]
    angles_2 = [d.get('angle_2', 0.0) for u, v, d in sigma_edges]

    # Переименованы ключи с hb на sigma
    return {
        'num_sigma': num_sigma,
        'avg_distance': np.mean(distances) if distances else 0.0,
        'min_distance': min(distances) if distances else 0.0,
        'max_distance': max(distances) if distances else 0.0,
        'avg_angle_1': np.mean(angles_1) if angles_1 else 0.0,
        'min_angle_1': min(angles_1) if angles_1 else 0.0,
        'max_angle_1': max(angles_1) if angles_1 else 0.0,
        'avg_angle_2': np.mean(angles_2) if angles_2 else 0.0,
        'avg_sigma_per_donor': np.mean(list(donor_counts.values())) if donor_counts else 0.0,
        'avg_sigma_per_acceptor': np.mean(list(acceptor_counts.values())) if acceptor_counts else 0.0,
    }

def compute_dataset_statistics(results_list):
    """
    Агрегирует статистику по набору структур.

    Parameters
    ----------
    results_list : list of dict
        Каждый элемент — результат compute_hb_statistics() для одной структуры.

    Returns
    -------
    dict
        Сводная статистика по всему набору.
    """
    if not results_list:
        return {}

    num_hbs = [r['num_hb'] for r in results_list]
    max_components = [r['max_component_size'] for r in results_list]

    return {
        'num_structures': len(results_list),
        'avg_hb_per_structure': np.mean(num_hbs),
        'min_hb_per_structure': min(num_hbs),
        'max_hb_per_structure': max(num_hbs),
        'std_hb_per_structure': np.std(num_hbs),
        'avg_max_component_size': np.mean(max_components),
        'min_max_component_size': min(max_components),
        'max_max_component_size': max(max_components),
        'total_suspicious': sum(r['num_suspicious'] for r in results_list),
    }

def compare_rules_statistics(results_A, results_B, results_C):
    """
    Сравнивает метрики для трёх правил A/B/C на одном наборе структур.

    Returns
    -------
    dict
        Сравнительная таблица.
    """
    stats_A = compute_dataset_statistics(results_A)
    stats_B = compute_dataset_statistics(results_B)
    stats_C = compute_dataset_statistics(results_C)

    return {
        'A': stats_A,
        'B': stats_B,
        'C': stats_C,
        'comparison': {
            'avg_hb_ratio_A_to_B': stats_A.get('avg_hb_per_structure', 0) / max(stats_B.get('avg_hb_per_structure', 1), 1e-9),
            'avg_hb_ratio_C_to_B': stats_C.get('avg_hb_per_structure', 0) / max(stats_B.get('avg_hb_per_structure', 1), 1e-9),
            'lost_hb_C_vs_B': stats_B.get('avg_hb_per_structure', 0) - stats_C.get('avg_hb_per_structure', 0),
            'extra_hb_A_vs_B': stats_A.get('avg_hb_per_structure', 0) - stats_B.get('avg_hb_per_structure', 0),
        }
    }