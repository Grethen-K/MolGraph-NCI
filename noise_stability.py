# ============================================================
# 6. noise_stability.py Устойчивость графа к малым изменениям геометрии (раздел 6.2 ТЗ).
# ============================================================
import numpy as np
import copy


def add_coordinate_noise(coords, sigma=0.01, seed=None):
    """
    Добавляет случайный гауссов шум к координатам атомов.

    Parameters
    ----------
    coords : list or np.ndarray, shape (N, 3)
        Исходные координаты.
    sigma : float
        Стандартное отклонение шума в Ангстремах.
    seed : int or None
        Seed для воспроизводимости.

    Returns
    -------
    np.ndarray
        Зашумлённые координаты.
    """
    coords = np.asarray(coords, dtype=float)
    if seed is not None:
        np.random.seed(seed)
    noise = np.random.normal(0, sigma, coords.shape)
    return coords + noise


def compute_edge_jaccard(edges_original, edges_noisy):
    """
    Вычисляет Jaccard similarity между множествами рёбер.

    Parameters
    ----------
    edges_original : set of tuple(int, int)
        Исходные рёбра (отсортированные пары).
    edges_noisy : set of tuple(int, int)
        Рёбра после добавления шума.

    Returns
    -------
    float
        Jaccard index: |intersection| / |union|.
    """
    intersection = edges_original & edges_noisy
    union = edges_original | edges_noisy
    if not union:
        return 1.0
    return len(intersection) / len(union)


def compute_edge_f1(edges_original, edges_noisy):
    """
    F1-score для сравнения множеств рёбер.
    """
    tp = len(edges_original & edges_noisy)
    fp = len(edges_noisy - edges_original)
    fn = len(edges_original - edges_noisy)
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate_noise_stability(build_graph_func, atoms, coords,
                              sigma_values=(0.01, 0.02, 0.03, 0.04, 0.05),
                              n_repeats=5, seed_base=42):
    """
    Оценивает устойчивость графа к шуму для разных уровней sigma.

    Parameters
    ----------
    build_graph_func : callable
        Функция, которая принимает (atoms, coords) и возвращает (G, result_dict).
    atoms : list of str
    coords : list or np.ndarray
    sigma_values : tuple of float
        Уровни шума в Ангстремах.
    n_repeats : int
        Число повторов для каждого уровня шума.
    seed_base : int
        Базовый seed.

    Returns
    -------
    dict
        Результаты устойчивости.
    """
    # Исходный граф
    G_orig, _ = build_graph_func(atoms, coords)
    edges_orig = _get_edge_set(G_orig)

    results = {}
    for sigma in sigma_values:
        jaccards = []
        f1s = []
        for r in range(n_repeats):
            seed = seed_base + r
            coords_noisy = add_coordinate_noise(coords, sigma, seed)
            G_noisy, _ = build_graph_func(atoms, coords_noisy)
            edges_noisy = _get_edge_set(G_noisy)
            jaccards.append(compute_edge_jaccard(edges_orig, edges_noisy))
            f1s.append(compute_edge_f1(edges_orig, edges_noisy))

        results[sigma] = {
            'jaccard_mean': np.mean(jaccards),
            'jaccard_std': np.std(jaccards),
            'jaccard_min': np.min(jaccards),
            'jaccard_max': np.max(jaccards),
            'f1_mean': np.mean(f1s),
            'f1_std': np.std(f1s),
        }

    return {
        'original_edges': len(edges_orig),
        'stability_by_sigma': results,
        'overall_stability': np.mean([v['jaccard_mean'] for v in results.values()])
    }


def evaluate_noise_stability_dataset(build_graph_func, dataset,
                                      sigma_values=(0.01, 0.02, 0.03, 0.04, 0.05),
                                      n_repeats=5):
    """
    Оценивает устойчивость для набора структур.

    Parameters
    ----------
    dataset : list of dict
        [{'atoms': [...], 'coords': [...], 'name': '...'}, ...]

    Returns
    -------
    dict
        Агрегированные результаты по всему набору.
    """
    all_results = []
    for item in dataset:
        res = evaluate_noise_stability(
            build_graph_func,
            item['atoms'],
            item['coords'],
            sigma_values=sigma_values,
            n_repeats=n_repeats
        )
        all_results.append({
            'name': item.get('name', 'unknown'),
            'result': res
        })

    # Агрегация по всем структурам
    aggregated = {}
    for sigma in sigma_values:
        jaccards = [r['result']['stability_by_sigma'][sigma]['jaccard_mean']
                    for r in all_results]
        aggregated[sigma] = {
            'avg_jaccard': np.mean(jaccards),
            'std_jaccard': np.std(jaccards),
            'min_jaccard': np.min(jaccards),
            'max_jaccard': np.max(jaccards),
        }

    return {
        'per_structure': all_results,
        'aggregated': aggregated
    }


def _get_edge_set(G):
    """
    Возвращает множество рёбер графа как frozenset для сравнения.
    """
    edges = set()
    for u, v in G.edges():
        edges.add(tuple(sorted((u, v))))
    return edges