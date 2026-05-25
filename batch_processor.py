# ============================================================
# batch_processor.py Обработка набора XYZ-файлов: batch processing, сравнение правил, сохранение результатов.
# ============================================================
import os
import json
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

from io_utils import read_raw_xyz
from covalent_detector import get_covalent_skeleton_combined, get_covalent_skeleton_radii
from fragment_detector import assign_fragment_ids_to_graph, detect_fragments
from hb_detectors import apply_hb_rules
from sigma_detectors import apply_sigma_rules
from graph_builder import build_full_graph
from metrics import compute_hb_statistics, compute_dataset_statistics, compare_rules_statistics
from noise_stability import evaluate_noise_stability
from distance_matrix import compute_distance_matrix


def process_single_structure(xyz_path, rules=('A', 'B', 'C'),
                             inter_types=('HB', 'sigma'),
                             use_radii_fallback=True,
                             covalent_radii=None,
                             vdw_radii=None,
                             compute_noise=False,
                             noise_sigmas=(0.01, 0.02, 0.03, 0.04, 0.05)):
    """
    Обрабатывает одну XYZ-структуру для всех правил и типов взаимодействий.

    Parameters
    ----------
    xyz_path : str
        Путь к XYZ-файлу.
    rules : tuple of str
        Правила для оценки ('A', 'B', 'C').
    inter_types : tuple of str
        Типы взаимодействий ('HB', 'sigma').
    use_radii_fallback : bool
        Использовать ли fallback на ковалентные радиусы.
    covalent_radii : dict or None
    vdw_radii : dict or None
    compute_noise : bool
        Вычислять ли устойчивость к шуму.
    noise_sigmas : tuple
        Уровни шума.

    Returns
    -------
    dict
        Полные результаты для структуры.
    """
    filename = os.path.basename(xyz_path)
    atoms, coords = read_raw_xyz(xyz_path)
    coords_arr = np.array(coords, dtype=float)

    # Ковалентный скелет
    # FIX #1: Убран неработающий xyz2mol_func, используем только радиусный метод
    # FIX #2: Добавлен импорт get_covalent_skeleton_radii
    if use_radii_fallback and covalent_radii is not None:
        G_cov, dist_mat = get_covalent_skeleton_radii(
            atoms, coords, covalent_radii,
            scale=1.15, tolerance=0.0
        )
        method = 'radii'
    else:
        # Если xyz2mol доступен, можно добавить здесь логику
        # Пока используем радиусный метод как единственный надежный
        G_cov, dist_mat = get_covalent_skeleton_radii(
            atoms, coords, covalent_radii,
            scale=1.15, tolerance=0.0
        )
        method = 'radii'

    # Фрагменты
    G_cov, num_frags = assign_fragment_ids_to_graph(G_cov)
    fragment_ids = {node: data.get('fragment_id', 0)
                    for node, data in G_cov.nodes(data=True)}

    covalent_edges = [(u, v) for u, v, d in G_cov.edges(data=True)
                      if d.get('type') == 'covalent']

    results = {
        'filename': filename,
        'num_atoms': len(atoms),
        'num_covalent_bonds': len(covalent_edges),
        'num_fragments': num_frags,
        'covalent_method': method,
        'rules': {}
    }

    # Обрабатываем каждое правило и тип взаимодействия
    for rule in rules:
        results['rules'][rule] = {}
        for inter_type in inter_types:
            if inter_type == 'HB':
                nci_edges_raw = apply_hb_rules(
                    G_cov, atoms, coords_arr, vdw_radii or {},
                    rule=rule, interaction_type='HB'
                )
                nci_edges = []
                for hb in nci_edges_raw:
                    nci_edges.append({
                        'i': hb[0], 'j': hb[1],
                        'type': f'HB_{rule}',
                        'distance': float(hb[2]),
                        'angle': float(hb[3]),
                        'donor_element': hb[4],
                        'acceptor_element': hb[5]
                    })
            else:
                nci_edges_raw = apply_sigma_rules(
                    G_cov, atoms, coords_arr, vdw_radii or {},
                    rule=rule
                )
                nci_edges = []
                for sg in nci_edges_raw:
                    nci_edges.append({
                        'i': sg[0], 'j': sg[1],
                        'type': f'sigma_{rule}',
                        'distance': float(sg[2]),
                        'angle_1': float(sg[3]),
                        'angle_2': float(sg[4]),
                        'donor_element': sg[5],
                        'acceptor_element': sg[6]
                    })

            # Строим полный граф
            G_full, graph_data = build_full_graph(
                atoms, coords_arr, covalent_edges, nci_edges,
                fragment_ids=fragment_ids
            )

            # Метрики
            if inter_type == 'HB':
                stats = compute_hb_statistics(G_full, hb_type=f'HB_{rule}')
            else:
                from metrics import compute_sigma_statistics
                stats = compute_sigma_statistics(G_full, sigma_type=f'sigma_{rule}')

            results['rules'][rule][inter_type] = {
                'num_nci': len(nci_edges),
                'graph': graph_data,
                'statistics': stats
            }

    # Устойчивость к шуму (только для правила B HB)
    if compute_noise and 'B' in results['rules'] and 'HB' in results['rules']['B']:
        # FIX #11: Передаем G_cov явно, а не через замыкание
        def build_graph_for_noise(at, co, G_cov_ref=G_cov, covalent_edges_ref=covalent_edges,
                                   fragment_ids_ref=fragment_ids, vdw_radii_ref=vdw_radii):
            nci = apply_hb_rules(
                G_cov_ref, at, np.array(co), vdw_radii_ref or {},
                rule='B', interaction_type='HB'
            )
            nci_edges = [{'i': hb[0], 'j': hb[1], 'type': 'HB_B',
                          'distance': hb[2], 'angle': hb[3]} for hb in nci]
            G, _ = build_full_graph(at, np.array(co), covalent_edges_ref, nci_edges,
                                    fragment_ids=fragment_ids_ref)
            return G, None

        noise_result = evaluate_noise_stability(
            build_graph_for_noise, atoms, coords_arr,
            sigma_values=noise_sigmas, n_repeats=3
        )
        results['noise_stability'] = noise_result

    return results


def process_dataset(xyz_dir, output_dir='results',
                    rules=('A', 'B', 'C'),
                    inter_types=('HB', 'sigma'),
                    covalent_radii=None,
                    vdw_radii=None,
                    compute_noise=False,
                    n_workers=1):
    """
    Обрабатывает все XYZ-файлы в директории.

    Parameters
    ----------
    xyz_dir : str
        Директория с XYZ-файлами.
    output_dir : str
        Директория для результатов.
    n_workers : int
        Число параллельных процессов (1 = последовательно).

    Returns
    -------
    dict
        Сводные результаты по всему набору.
    """
    os.makedirs(output_dir, exist_ok=True)

    xyz_files = [os.path.join(xyz_dir, f) for f in os.listdir(xyz_dir)
                 if f.endswith('.xyz')]
    xyz_files.sort()

    print(f"Found {len(xyz_files)} XYZ files in {xyz_dir}")

    all_results = []

    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(
                    process_single_structure, f, rules, inter_types,
                    True, covalent_radii, vdw_radii, compute_noise
                ): f for f in xyz_files
            }
            for future in as_completed(futures):
                try:
                    result = future.result()
                    all_results.append(result)
                    print(f"  Processed: {result['filename']}")
                except Exception as e:
                    print(f"  Error processing {futures[future]}: {e}")
    else:
        for f in xyz_files:
            try:
                result = process_single_structure(
                    f, rules, inter_types,
                    True, covalent_radii, vdw_radii, compute_noise
                )
                all_results.append(result)
                print(f"  Processed: {result['filename']}")
            except Exception as e:
                print(f"  Error processing {f}: {e}")

    # Сохраняем результаты
    _save_results(all_results, output_dir, rules, inter_types)

    return {
        'num_structures': len(all_results),
        'results': all_results
    }


def _save_results(all_results, output_dir, rules, inter_types):
    """
    Сохраняет результаты в JSON и CSV.
    """
    # JSON с полными данными
    json_path = os.path.join(output_dir, 'results_full.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False,
                  default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o))
    print(f"Saved full results to {json_path}")

    # CSV со статистикой
    for inter_type in inter_types:
        csv_path = os.path.join(output_dir, f'summary_{inter_type}.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Заголовок
            header = ['filename', 'num_atoms', 'num_fragments', 'rule',
                      'num_nci', 'avg_distance', 'min_distance', 'max_distance']
            if inter_type == 'HB':
                header += ['avg_angle', 'min_angle', 'max_angle',
                           'max_component_size', 'num_suspicious']
            writer.writerow(header)

            for res in all_results:
                for rule in rules:
                    if inter_type not in res['rules'][rule]:
                        continue
                    stats = res['rules'][rule][inter_type]['statistics']
                    row = [
                        res['filename'],
                        res['num_atoms'],
                        res['num_fragments'],
                        rule,
                        res['rules'][rule][inter_type]['num_nci'],
                        f"{stats.get('avg_distance', 0):.3f}",
                        f"{stats.get('min_distance', 0):.3f}",
                        f"{stats.get('max_distance', 0):.3f}",
                    ]
                    if inter_type == 'HB':
                        row += [
                            f"{stats.get('avg_angle', 0):.1f}",
                            f"{stats.get('min_angle', 0):.1f}",
                            f"{stats.get('max_angle', 0):.1f}",
                            stats.get('max_component_size', 0),
                            stats.get('num_suspicious', 0)
                        ]
                    writer.writerow(row)
            print(f"Saved summary CSV to {csv_path}")

    # Сравнительная статистика A/B/C
    for inter_type in inter_types:
        results_by_rule = {rule: [] for rule in rules}
        for res in all_results:
            for rule in rules:
                if inter_type in res['rules'][rule]:
                    results_by_rule[rule].append(
                        res['rules'][rule][inter_type]['statistics']
                    )

        if all(results_by_rule[r] for r in rules):
            comparison = compare_rules_statistics(
                results_by_rule['A'],
                results_by_rule['B'],
                results_by_rule['C']
            )
            comp_path = os.path.join(output_dir, f'comparison_{inter_type}.json')
            with open(comp_path, 'w', encoding='utf-8') as f:
                json.dump(comparison, f, indent=2, ensure_ascii=False,
                          default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o))
            print(f"Saved comparison to {comp_path}")