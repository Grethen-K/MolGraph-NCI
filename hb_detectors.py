# ============================================================
# # hb_detectors.py  Детекторы водородных связей с проверкой типа донора, параметризацией и возвратом типа ребра.
# ============================================================
import numpy as np
import networkx as nx
from collections import defaultdict

def calculate_angle(p1, p2, p3):
    """Считаем угол D-H...A."""
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm == 0:
        return 0.0
    res = np.degrees(np.arccos(np.clip(np.dot(v1, v2) / norm, -1.0, 1.0)))
    return res

def apply_hb_rules(G, atoms, coords, vdw_radii, rule='B', interaction_type='HB',
                   donor_elements=None, acceptor_elements=None,
                   max_covalent_dist=4, return_edge_type=True,
                   check_donor_type=True, include_weak_donors=False):
    """
    Ищем водородные связи с улучшенной логикой.

    Parameters
    ----------
    G : networkx.Graph
        Ковалентный граф.
    atoms : list of str
        Символы атомов.
    coords : np.ndarray
        Координаты.
    vdw_radii : dict
        ВдВ радиусы.
    rule : str
        'A', 'B' или 'C'.
    interaction_type : str
        'HB' или другой.
    donor_elements : list or None
        Допустимые элементы доноров X. Если None — используется правило.
    acceptor_elements : list or None
        Допустимые элементы акцепторов. Если None — используется правило.
    max_covalent_dist : int
        Максимальное ковалентное расстояние для исключения.
    return_edge_type : bool
        Возвращать ли тип ребра ('HB_A', 'HB_B', 'HB_C').
    check_donor_type : bool
        Проверять ли тип донора X (O, N и т.д.).
    include_weak_donors : bool
        Включать ли слабые доноры (C, B, Si, P, As, Se) в дополнение к O, N, S.

    Returns
    -------
    list of tuple
        Каждый элемент: (H_idx, A_idx, dist, angle, H_sym, A_sym, edge_type, is_intermolecular)
    """
    coords = np.asarray(coords, dtype=float)

    #Разделены стандартные и слабые доноры
    standard_donors = ['O', 'N', 'S']
    weak_donors = ['C', 'B', 'Si', 'P', 'As', 'Se']
    default_acceptors = ['O', 'N', 'F', 'S', 'Cl', 'Br', 'I']

    # Настройки по правилу
    if rule == 'A':
        d_max = 3.25
        a_min = 0
    elif rule == 'B':
        d_max = 'vdw_sum'
        a_min = 120  # по IUPAC можно опустить до 110
    elif rule == 'C':
        d_max = 'vdw_sum_minus_0.2'  # проверить от 0,1 до 0,25
        a_min = 150
    else:
        return []

    # Формируем список доноров
    if donor_elements is not None:
        donors = donor_elements
    else:
        donors = standard_donors[:]
        if include_weak_donors:
            donors.extend(weak_donors)

    acceptors = acceptor_elements if acceptor_elements is not None else default_acceptors

    # Определяем фрагменты для проверки межмолекулярности
    from fragment_detector import detect_fragments
    fragments, _ = detect_fragments(G)

    all_found = []

    for i, atom in enumerate(atoms):
        if interaction_type == 'HB' and atom != 'H':
            continue

        # FIX #3: Находим ВСЕХ ковалентных соседей и выбираем правильного донора
        neighbors = list(G.neighbors(i))
        if not neighbors:
            continue

        # Выбираем соседа, который является валидным донором
        valid_donors = []
        for n_idx in neighbors:
            if atoms[n_idx] in donors:
                valid_donors.append(n_idx)

        if not valid_donors:
            continue

        # Если несколько валидных доноров — берем первого (редкий случай)
        d_idx = valid_donors[0]
        donor_element = atoms[d_idx]

        # Проверка типа донора
        if check_donor_type and donor_element not in donors:
            continue

        # Топологический фильтр
        forbidden_dict = nx.single_source_shortest_path_length(G, i, cutoff=max_covalent_dist)
        forbidden = set(forbidden_dict.keys())

        for a_idx, a_atom in enumerate(atoms):
            if a_idx in forbidden or a_atom not in acceptors:
                continue

            p_h = coords[i]
            p_a = coords[a_idx]
            dist = float(np.linalg.norm(p_h - p_a))

            # Порог по расстоянию
            vdw_sum = vdw_radii.get(atom, 1.2) + vdw_radii.get(a_atom, 1.5)
            if d_max == 'vdw_sum':
                threshold = vdw_sum
            elif d_max == 'vdw_sum_minus_0.2':
                threshold = vdw_sum - 0.2
            else:
                threshold = d_max

            if dist <= threshold:
                angle = calculate_angle(coords[d_idx], p_h, p_a)
                if angle >= a_min:
                    # Проверка межмолекулярности
                    is_intermolecular = fragments.get(i, -1) != fragments.get(a_idx, -1)

                    edge_type = f'HB_{rule}' if return_edge_type else 'HB'
                    all_found.append([
                        i, a_idx, dist, angle, atom, a_atom,
                        edge_type, is_intermolecular
                    ])

    # Фильтрация бифуркации
    if not all_found:
        return []

    by_hydrogen = defaultdict(list)
    for hb in all_found:
        by_hydrogen[hb[0]].append(hb)

    final_filtered = []
    for h_idx in by_hydrogen:
        bonds = sorted(by_hydrogen[h_idx], key=lambda x: x[2])
        min_dist = bonds[0][2]
        threshold = min_dist * 1.10
        for b in bonds:
            if b[2] <= threshold:
                final_filtered.append(tuple(b))
            else:
                break

    return final_filtered


# Обратная совместимость
apply_hb_rules = apply_hb_rules