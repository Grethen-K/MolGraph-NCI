import numpy as np
import networkx as nx
from collections import defaultdict

def calculate_angle(p1, p2, p3):
    """Считаем угол D-H...A. Замер соосности."""
    v1 = p1 - p2
    v2 = p3 - p2
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm == 0: return 0
    res = np.degrees(np.arccos(np.clip(np.dot(v1, v2) / norm, -1.0, 1.0)))
    return res

def apply_hb_rules(G, atoms, coords, vdw_radii, rule='B', interaction_type='HB'):
    """
    Ищем невалентные связи. 
    Теперь с жестким отсечением лишних для одного водорода.
    """
    all_found = []
    acceptor_list = ['O', 'N', 'F', 'S', 'Cl', 'Br', 'I']
    
    # === НАСТРОЙКИ СКЕЛЕТА ===
    # max_covalent_dist = 4: исключаем 1-2, 1-3, 1-4, 1-5 контакты.
    max_covalent_dist = 4 
    # =========================

    for i, atom in enumerate(atoms):
        if interaction_type == 'HB' and atom != 'H': continue
        
        # Находим прямого донора (1-2)
        neighbors = list(G.neighbors(i))
        if not neighbors: continue
        d_idx = neighbors[0] 
        
        # Запретка по ковалентному скелету
        forbidden_dict = nx.single_source_shortest_path_length(G, i, cutoff=max_covalent_dist)
        forbidden = set(forbidden_dict.keys())
            
        for a_idx, a_atom in enumerate(atoms):
            if a_idx in forbidden or a_atom not in acceptor_list:
                continue
            
            p_h = np.array(coords[i])
            p_a = np.array(coords[a_idx])
            dist = np.linalg.norm(p_h - p_a)
            
            # Пороги из склада
            vdw_sum = vdw_radii.get(atom, 1.2) + vdw_radii.get(a_atom, 1.5)
            
            if rule == 'A':
                d_max = 3.25
                a_min = 0 
            elif rule == 'B':
                d_max = vdw_sum
                a_min = 130
            elif rule == 'C':
                d_max = vdw_sum - 0.2
                a_min = 150
            else: continue

            if dist <= d_max:
                angle = calculate_angle(np.array(coords[d_idx]), p_h, p_a)
                if angle >= a_min:
                    # Собираем всё в кучу: (H_idx, A_idx, dist, angle, H_sym, A_sym)
                    all_found.append([i, a_idx, dist, angle, atom, a_atom])
    
    # === ШАГ СОРТИРОВКИ И ФИЛЬТРАЦИИ ===
    if not all_found:
        return []

    # Группируем связи по индексу водорода
    by_hydrogen = defaultdict(list)
    for hb in all_found:
        by_hydrogen[hb[0]].append(hb)

    final_filtered_hb = []
    
    for h_idx in by_hydrogen:
        # 1. Сортируем связи для этого водорода по длине (от коротких к длинным)
        bonds = sorted(by_hydrogen[h_idx], key=lambda x: x[2])
        
        # 2. Берем самую короткую связь как эталон
        min_dist = bonds[0][2]
        
        # 3. Оставляем только те, что не длиннее эталона на 10%
        # (Если d > min_dist * 1.10 — это не бифуркатные)
        threshold = min_dist * 1.10
        
        for b in bonds:
            if b[2] <= threshold:
                final_filtered_hb.append(tuple(b))
            else:
                # Всё, что дальше — игнорируем
                continue
                
    return final_filtered_hb