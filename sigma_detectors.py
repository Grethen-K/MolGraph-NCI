import numpy as np
import networkx as nx
from collections import defaultdict

def calculate_angle(p_center, p_attached, p_target):
    """
    Вычисляет угол между тремя точками.
    p_center: Координаты центральной вершины (вершина угла).
    p_attached: Координаты первого связанного атома.
    p_target: Координаты второго связанного атома.
    """
    vector_1 = p_attached - p_center
    vector_2 = p_target - p_center
    
    norm = np.linalg.norm(vector_1) * np.linalg.norm(vector_2)
    if norm == 0:
        return 0.0
    
    # Ограничение значения для arccos во избежание вычислительных ошибок
    cosine = np.clip(np.dot(vector_1, vector_2) / norm, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))

def apply_sigma_rules(G, atoms, coords, vdw_radii, rule='A'):
    """
    Поиск сигма-дырочных взаимодействий с анализом геометрии двух углов.
    
    Критерии:
    Угол 1 (R1-X...A): стремится к 180 градусам (соосность).
    Угол 2 (X...A-R2): стремится к 90 градусам для XB, и только для них, тк ChB & PnB нет этих углов.
    """
    all_found = []
    acceptors = ['O', 'N', 'F', 'S', 'Cl', 'Br', 'I', 'P', 'As', 'Se']
    sigma_donors = ['F', 'Cl', 'Br', 'I', 'At', 'O', 'S', 'Se', 'Te', 'N', 'P', 'As', 'Sb', 'Bi']
    
    # Исключение внутренних контактов до 1-5 включительно   # варьировать от 3 до 5
    max_covalent_dist = 4 

    for i, atom in enumerate(atoms):
        if atom not in sigma_donors:
            continue
        
        # Поиск ковалентного соседа донора (R1)
        d_neighbors = list(G.neighbors(i))
        if not d_neighbors:
            continue
        r1_idx = d_neighbors[0]
        
        # Топологический фильтр
        forbidden_dict = nx.single_source_shortest_path_length(G, i, cutoff=max_covalent_dist)
        forbidden = set(forbidden_dict.keys())
            
        for a_idx, a_atom in enumerate(atoms):
            if a_idx in forbidden or a_atom not in acceptors:
                continue
            
            p_x = np.array(coords[i])
            p_a = np.array(coords[a_idx])
            dist = np.linalg.norm(p_x - p_a)
            
            # Сумма ВдВ радиусов из словаря
            vdw_sum = vdw_radii.get(atom, 1.5) + vdw_radii.get(a_atom, 1.5)
            
            # Определение граничных условий сценариев
            if rule == 'A':
                d_max = 4.14
                angle_1_min = 0.0
            elif rule == 'B':
                d_max = vdw_sum
                angle_1_min = 110.0
            elif rule == 'C':
                # Сценарий C: дистанция сокращена на 10% от суммы ВдВ
                d_max = vdw_sum * 0.9
                angle_1_min = 130.0
            else:
                continue

            if dist <= d_max:
                p_r1 = np.array(coords[r1_idx])
                
                # Угол 1: R1 - X ... A (центральный атом X)
                angle_1 = calculate_angle(p_x, p_r1, p_a)
                
                if angle_1 >= angle_1_min:
                    # Поиск ковалентного соседа акцептора (R2) для расчета второго угла
                    a_neighbors = list(G.neighbors(a_idx))
                    angle_2 = 0.0
                    if a_neighbors:
                        r2_idx = a_neighbors[0]
                        p_r2 = np.array(coords[r2_idx])
                        # Угол 2: X ... A - R2 (центральный атом A)
                        angle_2 = calculate_angle(p_a, p_x, p_r2)
                    
                    all_found.append([i, a_idx, dist, angle_1, angle_2, atom, a_atom])
    
    if not all_found:
        return []

    # Группировка и фильтрация по дистанции (порог 20%) # Варьировать от 15 до 30
    by_donor = defaultdict(list)
    for res in all_found:
        by_donor[res[0]].append(res)

    final_results = []
    for d_idx in by_donor:
        interactions = sorted(by_donor[d_idx], key=lambda x: x[2])
        min_dist = interactions[0][2]
        threshold = min_dist * 1.20   # Варьировать от 1,15 до 1,30
        
        for inter in interactions:
            if inter[2] <= threshold:
                final_results.append(tuple(inter))
                
    return final_results