import numpy as np
import networkx as nx

def calculate_angle(p1, p2, p3):
    """Считаем угол D-H...A. Как замер соосности двух труб."""
    v1 = p1 - p2
    v2 = p3 - p2
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm == 0: return 0
    res = np.degrees(np.arccos(np.clip(np.dot(v1, v2) / norm, -1.0, 1.0)))
    return res

def apply_hb_rules(G, atoms, coords, vdw_radii, rule='B', interaction_type='HB'):
    """
    Ищем невалентные связи. 
    Жестко вычитаем внутренние детали молекулы (ковалентные цепочки).
    """
    h_bonds = []
    acceptor_list = ['O', 'N', 'F', 'S', 'Cl', 'Br', 'I']
    
    # === РЫЧАГ УПРАВЛЕНИЯ (ТОТ САМЫЙ КОММЕНТАРИЙ) ===
    # max_covalent_dist = 3: исключаем 1-2, 1-3, 1-4 контакты.
    # max_covalent_dist = 4: исключаем еще и 1-5 контакты.
    # Для муравьиной кислоты (HO-CH=O) ставь 4, чтобы H(OH) не видел O(C=O).
    max_covalent_dist = 4 
    # ===============================================

    for i, atom in enumerate(atoms):
        if interaction_type == 'HB' and atom != 'H': continue
        
        # Находим прямого донора (1-2), чтобы знать, откуда считать углы
        neighbors = list(G.neighbors(i))
        if not neighbors: continue
        d_idx = neighbors[0] 
        
        # СОСТАВЛЯЕМ СПИСОК «ЗАПРЕТКИ» (Substracting the pipes)
        # Находим всех, до кого можно дотянуться по трубам за N шагов
        forbidden_dict = nx.single_source_shortest_path_length(G, i, cutoff=max_covalent_dist)
        forbidden = set(forbidden_dict.keys())
            
        for a_idx, a_atom in enumerate(atoms):
            # ВЫЧИТАНИЕ: если акцептор слишком близко по ковалентной цепи — это не HB!
            if a_idx in forbidden:
                continue
                
            if a_atom not in acceptor_list: continue
            
            # Геометрия
            p_h = np.array(coords[i])
            p_a = np.array(coords[a_idx])
            dist = np.linalg.norm(p_h - p_a)
            
            # Радиусы из нашего склада (json)
            vdw_sum = vdw_radii.get(atom, 1.2) + vdw_radii.get(a_atom, 1.5)
            
            if rule == 'A':
                d_max = 3.3
                a_min = 0  # Sloppy mode
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
                    h_bonds.append((i, a_idx, dist, angle, atom, a_atom))
                    
    return h_bonds