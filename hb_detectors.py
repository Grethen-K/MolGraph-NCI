import numpy as np

def calculate_angle(p1, p2, p3):
    """Считаем угол D-H...A."""
    v1 = p1 - p2
    v2 = p3 - p2
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm == 0: return 0
    res = np.degrees(np.arccos(np.clip(np.dot(v1, v2) / norm, -1.0, 1.0)))
    return res

def apply_hb_rules(G, atoms, coords, rule='B'):
    """Применяем один из трёх вариантов детектора."""
    # Пороги (пока ставим проверенные значения)
    thresholds = {
        'A': {'dist': 3.0, 'angle': 0},
        'B': {'dist': 2.5, 'angle': 130},
        'C': {'dist': 2.2, 'angle': 150}
    }
    
    params = thresholds[rule]
    h_bonds = []
    
    # Ищем водороды
    for i, atom in enumerate(atoms):
        if atom != 'H': continue
        
        # Находим донора (кто держит этот водород по ковалентной связи)
        neighbors = list(G.neighbors(i))
        if not neighbors: continue
        d_idx = neighbors[0] # Донор
        
        # Ищем потенциальных акцепторов (O, N, F...)
        for a_idx, a_atom in enumerate(atoms):
            if a_atom not in ['O', 'N', 'F', 'S', 'I'] or a_idx == d_idx: continue
            
            # Считаем расстояние H...A
            dist = np.linalg.norm(np.array(coords[i]) - np.array(coords[a_idx]))
            
            if dist < params['dist']:
                angle = calculate_angle(np.array(coords[d_idx]), 
                                        np.array(coords[i]), 
                                        np.array(coords[a_idx]))
                
                if angle >= params['angle']:
                    h_bonds.append((i, a_idx, dist, angle))
    return h_bonds