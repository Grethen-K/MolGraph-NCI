import numpy as np

def calculate_angle(p1, p2, p3):
    """Считаем угол R-X...A или D-H...A."""
    v1 = p1 - p2 # Vector from X to R
    v2 = p3 - p2 # Vector from X to A
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm == 0: return 0
    res = np.degrees(np.arccos(np.clip(np.dot(v1, v2) / norm, -1.0, 1.0)))
    return res

def apply_hb_rules(G, atoms, coords, vdw_radii, rule='B', interaction_type='HB'):
    """Применяем детекторы. Now with Sigma-holes support!"""
    found = []
    
    # List of greedy atoms that want to grab a connection
    acceptors = ['O', 'N', 'F', 'S', 'Cl', 'Br', 'I', 'P', 'As', 'Se']
    
    # List of atoms that can have a Sigma-hole (Halogens, Chalcogens, Pnictogens)
    sigma_donors = ['F', 'Cl', 'Br', 'I', 'S', 'Se', 'P', 'As']

    for i, atom in enumerate(atoms):
        # 1. Identify the 'Probe' (H for HB, Halogen/etc for SIGMA)
        if interaction_type == 'HB':
            if atom != 'H': continue
        else: # SIGMA mode
            if atom not in sigma_donors: continue

        # 2. Find the 'Parent' atom (the one holding our probe)
        neighbors = list(G.neighbors(i))
        if not neighbors: continue
        parent_idx = neighbors[0] # R in R-X...A
        
        # 3. Check everyone else for a potential 'leak'
        for a_idx, a_atom in enumerate(atoms):
            # Don't connect to yourself or your parent, blyat!
            if a_idx == i or a_idx == parent_idx or a_atom not in acceptors:
                continue
            
            # Geometry check
            p_probe = np.array(coords[i])
            p_parent = np.array(coords[parent_idx])
            p_acceptor = np.array(coords[a_idx])
            
            dist = np.linalg.norm(p_probe - p_acceptor)
            angle = calculate_angle(p_parent, p_probe, p_acceptor)
            
            # 4. Apply Rules (The 'Valera' Thresholds)
            vdw_sum = vdw_radii.get(atom, 1.5) + vdw_radii.get(a_atom, 1.5)
            
            if rule == 'A': # Sloppy - just looking for big holes
                d_max = 3.3 if interaction_type == 'HB' else 4.14
                a_min = 0 # Poxuy on angle!
            elif rule == 'B': # Standard - the chemical sweet spot
                d_max = vdw_sum
                a_min = 130 if interaction_type == 'HB' else 150
            elif rule == 'C': # Strict - high pressure seal
                d_max = vdw_sum - 0.2
                a_min = 150 if interaction_type == 'HB' else 170
            else:
                d_max, a_min = 0, 180 # Close the valve!

            if dist <= d_max and angle >= a_min:
                found.append((i, a_idx, dist, angle, atom, a_atom))
                
    return found