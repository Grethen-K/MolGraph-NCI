import sys
from io_utils import load_radii
from covalent_engine import get_covalent_skeleton
from hb_detectors import apply_hb_rules

def main(xyz_file, rule='B', inter_type='HB'):
    print(f"--- Запуск системы MolGraph-NCI ({inter_type}, Правило {rule}) ---")
    
    # 1. Сварка остова - putting the main pipes together
    G, coords = get_covalent_skeleton(xyz_file)
    if not G:
        print("Pizdec! Остов не собрался. Main pipe is clogged!")
        return

    # 2. Loading the gaskets (VdW radii)
    try:
        vdw_radii = load_radii('vdw_radii.json')
    except Exception as e:
        print(f"Blyat! Radii warehouse is closed: {e}")
        return

    # 3. Поиск связей - looking for leaks
    atoms = [G.nodes[i]['element'] for i in range(len(G.nodes))]
    interactions = apply_hb_rules(G, atoms, coords, vdw_radii, rule=rule, interaction_type=inter_type)
    
    print(f"Ковалентных связей: {G.number_of_edges()}")
    print(f"Найдено взаимодействий ({inter_type}): {len(interactions)}")
    
    for inter in interactions:
        # i, a_idx, dist, angle, atom_symbol, acceptor_symbol
        print(f"  Связь: {inter[4]}({inter[0]})--{inter[5]}({inter[1]}) | d={inter[2]:.2f} A | Угол={inter[3]:.1f}")

if __name__ == "__main__":
    # How to use this wrench: python main.py water.xyz B SIGMA
    file = sys.argv[1] if len(sys.argv) > 1 else "water_dimer.xyz"
    mode_rule = sys.argv[2] if len(sys.argv) > 2 else "B"
    inter_type = sys.argv[3].upper() if len(sys.argv) > 3 else "HB"
    main(file, mode_rule, inter_type)