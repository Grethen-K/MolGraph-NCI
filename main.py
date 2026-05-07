import sys
from io_utils import load_radii
from covalent_engine import get_covalent_skeleton
from hb_detectors import apply_hb_rules

def main(xyz_file, rule='B'):
    print(f"--- Запуск системы MolGraph-NCI (Правило {rule}) ---")
    
    # 1. Сварка остова
    G, coords = get_covalent_skeleton(xyz_file)
    if not G:
        print("Pizdec! Остов не собрался.")
        return

    # 2. Поиск H-связей
    atoms = [G.nodes[i]['element'] for i in range(len(G.nodes))]
    h_bonds = apply_hb_rules(G, atoms, coords, rule=rule)
    
    print(f"Ковалентных связей: {G.number_of_edges()}")
    print(f"Водородных связей найдено: {len(h_bonds)}")
    
    for hb in h_bonds:
        print(f"  Связь: H({hb[0]})--A({hb[1]}) | d={hb[2]:.2f} A | Угол={hb[3]:.1f}")

if __name__ == "__main__":
    file = sys.argv[1] if len(sys.argv) > 1 else "water_dimer.xyz"
    mode = sys.argv[2] if len(sys.argv) > 2 else "B"
    main(file, mode)