import sys
from io_utils import load_radii
from covalent_engine import get_covalent_skeleton
from hb_detectors import apply_hb_rules
from sigma_detectors import apply_sigma_rules

def main(xyz_file, rule='B', inter_type='HB'):
    print(f"--- Запуск системы MolGraph-NCI ({inter_type}, Правило {rule}) ---")
    
    # 1. Сварка остова
    G, coords = get_covalent_skeleton(xyz_file)
    if not G:
        print("Pizdec! Остов не собрался.")
        return

    # Индексация атомов (Шаг 1 из ТЗ)
    atoms = [G.nodes[i]['element'] for i in range(len(G.nodes))]
    
    print(f"Загружено атомов: {len(atoms)}")
    print(f"Ковалентных связей найдено: {G.number_of_edges()}")
    
    # Показываем список ковалентных связей с водородами (Шаг 2 из ТЗ)
    print("Список ковалентных связей (H-Donor):")
    for u, v in G.edges():
        if atoms[u] == 'H' or atoms[v] == 'H':
            print(f"  [{u}] {atoms[u]} -- {atoms[v]} [{v}]")

    # 2. Загрузка прокладок
    vdw_radii = load_radii('vdw_radii.json')

    # 3. Поиск HB/sigma с вычитанием (Шаг 3 из ТЗ)
    if inter_type == "HB":
        h_bonds = apply_hb_rules(G, atoms, coords, vdw_radii, rule=rule, interaction_type=inter_type)
    else:
        h_bonds = apply_sigma_rules(G, atoms, coords, vdw_radii, rule=rule)

    print(f"\nИстинных NCI ({inter_type}) найдено: {len(h_bonds)}")
    for hb in h_bonds:
        print(f"  Связь: {hb[4]}({hb[0]})--{hb[5]}({hb[1]}) | d={hb[2]:.2f} A | Угол={hb[3]:.1f}")

if __name__ == "__main__":
    file = sys.argv[1] if len(sys.argv) > 1 else "water_dimer.xyz"
    mode = sys.argv[2] if len(sys.argv) > 2 else "B"
    inter = sys.argv[3].upper() if len(sys.argv) > 3 else "HB"
    main(file, mode, inter)