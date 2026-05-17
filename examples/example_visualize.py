# ============================================================
# 22. examples/example_visualize.py examples/example_visualize.py
#Пример визуализации структуры с ВС-графом.
# ============================================================
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from io_utils import read_raw_xyz
from covalent_detector import get_covalent_skeleton_radii
from fragment_detector import assign_fragment_ids_to_graph
from hb_detectors import apply_hb_rules
from graph_builder import build_full_graph
from visualizer import visualize_structure, visualize_hb_network
from config import COVALENT_CONFIG


# Тестовые радиусы
COVALENT_RADII = {
    'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57,
    'P': 1.07, 'S': 1.05, 'Cl': 1.02
}

VDW_RADII = {
    'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'F': 1.47,
    'P': 1.80, 'S': 1.80, 'Cl': 1.75
}


if __name__ == "__main__":
    xyz_file = os.path.join("..", "test", "M06_zora_HB3_fr.xyz")
    
    print("=" * 60)
    print("Example: Visualization")
    print("=" * 60)
    
    # Читаем структуру
    atoms, coords = read_raw_xyz(xyz_file)
    coords_arr = [list(map(float, c)) for c in coords]
    
    # Строим ковалентный скелет
    G_cov, _ = get_covalent_skeleton_radii(
        atoms, coords_arr, COVALENT_RADII,
        scale=COVALENT_CONFIG['scale_factor']
    )
    G_cov, _ = assign_fragment_ids_to_graph(G_cov)
    
    fragment_ids = {node: data.get('fragment_id', 0)
                    for node, data in G_cov.nodes(data=True)}
    covalent_edges = [(u, v) for u, v, d in G_cov.edges(data=True)
                      if d.get('type') == 'covalent']
    
    # Находим ВС для всех трёх правил
    for rule in ('A', 'B', 'C'):
        hbs = apply_hb_rules(G_cov, atoms, coords_arr, VDW_RADII, rule=rule)
        nci_edges = []
        for hb in hbs:
            nci_edges.append({
                'i': hb[0], 'j': hb[1],
                'type': f'HB_{rule}',
                'distance': float(hb[2]),
                'angle': float(hb[3])
            })
        
        # Строим полный граф
        G_full, _ = build_full_graph(
            atoms, coords_arr, covalent_edges, nci_edges,
            fragment_ids=fragment_ids
        )
        
        # Визуализация
        output_dir = 'results_example_visualize'
        os.makedirs(output_dir, exist_ok=True)
        
        vis_path = os.path.join(output_dir, f'viz_rule_{rule}.png')
        visualize_structure(
            G_full,
            title=f"Rule {rule} — {os.path.basename(xyz_file)}",
            save_path=vis_path,
            highlight_suspicious=(rule == 'A')
        )
        print(f"Saved visualization for rule {rule} to {vis_path}")
    
    print("\nVisualization complete!")