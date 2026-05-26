"""
main.py
Tochka vkhoda dlya CLI i programmnogo API.
"""
import sys
import os
import json
import argparse

import numpy as np
import networkx as nx

from config import (
    HB_RULES, SIGMA_RULES, COVALENT_CONFIG, NOISE_CONFIG,
    VISUALIZATION_CONFIG, BATCH_CONFIG, RESULTS_DIR
)
from io_utils import read_raw_xyz, load_radii
from covalent_detector import get_covalent_skeleton_radii
from fragment_detector import assign_fragment_ids_to_graph
from hb_detectors import apply_hb_rules
from sigma_detectors import apply_sigma_rules
from graph_builder import build_full_graph
from metrics import compute_hb_statistics, compute_sigma_statistics
from noise_stability import evaluate_noise_stability
from visualizer import visualize_structure, visualize_compare_rules
from batch_processor import process_single_structure, process_dataset
from distance_matrix import compute_distance_matrix


def build_graph_for_structure(atoms, coords, covalent_radii, vdw_radii,
                               rule='B', inter_type='HB',
                               use_radii_fallback=True):
    """
    Stroit polnyy graf dlya struktury.
    """
    coords_arr = np.array(coords, dtype=float)

    # Kovalentnyy skelet
    G_cov, dist_mat = get_covalent_skeleton_radii(
        atoms, coords_arr, covalent_radii,
        scale=COVALENT_CONFIG['scale_factor'],
        tolerance=COVALENT_CONFIG['tolerance']
    )

    # Fragmenty
    G_cov, num_frags = assign_fragment_ids_to_graph(G_cov)
    fragment_ids = {node: data.get('fragment_id', 0)
                    for node, data in G_cov.nodes(data=True)}

    covalent_edges = [(u, v) for u, v, d in G_cov.edges(data=True)
                      if d.get('type') == 'covalent']

    # NCI
    if inter_type == 'HB':
        nci_raw = apply_hb_rules(
        G_cov, atoms, coords_arr, vdw_radii,
        rule=rule, interaction_type='HB',
        include_weak_donors=True 
        )
        nci_edges = []
        for hb in nci_raw:
            nci_edges.append({
                'i': hb[0], 'j': hb[1],
                'type': f'HB_{rule}',
                'distance': float(hb[2]),
                'angle': float(hb[3]),
                'donor_element': hb[4],
                'acceptor_element': hb[5],
                'is_intermolecular': hb[7] if len(hb) > 7 else False
            })
    else:
        nci_raw = apply_sigma_rules(
            G_cov, atoms, coords_arr, vdw_radii, rule=rule
        )
        nci_edges = []
        for sg in nci_raw:
            nci_edges.append({
                'i': sg[0], 'j': sg[1],
                'type': f'sigma_{rule}',
                'distance': float(sg[2]),
                'angle_1': float(sg[3]),
                'angle_2': float(sg[4]),
                'donor_element': sg[5],
                'acceptor_element': sg[6]
            })

    # Polnyy graf
    G_full, graph_data = build_full_graph(
        atoms, coords_arr, covalent_edges, nci_edges,
        fragment_ids=fragment_ids
    )

    return G_full, graph_data, G_cov, dist_mat


def main_single(xyz_file, rule='B', inter_type='HB',
                save_results=True, visualize=True,
                output_dir=RESULTS_DIR):
    """
    Obrabotka odnogo XYZ-fayla.
    """
    print(f"\n{'='*60}")
    print(f"MolGraph-NCI: Single Structure Analysis")
    print(f"{'='*60}")
    print(f"File: {xyz_file}")
    print(f"Rule: {rule} | Interaction: {inter_type}")

    atoms, coords = read_raw_xyz(xyz_file)
    covalent_radii = load_radii('covalent_radii.json')
    vdw_radii = load_radii('vdw_radii.json')

    G_full, graph_data, G_cov, dist_mat = build_graph_for_structure(
        atoms, coords, covalent_radii, vdw_radii, rule, inter_type
    )

    # Metriki
    if inter_type == 'HB':
        stats = compute_hb_statistics(G_full, hb_type=f'HB_{rule}')
    else:
        stats = compute_sigma_statistics(G_full, sigma_type=f'sigma_{rule}')

    # Vyvod
    print(f"\nAtoms: {len(atoms)}")
    print(f"Covalent bonds: {G_cov.number_of_edges()}")
    print(f"Fragments: {len(set(nx.get_node_attributes(G_cov, 'fragment_id').values()))}")
    print(f"NCI found: {stats.get('num_hb', stats.get('num_sigma', 0))}")

    if inter_type == 'HB':
        print(f"\nH-bond statistics:")
        print(f"  Count: {stats['num_hb']}")
        print(f"  Avg distance: {stats['avg_distance']:.3f} A")
        print(f"  Avg angle: {stats['avg_angle']:.1f} deg")
        print(f"  Max component size: {stats['max_component_size']}")
        print(f"  Suspicious contacts: {stats['num_suspicious']}")

    # ВЫВОД КОНКРЕТНЫХ СВЯЗЕЙ (+ 0-based индексация → нумерация с 1!)
        print(f"\nH-bond details (1-based indexing, 0-based in brackets):")
        for u, v, d in G_full.edges(data=True):
            if d.get('type', '').startswith('HB_'):
                print(f" {G_full.nodes[u]['element']}{u+1}(0:{u}) --- "
                      f"{G_full.nodes[v]['element']}{v+1}(0:{v}) "
                      f"(dist={d.get('distance',0):.3f}A, "
                      f"angle={d.get('angle',0):.1f}deg)")
    elif inter_type == 'sigma':
        print(f"\nSigma-bond details (1-based indexing, 0-based in brackets):")
        for u, v, d in G_full.edges(data=True):
            if d.get('type', '').startswith('sigma_'):
                print(f" {G_full.nodes[u]['element']}{u+1}(0:{u}) --- "
                      f"{G_full.nodes[v]['element']}{v+1}(0:{v}) "
                      f"(dist={d.get('distance',0):.3f}A, "
                      f"angle1={d.get('angle_1',0):.1f}deg)")

    # Sokhranenie
    if save_results:
        os.makedirs(output_dir, exist_ok=True)
        basename = os.path.splitext(os.path.basename(xyz_file))[0]
        result = {
            'filename': os.path.basename(xyz_file),
            'num_atoms': len(atoms),
            'rule': rule,
            'interaction_type': inter_type,
            'statistics': stats,
            'edge_list': graph_data['edge_list'],
            'edge_index': graph_data['edge_index'].tolist(),
            'node_features': graph_data['node_features'].tolist(),
        }
        out_path = os.path.join(output_dir, f"{basename}_{inter_type}_{rule}.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nSaved results to {out_path}")

    # Vizualizatsiya
    if visualize:
        try:
            vis_path = os.path.join(output_dir, f"{basename}_{inter_type}_{rule}.png")
            visualize_structure(
                G_full,
                title=f"{os.path.basename(xyz_file)} - {inter_type} Rule {rule}",
                save_path=vis_path,
                highlight_suspicious=(rule == 'A'),
                suspicious_threshold_dist=VISUALIZATION_CONFIG['suspicious_threshold_dist'],
                suspicious_threshold_angle=VISUALIZATION_CONFIG['suspicious_threshold_angle']
            )
        except Exception as e:
            print(f"Visualization error: {e}")

    return G_full, stats


def main_compare(xyz_file, inter_type='HB', output_dir=RESULTS_DIR):
    """
    Sravnenie pravil A/B/C na odnoy strukture.
    """
    print(f"\n{'='*60}")
    print(f"MolGraph-NCI: Compare Rules A/B/C")
    print(f"{'='*60}")

    atoms, coords = read_raw_xyz(xyz_file)
    covalent_radii = load_radii('covalent_radii.json')
    vdw_radii = load_radii('vdw_radii.json')

    graphs = {}
    stats_all = {}

    for rule in ('A', 'B', 'C'):
        G_full, _, _, _ = build_graph_for_structure(
            atoms, coords, covalent_radii, vdw_radii, rule, inter_type
        )
        graphs[rule] = G_full
        if inter_type == 'HB':
            stats_all[rule] = compute_hb_statistics(G_full, hb_type=f'HB_{rule}')
        else:
            stats_all[rule] = compute_sigma_statistics(G_full, sigma_type=f'sigma_{rule}')

    # Tablitsa sravneniya
    print(f"\n{'Rule':<8} {'NCI':<8} {'Avg d':<10} {'Avg angle':<12} {'Max comp':<10}")
    print('-' * 50)
    for rule in ('A', 'B', 'C'):
        s = stats_all[rule]
        if inter_type == 'HB':
            print(f"{rule:<8} {s['num_hb']:<8} {s['avg_distance']:<10.3f} "
                  f"{s['avg_angle']:<12.1f} {s['max_component_size']:<10}")
        else:
            print(f"{rule:<8} {s['num_sigma']:<8} {s['avg_distance']:<10.3f} "
                  f"{s['avg_angle_1']:<12.1f} {'-':<10}")

    # Vizualizatsiya sravneniya
    try:
        basename = os.path.splitext(os.path.basename(xyz_file))[0]
        vis_path = os.path.join(output_dir, f"{basename}_{inter_type}_compare.png")
        visualize_compare_rules(
            graphs['A'], graphs['B'], graphs['C'],
            title_prefix=f"{basename} -",
            save_path=vis_path
        )
        print(f"\nSaved comparison plot to {vis_path}")
    except Exception as e:
        print(f"Comparison visualization error: {e}")

    return stats_all


def main_batch(xyz_dir, rules=('A', 'B', 'C'), inter_types=('HB',),
               output_dir=RESULTS_DIR, n_workers=1):
    """
    Batch-obrabotka direktorii s XYZ-faylami.
    """
    print(f"\n{'='*60}")
    print(f"MolGraph-NCI: Batch Processing")
    print(f"{'='*60}")

    covalent_radii = load_radii('covalent_radii.json')
    vdw_radii = load_radii('vdw_radii.json')

    result = process_dataset(
        xyz_dir, output_dir=output_dir,
        rules=rules, inter_types=inter_types,
        covalent_radii=covalent_radii,
        vdw_radii=vdw_radii,
        compute_noise=True,
        n_workers=n_workers
    )

    print(f"\nBatch processing complete!")
    print(f"  Structures processed: {result['num_structures']}")
    print(f"  Results saved to: {output_dir}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description='MolGraph-NCI: Molecular Graph Builder with Non-Covalent Interactions'
    )
    parser.add_argument('input', nargs='?',
                        help='XYZ file or directory')
    parser.add_argument('--mode', choices=['single', 'compare', 'batch'],
                        default='single',
                        help='Processing mode')
    parser.add_argument('--rule', choices=['A', 'B', 'C'], default='B',
                        help='HB rule (A=soft, B=sensible, C=strict)')
    parser.add_argument('--inter', choices=['HB', 'sigma'], default='HB',
                        help='Interaction type')
    parser.add_argument('--output', default=RESULTS_DIR,
                        help='Output directory')
    parser.add_argument('--no-viz', action='store_true',
                        help='Disable visualization')
    parser.add_argument('--no-save', action='store_true',
                        help='Disable saving results')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel workers (batch mode)')

    args = parser.parse_args()

    if args.input and not os.path.exists(args.input):
        parser.error(f"Input path does not exist: {args.input}")

    if not args.input:
        args.input = os.path.join(os.path.dirname(__file__), 'test')
        args.mode = 'batch'

    if args.mode == 'single':
        main_single(
            args.input, rule=args.rule, inter_type=args.inter,
            save_results=not args.no_save,
            visualize=not args.no_viz,
            output_dir=args.output
        )
    elif args.mode == 'compare':
        main_compare(args.input, inter_type=args.inter, output_dir=args.output)
    elif args.mode == 'batch':
        main_batch(
            args.input,
            rules=('A', 'B', 'C'),
            inter_types=(args.inter,),
            output_dir=args.output,
            n_workers=args.workers
        )


if __name__ == "__main__":
    main()
