#============================================================
# 12. __init__.py
#MolGraph-NCI: Molecular Graph Builder with Non-Covalent Interactions
#Модуль для построения графов молекулярных структур с учётом
#ковалентных и невалентных взаимодействий (водородные связи,
#сигма-дырочные взаимодействия).
# ============================================================
__version__ = "0.2.0"
__author__ = "Grethen-K"

from .io_utils import read_raw_xyz, load_radii
from .distance_matrix import compute_distance_matrix
from .covalent_detector import get_covalent_skeleton_radii, get_covalent_skeleton_combined
from .fragment_detector import detect_fragments, assign_fragment_ids_to_graph
from .hb_detectors import apply_hb_rules
from .sigma_detectors import apply_sigma_rules
from .graph_builder import build_full_graph, to_pyg_data, get_hb_subgraph
from .metrics import compute_hb_statistics, compute_sigma_statistics, compute_dataset_statistics, compare_rules_statistics
from .noise_stability import evaluate_noise_stability, add_coordinate_noise, compute_edge_jaccard
from .visualizer import visualize_structure, visualize_compare_rules, visualize_hb_network
from .batch_processor import process_single_structure, process_dataset

__all__ = [
    'read_raw_xyz', 'load_radii',
    'compute_distance_matrix',
    'get_covalent_skeleton_radii', 'get_covalent_skeleton_combined',
    'detect_fragments', 'assign_fragment_ids_to_graph',
    'apply_hb_rules', 'apply_sigma_rules',
    'build_full_graph', 'to_pyg_data', 'get_hb_subgraph',
    'compute_hb_statistics', 'compute_sigma_statistics',
    'compute_dataset_statistics', 'compare_rules_statistics',
    'evaluate_noise_stability', 'add_coordinate_noise', 'compute_edge_jaccard',
    'visualize_structure', 'visualize_compare_rules', 'visualize_hb_network',
    'process_single_structure', 'process_dataset',
]
