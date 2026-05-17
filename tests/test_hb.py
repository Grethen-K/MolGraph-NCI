# ============================================================
# 14. tests/test_hb.py Unit tests for hydrogen bond detectors.
# ============================================================
import pytest
import numpy as np
import networkx as nx
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from covalent_detector import get_covalent_skeleton_radii
from hb_detectors import apply_hb_rules


# Тестовые данные: димер воды (известная структура)
WATER_DIMER_ATOMS = ['O', 'H', 'H', 'O', 'H', 'H']
# Мономер 1: O-H...O
# Мономер 2: принимает ВС
WATER_DIMER_COORDS = [
    [0.0, 0.0, 0.0],      # O0
    [0.96, 0.0, 0.0],     # H1 - донор
    [-0.24, 0.93, 0.0],   # H2
    [2.8, 0.0, 0.0],      # O3 - акцептор
    [3.5, 0.5, 0.0],      # H4
    [3.0, -0.8, 0.0],     # H5
]

COVALENT_RADII = {
    'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57
}

VDW_RADII = {
    'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'F': 1.47
}


def build_water_dimer_graph():
    """Строит ковалентный граф димера воды."""
    G, _ = get_covalent_skeleton_radii(
        WATER_DIMER_ATOMS, WATER_DIMER_COORDS, COVALENT_RADII, scale=1.15
    )
    return G


def test_hb_rule_B_finds_dimer():
    """Правило B должно найти 1 водородную связь в димере воды."""
    G = build_water_dimer_graph()
    coords = np.array(WATER_DIMER_COORDS)
    
    hbs = apply_hb_rules(
        G, WATER_DIMER_ATOMS, coords, VDW_RADII,
        rule='B', interaction_type='HB'
    )
    
    # Должна быть 1 ВС: H1...O3
    assert len(hbs) >= 1
    # Проверяем, что найдена связь с O3
    found_o3 = any(hb[1] == 3 for hb in hbs)
    assert found_o3


def test_hb_rule_A_more_permissive():
    """Правило A должно быть мягче B (не меньше связей)."""
    G = build_water_dimer_graph()
    coords = np.array(WATER_DIMER_COORDS)
    
    hbs_A = apply_hb_rules(G, WATER_DIMER_ATOMS, coords, VDW_RADII, rule='A')
    hbs_B = apply_hb_rules(G, WATER_DIMER_ATOMS, coords, VDW_RADII, rule='B')
    
    assert len(hbs_A) >= len(hbs_B)


def test_hb_rule_C_more_strict():
    """Правило C должно быть жёстче B (не больше связей)."""
    G = build_water_dimer_graph()
    coords = np.array(WATER_DIMER_COORDS)
    
    hbs_B = apply_hb_rules(G, WATER_DIMER_ATOMS, coords, VDW_RADII, rule='B')
    hbs_C = apply_hb_rules(G, WATER_DIMER_ATOMS, coords, VDW_RADII, rule='C')
    
    assert len(hbs_C) <= len(hbs_B)


def test_hb_edge_type():
    """Проверяем, что возвращается правильный тип ребра."""
    G = build_water_dimer_graph()
    coords = np.array(WATER_DIMER_COORDS)
    
    hbs = apply_hb_rules(G, WATER_DIMER_ATOMS, coords, VDW_RADII, rule='B')
    
    for hb in hbs:
        assert hb[6] == 'HB_B'  # edge_type


def test_hb_donor_check():
    """Проверяем, что донор проверяется (O или N)."""
    G = build_water_dimer_graph()
    coords = np.array(WATER_DIMER_COORDS)
    
    # С включённой проверкой донора
    hbs_checked = apply_hb_rules(
        G, WATER_DIMER_ATOMS, coords, VDW_RADII,
        rule='B', check_donor_type=True
    )
    
    # С выключенной проверкой
    hbs_unchecked = apply_hb_rules(
        G, WATER_DIMER_ATOMS, coords, VDW_RADII,
        rule='B', check_donor_type=False
    )
    
    # С проверкой должно быть не больше, чем без
    assert len(hbs_checked) <= len(hbs_unchecked)

