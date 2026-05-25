# ============================================================
# Централизованная конфигурация проекта.
# ============================================================
import os

# Пути
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
TEST_DIR = os.path.join(BASE_DIR, 'test')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Правила для водородных связей
HB_RULES = {
    'A': {
        'name': 'soft',
        'description': 'Слишком мягкое правило — возможно выдаст много "лишних" контактов',
        'max_distance': 3.25,  # Å как vdw_самого длинного + H → Sb or Te + 1.2
        'min_angle': 0,  # ° (не учитывается)
        'donor_elements': ['O', 'N', 'S'],
        'acceptor_elements': ['O', 'N', 'F', 'S', 'Cl', 'Br', 'I'],
    },
    'B': {
        'name': 'sensible',
        'description': 'Химически осмысленное правило',
        'max_distance': 'vdw_sum',  # сумма ВдВ радиусов
        'min_angle': 130,  # °
        'donor_elements': ['O', 'N'],
        'acceptor_elements': ['O', 'N', 'F', 'S', 'Cl', 'Br', 'I'],
    },
    'C': {
        'name': 'strict',
        'description': 'Слишком жёсткое правило — может терять реальные ВС',
        'max_distance': 'vdw_sum * 0.93',  # ~vdw_sum - 0.2 Å
        'min_angle': 150,  # °
        'donor_elements': ['O', 'N'],
        'acceptor_elements': ['O', 'N', 'F'],
    }
}

# Правила для сигма-дырочных взаимодействий
SIGMA_RULES = {
    'A': {
        'name': 'soft',
        'max_distance': 4.14,  # Å как vdw_самого длинного * 2 → Sb or Te
        'min_angle_1': 0,  # °
        'min_angle_2': 0,  # ° (не используется)
        'donor_elements': ['F', 'Cl', 'Br', 'I', 'S', 'Se', 'Te', 'P', 'As', 'Sb'],
        'acceptor_elements': ['O', 'N', 'F', 'S', 'Cl', 'Br', 'I', 'P', 'As', 'Se'],
    },
    'B': {
        'name': 'sensible',
        'max_distance': 'vdw_sum',
        'min_angle_1': 110,  # °
        'min_angle_2': 90,  # °
        'donor_elements': ['F', 'Cl', 'Br', 'I', 'S', 'Se', 'Te', 'P', 'As', 'Sb'],
        'acceptor_elements': ['O', 'N', 'F', 'S', 'Cl', 'Br', 'I', 'P', 'As', 'Se'],
    },
    'C': {
        'name': 'strict',
        # FIX #7: Унифицирована нотация
        'max_distance': 'vdw_sum * 0.9',  # 90% суммы ВдВ
        'min_angle_1': 130,  # °
        'min_angle_2': 110,  # °
        'donor_elements': ['F', 'Cl', 'Br', 'I', 'S', 'Se', 'Te', 'P', 'As', 'Sb'],
        'acceptor_elements': ['O', 'N', 'F', 'S', 'Cl', 'Br', 'I'],
    }
}

# Ковалентные связи
COVALENT_CONFIG = {
    'scale_factor': 1.15,
    'tolerance': 0.0,
    'max_covalent_dist_topology': 4,  # исключение 1-2...1-5
}

# Шум
NOISE_CONFIG = {
    'sigma_values': (0.01, 0.02, 0.03, 0.04, 0.05),
    'n_repeats': 5,
    'seed_base': 42,
}

# Визуализация
VISUALIZATION_CONFIG = {
    'suspicious_threshold_dist': 2.8,  # Å
    'suspicious_threshold_angle': 100,  # °
    'dpi': 150,
    'figsize': (12, 10),
}

# Batch processing
BATCH_CONFIG = {
    'n_workers': 1,  # 1 = последовательно, >1 = multiprocessing
    'output_formats': ['json', 'csv'],
}