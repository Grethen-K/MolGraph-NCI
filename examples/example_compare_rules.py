# ============================================================
# 21. examples/example_compare_rules.py examples/example_compare_rules.py
#Пример сравнения правил A/B/C на одной структуре.
# ============================================================
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import main_compare


if __name__ == "__main__":
    xyz_file = os.path.join("..", "test", "M06_zora_HB3_fr.xyz")
    
    print("=" * 60)
    print("Example: Compare rules A/B/C")
    print("=" * 60)
    
    stats_all = main_compare(
        xyz_file,
        inter_type='HB',
        output_dir='results_example_compare'
    )
    
    print("\nComparison complete!")
    print("Check results_example_compare/ for comparison plot.")