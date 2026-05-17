# ============================================================
# 20. examples/example_batch.py examples/example_batch.py
#Пример batch-обработки директории с XYZ-файлами.
# ============================================================
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import main_batch


if __name__ == "__main__":
    # Директория с тестовыми файлами
    xyz_dir = os.path.join("..", "test")
    
    print("=" * 60)
    print("Example: Batch processing")
    print("=" * 60)
    
    result = main_batch(
        xyz_dir,
        rules=('A', 'B', 'C'),
        inter_types=('HB', 'sigma'),
        output_dir='results_example_batch',
        n_workers=1  # Для демонстрации — последовательно
    )
    
    print(f"\nProcessed {result['num_structures']} structures.")
    print("Check results_example_batch/ for outputs.")