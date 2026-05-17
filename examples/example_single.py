# ============================================================
# 19. examples/example_single.py  examples/example_single.py
#Пример обработки одного XYZ-файла.
# ============================================================
import os
import sys

# Добавляем родительскую директорию в путь
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import main_single


if __name__ == "__main__":
    # Путь к тестовому файлу
    xyz_file = os.path.join("..", "test", "M06_zora_HB3_fr.xyz")
    
    # Обработка с правилом B (химически осмысленным)
    print("=" * 60)
    print("Example: Single structure analysis")
    print("=" * 60)
    
    G, stats = main_single(
        xyz_file,
        rule='B',
        inter_type='HB',
        save_results=True,
        visualize=True,
        output_dir='results_example_single'
    )
    
    print("\nDone! Check results_example_single/ for outputs.")