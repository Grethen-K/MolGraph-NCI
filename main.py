import sys
import os
import xyz2mol  # Импортируем скопированный файл

def get_covalent_mol(xyz_file):
    """
    Блок 1: Преобразование XYZ в RDKit Mol объект (ковалентный остов).
    """
    try:
        # Считываем атомы и координаты через встроенные средства xyz2mol
        atoms, charge, coords = xyz2mol.read_xyz_file(xyz_file)
        
        # Получаем список молекул (RDKit объекты)
        # Мы берем первую, так как работаем с кластером как единым целым для начала
        mols = xyz2mol.xyz2mol(atoms, coords, charge=charge, use_graph=True)
        
        if not mols:
            return None
        return mols[0]
    except Exception as e:
        print(f"Ошибка при обработке {xyz_file}: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python main.py <file.xyz>")
    else:
        test_file = sys.argv[1]
        mol = get_covalent_mol(test_file)
        if mol:
            print(f"Успешно! Атомов в остове: {mol.GetNumAtoms()}")
            print(f"Связей в остове: {mol.GetNumBonds()}")