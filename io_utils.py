import json
import os

def load_radii(filename):
    """Грузим радиусы."""
    path = os.path.join('data', filename)
    if not os.path.exists(path):
        # Сообщение в случае ошибки
        raise FileNotFoundError(
            f"Required data file not found: {path}. "
            f"Please ensure '{filename}' exists in the 'data/' directory."
        )
    with open(path, 'r') as f:
        return json.load(f)

def read_raw_xyz(filepath):
    """Читаем координаты"""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    num_atoms = int(lines[0])
    atoms = []
    coords = []
    for line in lines[2:2+num_atoms]:
        parts = line.split()
        atoms.append(parts[0])
        coords.append([float(x) for x in parts[1:4]])
    return atoms, coords