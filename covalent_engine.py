import xyz2mol
import networkx as nx

def get_covalent_skeleton(xyz_file):
    """Используем xyz2mol, чтобы собрать основной стояк."""
    atoms_nums, charge, coords = xyz2mol.read_xyz_file(xyz_file)
    # Получаем RDKit объекты
    mols = xyz2mol.xyz2mol(atoms_nums, coords, charge=charge, use_graph=True)
    
    if not mols:
        return None, None
    
    mol = mols[0]
    # Переводим в NetworkX, чтобы нам было удобно крутить гайки
    G = nx.Graph()
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(), 
                   element=atom.GetSymbol(), 
                   coords=coords[atom.GetIdx()])
    
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(), 
                   bond.GetEndAtomIdx(), 
                   type='covalent')
    
    return G, coords