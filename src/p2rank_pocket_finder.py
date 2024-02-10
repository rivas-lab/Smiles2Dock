import numpy  as np
import pandas as pd

from deepchem.utils.coordinate_box_utils import CoordinateBox
from deepchem.dock.pose_generation       import BindingPocketFinder

class P2RankPocketFinder(BindingPocketFinder):
    
    def __init__(self, pocket_csv_path, ligand_mol=None, threshold=None, padding=5.0):
        self.pocket_csv_path = pocket_csv_path
        self.ligand_mol      = ligand_mol
        self.threshold       = threshold
        self.padding         = padding
        self.num_pockets     = None

    @staticmethod
    def get_binding_box(center: np.ndarray, padding: float) -> CoordinateBox:
        x_bounds = (center[0] - padding, center[0] + padding)
        y_bounds = (center[1] - padding, center[1] + padding)
        z_bounds = (center[2] - padding, center[2] + padding)
        return CoordinateBox(x_bounds, y_bounds, z_bounds)

    def read_pockets_file(self):
        pocket_csv_file = pd.read_csv(self.pocket_csv_path)
        pocket_csv_file.columns = pocket_csv_file.columns.str.strip()
        
        if self.threshold is not None:
            pocket_csv_file = pocket_csv_file[pocket_csv_file.probability > self.threshold]
        
        self.num_pockets = len(pocket_csv_file)
        
        centers_x = pocket_csv_file.center_x.tolist()
        centers_y = pocket_csv_file.center_y.tolist()
        centers_z = pocket_csv_file.center_z.tolist()
        
        pockets_coords = zip(centers_x, centers_y, centers_z)
        return pockets_coords

    @staticmethod
    def calculate_ligand_dimensions(ligand_mol):
        
        conf   = ligand_mol.GetConformer()
        coords = [conf.GetAtomPosition(i) for i in range(ligand_mol.GetNumAtoms())]
        coords = np.array([[pos.x, pos.y, pos.z] for pos in coords])
        
        min_coords = coords.min(axis=0)
        max_coords = coords.max(axis=0)
        
        dimensions = max_coords - min_coords
        
        return dimensions

    def find_pockets(self, protein_file=None):
        
        if self.ligand_mol:
            ligand_dimensions = self.calculate_ligand_dimensions(self.ligand_mol)
            self.padding      = max(ligand_dimensions) / 2 + self.padding  # Adjust padding based on ligand size
        
        pockets_coords = self.read_pockets_file()
        
        boxes = []
        
        for coords in pockets_coords:
            center = np.array(coords)
            box = self.get_binding_box(center, self.padding)
            boxes.append(box)
        
        return boxes
    