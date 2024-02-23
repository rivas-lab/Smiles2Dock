import logging

logging.getLogger("deepchem").setLevel(logging.ERROR)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s\n', level=logging.INFO)

import os

from gc         import collect
from typing     import Optional

from openmm.app import PDBFile
from pdbfixer   import PDBFixer
from rdkit      import Chem
from rdkit.Chem import AllChem, rdchem
from subprocess import CalledProcessError, run, DEVNULL

def prepare_protein(
    protein_read_path: str, 
    protein_save_path: str,
    replace_nonstandard_residues: bool = True, 
    remove_heterogens: bool = True, 
    remove_water: bool = True, 
    add_hydrogens: bool = True, 
    pH: float = 7.4):
    
    try:
        if protein_read_path.endswith('.pdb'):
            fixer = PDBFixer(filename=protein_read_path)
        else:
            fixer = PDBFixer(url='https://files.rcsb.org/download/%s.pdb' % protein_read_path)

        if replace_nonstandard_residues:
            fixer.findMissingResidues()
            fixer.findNonstandardResidues()
            fixer.replaceNonstandardResidues()
        if remove_heterogens:
            fixer.removeHeterogens(keepWater=not remove_water)
        if add_hydrogens:
            fixer.addMissingHydrogens(pH)
        
        with open(protein_save_path, 'w') as file:
            PDBFile.writeFile(fixer.topology, fixer.positions, file)

        p = Chem.MolFromPDBFile(protein_save_path, sanitize=True)
            
        return p
    
    except Exception as e:
        logging.info('Failed to prepare the molecule with name: ' + protein_read_path)
        logging.info("An exception of type {} occurred.".format(type(e).__name__))
        logging.info(e)        
        return None
    finally:
        collect()

def prepare_ligand(
    ligand: str, 
    optimize_ligand: bool = True):
    
    try:
        if ligand.endswith('.pdb'):
            m = Chem.MolFromPDBFile(ligand, sanitize=True)
        else:
            m = Chem.MolFromSmiles(ligand, sanitize=True)

        if optimize_ligand:
            m = Chem.AddHs(m)
            AllChem.EmbedMolecule(m)
            AllChem.MMFFOptimizeMolecule(m)
        
        return m
    
    except Exception as e:
        logging.info('Failed to prepare the ligand with name: ' + ligand)
        logging.info("An exception of type {} occurred.".format(type(e).__name__))
        logging.info(e)
        return None
    finally:
        collect()

def convert_pdb_to_pdbqt(input_path, output_path, is_ligand=True):    
    try:
        if is_ligand:
            logging.info('Converting ligand using OpenBabel.')
            run(['obabel', '-ipdb', input_path, '-osdf', '-O', output_path], check=True)
        else:
            logging.info('Converting protein using OpenBabel.')
            run(['obabel', '-ipdb', input_path, '-opdbqt', '-O', output_path], check=True)
    except CalledProcessError as e:
        logging.info("Failed to convert file:", e)
        