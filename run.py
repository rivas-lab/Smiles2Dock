import logging

logging.getLogger("deepchem").setLevel(logging.ERROR)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s\n', level=logging.INFO)

import os
from os import path, remove
import pandas as pd
from pickle import dump, load
from rdkit import Chem
from sys import argv, exit
from time import time
from gc import collect
from psutil import virtual_memory

from deepchem.dock.pose_generation import VinaPoseGenerator

from src.p2rank_pocket_finder import P2RankPocketFinder
from src.preparation import prepare_protein, prepare_ligand, convert_pdb_to_pdbqt
from src.utils import print_block

# Start timing
start_time = time()

# Check command line argument for the directory path
if len(argv) > 1:
    directory_path = argv[1]
    print_block()
    logging.info("Directory path:")
    logging.info(directory_path)
else:
    logging.info("No file path provided. Please provide a file path as a command line argument.")
    exit(1)

# Identify the .txt file with CHEMBL IDs
txt_files = [f for f in os.listdir(directory_path) if f.endswith('.txt') and f.startswith('chembl')]

if txt_files:
    file_path = os.path.join(directory_path, txt_files[0])
    logging.info('File path:')
    logging.info(file_path)
    save_path = os.path.join(directory_path, 'scores_dict.pkl')
    logging.info('Save path:')
    logging.info(save_path)
else:
    logging.info('Error reading the data')
    exit(1)

# Read CHEMBL IDs and canonical SMILES from the .txt file
chembl  = pd.read_table(file_path, sep='\t')
ids     = chembl['chembl_id'].tolist()
ligands = chembl['canonical_smiles'].tolist()

# Check if the output dictionary already exists
if os.path.exists(save_path):
    logging.info('Processing started already, restarting from checkpoint.')
    with open(save_path, 'rb') as file:
        output_dic = load(file)
    # Exclude already processed CHEMBL IDs
    processed_ids = list(output_dic.keys())
    logging.info('Already processed before ' + str(len(processed_ids)) + ' ligands.')
    ids = [id_ for id_ in ids if id_ not in processed_ids]
    ligands = [ligands[i] for i, id_ in enumerate(chembl['chembl_id']) if id_ not in processed_ids]
    logging.info('Len of ligands: ' + str(len(ligands)))
else:
    logging.info('No processing done before, initializing new scores.')
    output_dic = {}

assert len(ids) == len(ligands), "The length of ids does not match the length of ligands."

protein      = 'proteins/slc6a19.pdb'
protein_name = 'slc6a19'

try:
    print_block()
    logging.info('Preparing protein.')
    protein_pdb_path = directory_path + '/' + protein_name + '.pdb'
    p = prepare_protein(protein, protein_pdb_path)
    Chem.rdmolfiles.MolToPDBFile(p, protein_pdb_path)
    #convert_pdb_to_pdbqt(protein_pdb_path, protein_pdb_path, is_ligand=False)
    logging.info('Finished preparing protein.')
except Exception as e:
    logging.info('Error when converting protein to pdbqt.')
    logging.info(e)
    logging.info("An exception of type {} occurred.".format(type(e).__name__))

assert p is not None, 'Preparing the protein failed.'

# PARAMETERS
num_modes      = 5
save_interval  = 5  
exhaustiveness = 5
cpu            = 3

initial_mem = virtual_memory()
logging.info(f"Initial free memory: {initial_mem.free / (1024**3):.2f} GB")

for i in range(len(ligands)):

    print_block()
    
    logging.info('Measuring time for open Babel Ligand')
    start_time_ligand = time()

    ligand = ligands[i]
    ligand_pdb_path = directory_path + '/' + ids[i] + '.pdb'
    ligand_sdf_path = directory_path + '/' + ids[i] + '.sdf'
    
    try:
        logging.info('Preparing the ligand ...')
        m = prepare_ligand(ligand)
        ligand_pdb_path = directory_path + '/' + ids[i] + '.pdb'
        Chem.rdmolfiles.MolToPDBFile(m, ligand_pdb_path)
        convert_pdb_to_pdbqt(ligand_pdb_path, ligand_sdf_path, is_ligand=True)
        logging.info('Finished preparing ligand.')
    except Exception as e:
        logging.info('Error when converting ligand to pdbqt.')
        logging.info(e)
        logging.info("An exception of type {} occurred.".format(type(e).__name__))
        output_dic[ids[i]] = 'Error 1 converting ligand'
        continue
    finally:
        end_time_ligand = time()
        duration_ligand = end_time_ligand - start_time_ligand
        logging.info(f"Ligand took {duration_ligand} seconds.")
            
    if m is None:
        logging.info(f'Failed to prepare the molecule {ligand}')
        logging.info('Going to the next iteration')
        output_dic[ids[i]] = 'Error 2 converting ligand'
        continue
    
    logging.info('Measuring time for docking one ligand')
    start_time_docking = time()
    
    if p is not None and m is not None:  
        try:
            
            logging.info("Docking ligand "+ str(i + 1))
            pocket_finder = P2RankPocketFinder(
                'p2rank_2.4.1/test_output/predict_slc6a19/slc6a19.pdb_predictions.csv',
                ligand_mol = m, threshold = 0.3, padding = 10.0)

            vpg = VinaPoseGenerator(pocket_finder = pocket_finder)
            
            logging.info('Running docking...')
            
            complexes, scores = vpg.generate_poses(
                molecular_complex=(protein_pdb_path, ligand_sdf_path), 
                out_dir = directory_path, generate_scores = True, 
                num_modes = num_modes, cpu = cpu, seed = 123, 
                exhaustiveness = exhaustiveness)
            
            logging.info('Finished docking.')
            
            # Break up the list of scores into list of list for each pocket
            final_scores = [scores[i:i + num_modes] for i in range(0, len(scores), num_modes)]
            smallest_scores_per_list = [min(sublist) for sublist in final_scores]

            logging.info(smallest_scores_per_list)
            output_dic[ids[i]] = smallest_scores_per_list
           
        except TypeError as e:
            logging.info('Error: file with atoms not correct for PDBQT format.')
            logging.info(e)
            output_dic[ids[i]] = 'Error atoms not valid for PDBQT'
        except Exception as e:
            logging.info('Error when trying to dock.')
            logging.info(e)
            logging.info("An exception of type {} occurred.".format(type(e).__name__))
            output_dic[ids[i]] = 'Error when docking'
        finally:
            if path.exists(ligand_pdb_path):
                remove(ligand_pdb_path)
            if path.exists(ligand_sdf_path):
                remove(ligand_sdf_path)
            if 'pocket_finder' in locals():
                del pocket_finder
            if 'vpg' in locals():
                del vpg
            collect()
            end_time_docking = time()
            duration_docking = end_time_docking - start_time_docking
            logging.info(f"Docking took {duration_docking} seconds.")
            
            mem = virtual_memory()
            logging.info(f"Free memory out of total: {mem.free / (1024**3):.2f} GB free out of {mem.total / (1024**3):.2f} GB total")

        # Save the scores every x iteration
        if (i + 1) % save_interval == 0 or (i + 1) == len(ligands):
            with open(save_path, 'wb') as file:
                dump(output_dic, file)
            logging.info(f"Results saved after processing {i + 1} ligands.")
            end_time = time()
            duration = end_time - start_time
            logging.info(f"Processing time since the start is {duration} seconds.")
    
end_time = time()
duration = end_time - start_time

logging.info(f"Code execution took {duration} seconds.")
