from argparse                      import ArgumentParser
from deepchem.dock.pose_generation import VinaPoseGenerator
from gc                            import collect
from logging                       import getLogger, basicConfig, info, ERROR, INFO
from multiprocessing               import cpu_count
from os                            import path, remove, listdir, makedirs
from pickle                        import dump, load
from pandas                        import read_table
from psutil                        import virtual_memory
from rdkit                         import Chem
from sys                           import argv, exit
from time                          import time

from src.p2rank_pocket_finder      import P2RankPocketFinder
from src.preparation               import prepare_protein, prepare_ligand, convert_pdb_to_pdbqt
from src.utils                     import print_block

getLogger("deepchem").setLevel(ERROR)
basicConfig(format='%(asctime)s - %(levelname)s - %(message)s\n', level=INFO)

def parse_args():
    parser = ArgumentParser(description="Your script description here")
    parser.add_argument("ligand_dir", type=str, help="Directory path containing ligands")
    parser.add_argument("protein_name", type=str, help="Name of the protein")
    return parser.parse_args()

start_time = time()

info('Number of CPU cores available: ' + str(cpu_count()))

args = parse_args()

directory_path = args.ligand_dir
protein_name   = args.protein_name

protein_specific_dir = path.join(directory_path, protein_name)
if not path.exists(protein_specific_dir):
    makedirs(protein_specific_dir)
    info('Directory for storing scores does not exist, creating it.')
else:
    info('Directory for storing scores already exists.')
    
print_block()
info("Directory path:")
info(directory_path)
info("Protein name:")
info(protein_name)

# Identify the .txt file with CHEMBL IDs
txt_files = [f for f in listdir(directory_path) if f.endswith('.txt') and f.startswith('chembl')]
info(txt_files)

try:
    file_path = path.join(directory_path, txt_files[0])
    info('File path:')
    info(file_path)
    save_path = path.join(directory_path, protein_name, 'scores_dict.pkl')
    info('Save path:')
    info(save_path)
except Exception as e:
    info(e)
    info('Error reading the data')
    exit(1)

# Read CHEMBL IDs and canonical SMILES from the .txt file
chembl  = read_table(file_path, sep='\t')
ids     = chembl['chembl_id'].tolist()
ligands = chembl['canonical_smiles'].tolist()

# Check if the output dictionary already exists
if path.exists(save_path):
    info('Processing started already, restarting from checkpoint.')
    with open(save_path, 'rb') as file:
        output_dic = load(file)
    # Exclude already processed CHEMBL IDs
    processed_ids = list(output_dic.keys())
    info('Already processed before ' + str(len(processed_ids)) + ' ligands.')
    ids = [id_ for id_ in ids if id_ not in processed_ids]
    ligands = [ligands[i] for i, id_ in enumerate(chembl['chembl_id']) if id_ not in processed_ids]
    info('Len of ligands: ' + str(len(ligands)))
else:
    info('No processing done before, initializing new scores.')
    output_dic = {}

assert len(ids) == len(ligands), "The length of ids does not match the length of ligands."

try:
    print_block()
    info('Preparing protein.')
    
    protein_read_path = path.join('./proteins', protein_name + '.pdb')
    protein_save_path = path.join(directory_path, protein_name, protein_name + '.pdb')

    info(protein_read_path)
    info(protein_save_path)

    p = prepare_protein(protein_read_path, protein_save_path)

    Chem.rdmolfiles.MolToPDBFile(p, protein_save_path)
    #convert_pdb_to_pdbqt(protein_pdb_path, protein_pdb_path, is_ligand=False)
    info('Finished preparing protein.')
except Exception as e:
    p = None
    info('Error when converting protein to pdbqt.')
    info(e)
    info("An exception of type {} occurred.".format(type(e).__name__))

assert p is not None, 'Preparing the protein failed.'

# PARAMETERS
num_modes      = 2
save_interval  = 5
exhaustiveness = 2
cpu            = 3
padding        = 10.0
threshold      = 0.5
num_pockets    = 1

initial_mem = virtual_memory()
info(f"Initial free memory: {initial_mem.free / (1024**3):.2f} GB")

for i in range(len(ligands)):

    print_block()
    
    info('Measuring time for open Babel Ligand')
    start_time_ligand = time()

    ligand = ligands[i]
    ligand_pdb_path = path.join(directory_path, protein_name, ids[i] + '.pdb')
    ligand_sdf_path = path.join(directory_path, protein_name, ids[i] + '.sdf')
    
    try:
        info('Preparing the ligand ...')
        m = prepare_ligand(ligand)
        Chem.rdmolfiles.MolToPDBFile(m, ligand_pdb_path)
        convert_pdb_to_pdbqt(ligand_pdb_path, ligand_sdf_path, is_ligand=True)
        info('Finished preparing ligand.')
    except Exception as e:
        info('Error when converting ligand to pdbqt.')
        info(e)
        info("An exception of type {} occurred.".format(type(e).__name__))
        output_dic[ids[i]] = 'Error 1 converting ligand'
        continue
    finally:
        end_time_ligand = time()
        duration_ligand = end_time_ligand - start_time_ligand
        info(f"Ligand took {duration_ligand} seconds.")
            
    if m is None:
        info(f'Failed to prepare the molecule {ligand}')
        info('Going to the next iteration')
        output_dic[ids[i]] = 'Error 2 converting ligand'
        continue
    
    info('Measuring time for docking one ligand')
    start_time_docking = time()
    
    if p is not None and m is not None:  
        try:
            
            info("Docking ligand "+ str(i + 1))

            pocket_predictions_csv_path = (
                './p2rank_2.4.1/test_output/predict_' + protein_name + '/' + protein_name + '.pdb_predictions.csv')

            info(pocket_predictions_csv_path)
            
            pocket_finder = P2RankPocketFinder(
                pocket_predictions_csv_path, ligand_mol = m, threshold = threshold, padding = padding)

            vpg = VinaPoseGenerator(pocket_finder = pocket_finder)
            
            info('Running docking...')
            info(protein_save_path)
            info(ligand_sdf_path)
            complexes, scores = vpg.generate_poses(
                molecular_complex=(protein_save_path, ligand_sdf_path), 
                out_dir = path.join(directory_path, protein_name), 
                generate_scores = True, num_modes = num_modes, cpu = cpu, 
                seed = 123, exhaustiveness = exhaustiveness, num_pockets=num_pockets)
            
            info('Finished docking.')
            
            # Break up the list of scores into list of list for each pocket
            final_scores = [scores[i:i + num_modes] for i in range(0, len(scores), num_modes)]
            smallest_scores_per_list = [min(sublist) for sublist in final_scores]

            info(smallest_scores_per_list)
            output_dic[ids[i]] = smallest_scores_per_list
           
        except TypeError as e:
            info('Error: file with atoms not correct for PDBQT format.')
            info(e)
            output_dic[ids[i]] = 'Error atoms not valid for PDBQT'
        except Exception as e:
            info('Error when trying to dock.')
            info(e)
            info("An exception of type {} occurred.".format(type(e).__name__))
            output_dic[ids[i]] = 'Error when docking'
        finally:
            out_pdbqt_docked = path.join(directory_path, protein_name, "%s_docked.pdbqt" % ids[i])
            out_pdbqt        = path.join(directory_path, protein_name, "%s.pdbqt"        % ids[i])

            info('Delete paths for pdbqt files:')
            info(out_pdbqt_docked)
            info(out_pdbqt)
            info(ligand_pdb_path)
            info(ligand_sdf_path)
            
            if path.exists(out_pdbqt_docked):
                remove(out_pdbqt_docked)
            if path.exists(out_pdbqt):
                remove(out_pdbqt)
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
            info(f"Docking took {duration_docking} seconds.")
            
            mem = virtual_memory()
            info(f"Free memory out of total: {mem.free / (1024**3):.2f} GB free out of {mem.total / (1024**3):.2f} GB total")
            
        if (i + 1) % save_interval == 0 or (i + 1) == len(ligands):
            with open(save_path, 'wb') as file:
                dump(output_dic, file)
            info(f"Results saved after processing {i + 1} ligands.")
            end_time = time()
            duration = end_time - start_time
            info(f"Processing time since the start is {duration} seconds.")
    
end_time = time()
duration = end_time - start_time

info(f"Code execution took {duration} seconds.")
