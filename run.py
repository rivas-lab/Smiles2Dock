import time
import logging
logging.getLogger("deepchem").setLevel(logging.ERROR)  
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s\n', level=logging.INFO)

start_time = time.time()

from pandas                         import read_table
from pickle                         import dump
from gc                             import collect
from os                             import listdir, path, remove
from sys                            import argv, exit
from rdkit                          import Chem

from deepchem.dock.pose_generation  import VinaPoseGenerator

from src.p2rank_pocket_finder       import P2RankPocketFinder
from src.preparation                import prepare_protein, prepare_ligand, convert_pdb_to_pdbqt
from src.utils                      import print_block

if len(argv) > 1:
    directory_path = argv[1] 
    print_block()
    logging.info("Directory path:")
    logging.info(directory_path)
else:
    logging.info("No file path provided. Please provide a file path as a command line argument.")
    exit(1)  

txt_files = [f for f in listdir(directory_path) if f.endswith('.txt')]

if txt_files:
    file_path = path.join(directory_path, txt_files[0])
    logging.info('File path:')
    logging.info(file_path)
    save_path = path.join(directory_path, 'scores_dict.pkl') 
    logging.info('Save path:')
    logging.info(save_path)
else:
    logging.info('Error reading the data')

chembl  = read_table(file_path, sep='\t')
ligands = chembl['canonical_smiles'].tolist()
ids     = chembl['chembl_id'].tolist()

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

output_dic    = {}   
num_modes     = 10
save_interval = 5  

for i in range(len(ligands)):
    
    print_block()
    
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
            
    if m is None:
        logging.info(f'Failed to prepare the molecule {ligand}')
        logging.info('Going to the next iteration')
        output_dic[ids[i]] = 'Error 2 converting ligand'
        continue
        
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
                out_dir = directory_path, generate_scores = True, num_modes = num_modes, 
                cpu = 1, seed = 123)
            
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
        
        # Save the scores every x iteration
        if (i + 1) % save_interval == 0 or (i + 1) == len(ligands):
            with open(save_path, 'wb') as file:
                dump(output_dic, file)
            logging.info(f"Results saved after processing {i + 1} ligands.")
    
end_time = time.time()
duration = end_time - start_time

logging.info(f"Code execution took {duration} seconds.")
