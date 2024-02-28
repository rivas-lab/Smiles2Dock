import os
import pandas as pd
import pickle
import numpy as np

from torch import load

from torch.utils.data import Dataset, DataLoader

class DockingDataset(Dataset):
    def __init__(self, ligand_embs, protein_embs, scores):
        self.ligand_embs  = ligand_embs
        self.protein_embs = protein_embs
        self.scores       = scores

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):
        return {
            "ligand_emb":  self.ligand_embs[idx],
            "protein_emb": self.protein_embs[idx],
            "score":       self.scores[idx]w}

def get_list_of_proteins(proteins_folder_path):
    
    proteins = os.listdir(proteins_folder_path)
    proteins = [p[:-4] for p in proteins if p.endswith('.pdb')]
    return proteins
    
def build_dataset_from_docking_scores_folder(proteins, docking_scores_folder_path

    proteins = os.listdir(proteins_folder_path)
    proteins = [p[:-4] for p in proteins if p.endswith('.pdb')]

    scores_dicts = []
    
    for protein in proteins:
    
        merged_dict = merge_pickled_dictionaries(docking_scores_folder_path, protein)
        scores_dicts.append(merged_dict)
        print('Read docking scores for protein: ' + protein)

    dfs = []
    
    for dict in dicts:
    
        if len(dict) == 0:
            print('Empty')
            dfs.append(pd.DataFrame())
        else:
            df = pd.DataFrame.from_dict(dict)
            df = df.T
            df.columns = ['score1']
            df = df[df.score1.apply(lambda x: isinstance(x, float))]
            df.score1 = pd.to_numeric(df.score1)
            dfs.append(df)
            print('Done converting to df')
            
    for i in range(len(proteins)):
        dfs[i]['protein'] = proteins[i]

    dataset_df = pd.concat(dfs)

    dataset_df = dataset_df.set_index('ligands')
    dataset_df = dataset_df.reset_index()

    dataset_df = dataset_df.sample(frac = 1)
    dataset_df['score_category'] = dataset_df['score1'].apply(lambda x: categorize_score_based_on_sigma(x, mean_score, sigma))

    return dataset

def get_protein_tensors_list(dataset_df, proteins):

    tensors = {}
    
    for protein in proteins:
        tensors[protein] = load('proteins/embeddings/' + protein + '_embedding.pt')

    proteins_list = dataset_df.protein.tolist()
    proteins_tensors = [tensors[protein] for protein in proteins_list]

    return proteins_tensors

def get_ligands_tensors_list(dataset_df):

    tensors = {}
    
    ligand_tensors = []
    ligands        = dataset_df.ligands.tolist()
    
    for ligand in ligands:
        tensors[protein] = load('proteins/embeddings/' + protein + '_embedding.pt')

    proteins = dataset_df.protein.tolist()
    protein_tensors = [tensors[protein] for protein in proteins]

    return protein_tensors
    

def categorize_score_based_on_sigma(score, mean, sigma):
    """
    Categorizes the score with 'Medium' split into 'Medium+' and 'Medium-',
    and 'Very Strong' and 'Very Weak' for scores beyond ±2 sigma of the mean.
    
    Parameters:
    - score: The docking score to categorize.
    - mean: Mean of the docking scores.
    - sigma: Standard deviation of the docking scores.
    
    Returns:
    - Category of the score based on sigma.
    """
    if score <= mean - 2*sigma:
        return 'Very Strong'
    elif score <= mean - sigma:
        return 'Strong'
    elif score < mean:
        return 'Medium+'
    elif score < mean + sigma:
        return 'Medium-'
    elif score < mean + 2*sigma:
        return 'Weak'
    else:
        return 'Very Weak'

def merge_pickled_dictionaries(main_folder_path, protein):
    i = 0
    merged_dict = {}
    for root, dirs, files in os.walk(main_folder_path):
        # Check if the current directory has a subdirectory with the protein name
        if protein in dirs:
            # Construct the path to the protein-specific subsubfolder
            protein_folder_path = os.path.join(root, protein)
            # Look for the 'scores_dict.pkl' file specifically in the protein folder
            for root_protein, dirs_protein, files_protein in os.walk(protein_folder_path):
                if 'scores_dict.pkl' in files_protein:
                    try:
                        file_path = os.path.join(root_protein, 'scores_dict.pkl')
                        with open(file_path, 'rb') as f:
                            data = pickle.load(f)
                        if isinstance(data, dict):
                            merged_dict.update(data)
                        else:
                            pass
                    except Exception as e:
                        print(f"Error: {e}")
                        i += 1
                        
    print(f"Number of errors: {i}")
    print()
    return merged_dict