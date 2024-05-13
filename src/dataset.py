import os
import pandas as pd
import pickle
import numpy as np

from torch import load

from torch.utils.data import Dataset, DataLoader

class DockingDataset(Dataset):
    def __init__(self, ligand_data, protein_emb_paths, scores):
        """
        Initializes the dataset with ligand embeddings as a PyTorch tensor and names,
        paths to the protein embeddings, and the corresponding scores.
        :param ligand_data: A dictionary containing 'combined_tensor' with all ligand embeddings
                            and 'names' as a list of names corresponding to these embeddings.
        :param protein_emb_paths: A list of paths to the protein embeddings files.
        :param scores: A list of scores corresponding to each ligand-protein pair.
        """
        self.ligand_embeddings = ligand_data['combined_tensor']
        self.ligand_names      = ligand_data['names']
        self.protein_emb_paths = protein_emb_paths
        self.scores = scores

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):
        # Retrieve the ligand embedding directly from the tensor using the index
        ligand_emb = self.ligand_embeddings[idx]

        # Load protein embedding from disk
        protein_emb = load(self.protein_emb_paths[idx])
        
        # Ensure that your embeddings are loaded as PyTorch tensors
        # If not, you would need to convert them here

        return {
            "ligand_emb": ligand_emb,
            "protein_emb": protein_emb,
            "score": self.scores[idx]
        }

def get_list_of_proteins(proteins_folder_path):
    
    proteins = os.listdir(proteins_folder_path)
    proteins = [p[:-4] for p in proteins if p.endswith('.pdb')]
    return proteins
    
def build_dataset_from_docking_scores_folder(proteins, docking_scores_folder_path):
                                             
    scores_dicts = []
    
    for protein in proteins:
    
        merged_dict = merge_pickled_dictionaries(docking_scores_folder_path, protein)
        scores_dicts.append(merged_dict)
        print('Number of scores: ' + str(len(merged_dict)))
        print('Read docking scores for protein: ' + protein)

    dfs = []
    
    for dic in scores_dicts:
    
        if len(dic) == 0:
            print('Empty')
            dfs.append(pd.DataFrame())
        else:
            try:
                df = pd.DataFrame.from_dict(dic)
                df = df.T
                df.columns = ['score1']
                df = df[df.score1.apply(lambda x: isinstance(x, float))]
                df.score1 = pd.to_numeric(df.score1)
                dfs.append(df)
                print('Done converting to df')
            except Exception as e: # Usually happens if a dict only contains errors which are strings
                print(e)
                print(dic)
                print('Error converting dic to df')
                dfs.append(pd.DataFrame())
            
    for i in range(len(proteins)):
        dfs[i]['protein'] = proteins[i]

    dataset_df = pd.concat(dfs)
    dataset_df = dataset_df.reset_index()
    dataset_df = dataset_df.rename(columns={'index':'ligand'})
    
    mean_score = dataset_df.score1.mean()
    std_score  = dataset_df.score1.std()

    dataset_df = dataset_df.sample(frac = 1)
    dataset_df['score_category'] = dataset_df['score1'].apply(lambda x: categorize_score_based_on_sigma(x, mean_score, std_score))

    return dataset_df

def turn_protein_column_to_path_columns(df):

    df['protein_paths'] = 'proteins/embeddings/' + df.protein + '_embedding.pt'
    print('Done adding paths...')
    return df

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