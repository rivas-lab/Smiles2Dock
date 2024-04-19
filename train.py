print('Reading source libraries')

from src.dataset import build_dataset_from_docking_scores_folder, get_list_of_proteins, turn_protein_ligand_columns_to_path_columns, DockingDataset
from src.model   import DockingModel

print('Reading external libraries')

from pytorch_lightning import Trainer
from pandas import read_csv
from torch.utils.data import DataLoader
from torch.cuda import device_count, get_device_name, is_available
from torch import stack, tensor, squeeze
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F

if is_available():
    num_gpus = device_count()
    print("Number of available GPUs:", num_gpus)
    gpu_name = get_device_name(0)  
    print("GPU Name:", gpu_name)
else:
    print("CUDA is not available. Using CPU.")
    
print('Loading dataset...')
train_df = read_csv('./datasets/smiles2dock_train.csv')
print(train_df.columns)
val_df   = read_csv('./datasets/smiles2dock_val.csv')
print('Loaded dataset...')

train_df = turn_protein_ligand_columns_to_path_columns(train_df)
val_df   = turn_protein_ligand_columns_to_path_columns(val_df)

print('Added paths to dfs')

train_dataset = DockingDataset(
    train_df.ligand_paths.tolist(), 
    train_df.protein_paths.tolist(), 
    train_df.score1.tolist())

val_dataset = DockingDataset(
    val_df.ligand_paths.tolist(), 
    val_df.protein_paths.tolist(), 
    val_df.score1.tolist())

print('Turned dfs into datasets')

model = DockingModel()

def custom_collate_fn(batch):
    # Extracting ligand embeddings, protein embeddings, and scores from the batch
    ligand_embs = [item['ligand_emb'] for item in batch]
    protein_embs = [item['protein_emb'] for item in batch]
    scores = [item['score'] for item in batch]
    
    # Convert lists to tensors
    ligand_embs = stack(ligand_embs)  # Stacking works because all ligand_embs are of the same size
    scores      = tensor(scores)  # Convert scores list to tensor

    # Determine the maximum sequence length and feature size across all protein embeddings
    max_length = max(emb.shape[0] for emb in protein_embs)
    max_size = max(emb.shape[1] for emb in protein_embs)

    # Pad each tensor to the maximum sequence length and feature size
    padded_protein_embs = [F.pad(emb, (0, 0, 0, max_size - emb.shape[1], 0, max_length - emb.shape[0]))
                            if emb.shape[0] < max_length or emb.shape[1] < max_size else emb
                            for emb in protein_embs]
    # Stack the padded tensors into a single tensor
    protein_embs_padded = stack(padded_protein_embs, dim=0)

    # Remove length-1 dimension
    protein_embs_padded = squeeze(protein_embs_padded)
    
    # Return a dictionary that matches the input structure
    return {'ligand_emb': ligand_embs, 'protein_emb': protein_embs_padded, 'score': scores}
    
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, collate_fn=custom_collate_fn)
val_loader   = DataLoader(val_dataset, batch_size=64, shuffle=False)

print('Loaded data')

trainer = Trainer(max_epochs=3)  

print('Training')

trainer.fit(model, train_loader)