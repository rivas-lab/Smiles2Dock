from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

import argparse
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pandas as pd

from src.dataset import build_dataset_from_docking_scores_folder, get_list_of_proteins, turn_protein_column_to_path_columns, DockingDataset
from src.model import DockingModel, custom_collate_fn

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import os
from pandas import read_csv
from torch.utils.data import DataLoader
from torch.cuda import device_count, get_device_name, is_available
from torch import load
from datasets import load_dataset

def main(protein_model_dim, hidden_dim, ligand_model_dim, dropout_rate):
    if is_available():
        num_gpus = device_count()
        print("Number of available GPUs:", num_gpus)
        gpu_name = get_device_name(0)
        print("GPU Name:", gpu_name)
    else:
        print("CUDA is not available. Using CPU.")

    checkpoint_filename = f'protein{protein_model_dim}_hidden{hidden_dim}_ligand{ligand_model_dim}_dropout{dropout_rate}_epochepoch=01.ckpt'
    checkpoint_path = os.path.join('checkpoints', checkpoint_filename)

    model = DockingModel.load_from_checkpoint(
        checkpoint_path,
        dropout_rate=dropout_rate,
        ligand_model_dim=ligand_model_dim,
        protein_model_dim=protein_model_dim,
        hidden_dim=hidden_dim
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    dataset = load_dataset('tlemenestrel/Smiles2Dock')
    df = pd.DataFrame(dataset['test'])
    print(len(df))
    ligand_tensor = load('tensors/final_tensor.pt')
    ligand_tensor['names'] = [name[:-3] for name in ligand_tensor['names']]

    df = turn_protein_column_to_path_columns(df)
    print('Added paths to dfs')
    print(df.columns)
    dataset = DockingDataset(ligand_tensor, df.protein_paths.tolist(), df.ligand.tolist(), df.score1.tolist())
    data_loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=custom_collate_fn, num_workers=3, pin_memory=True)

    print('Starting inference...')
    
    model.eval()  # Set the model to evaluation mode
    predictions = []
    original_data = []

    with torch.no_grad():  # Disable gradient calculations
        for batch in tqdm(data_loader, desc="Processing batches"):
            # Move each tensor in the batch dictionary to the device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            ligand_emb, protein_emb = batch['ligand_emb'], batch['protein_emb']
            outputs = model(ligand_emb, protein_emb)
            predictions.extend(outputs.cpu().numpy())  # Move the outputs to CPU and convert to numpy format
            print(batch.keys())
            original_data.extend(batch['scores'].cpu().numpy())

    df_results = pd.DataFrame({'Original': original_data, 'Prediction': predictions})
    print(df_results.head())
    output_path = "/share/pi/mrivas/tlmenest/" + checkpoint_filename + "_predictions.csv"
    df_results.to_csv(output_path, index=False)
    print(f'Saved predictions to {output_path}')
    df_results['Prediction'] = df_results['Prediction'].apply(lambda x: x[0])

    rmse = np.sqrt(mean_squared_error(df_results['Original'], df_results['Prediction']))
    r2   = r2_score(df_results['Original'], df_results['Prediction'])

    print(f'RMSE: {rmse}')
    print(f'R2 Score: {r2}')

    metrics_path = "/share/pi/mrivas/tlmenest/" + checkpoint_filename + "_metrics.txt"
    with open(metrics_path, 'w') as f:
        f.write(f'RMSE: {rmse}\n')
        f.write(f'R2 Score: {r2}\n')

    print(f'Saved metrics to {metrics_path}')
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--protein_model_dim', type=int, required=True)
    parser.add_argument('--hidden_dim', type=int, required=True)
    parser.add_argument('--ligand_model_dim', type=int, required=True)
    parser.add_argument('--dropout_rate', type=float, required=True)
    args = parser.parse_args()
    main(args.protein_model_dim, args.hidden_dim, args.ligand_model_dim, args.dropout_rate)
  
