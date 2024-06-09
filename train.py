import argparse
from src.dataset import build_dataset_from_docking_scores_folder, get_list_of_proteins, turn_protein_column_to_path_columns, DockingDataset
from src.model import DockingModel, custom_collate_fn

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import os
from pandas import read_csv
from torch.utils.data import DataLoader
from torch.cuda import device_count, get_device_name, is_available
from torch import load

def compute_rmse(model, val_loader, device):
    model.eval()
    mse_loss = nn.MSELoss()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs, targets = batch 
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = mse_loss(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
    rmse = np.sqrt(total_loss / total_samples)
    return rmse

class RMSECallback(pl.Callback):
    def __init__(self, val_loader, device):
        self.val_loader = val_loader
        self.device = device

    def on_epoch_end(self, trainer, pl_module):
        rmse = compute_rmse(pl_module, self.val_loader, self.device)
        print(f'Validation rMSE: {rmse:.4f}')

def main(protein_model_dim, hidden_dim, ligand_model_dim, dropout_rate):
    if is_available():
        num_gpus = device_count()
        print("Number of available GPUs:", num_gpus)
        gpu_name = get_device_name(0)
        print("GPU Name:", gpu_name)
    else:
        print("CUDA is not available. Using CPU.")

    print('Loading dataset...')
    val_df = read_csv('./datasets/smiles2dock_val.csv')
    train_df = read_csv('./datasets/smiles2dock_train.csv')
    print(train_df.columns)
    print('Loaded dataset...')
    checkpoint_filename = f'protein{protein_model_dim}_hidden{hidden_dim}_ligand{ligand_model_dim}_dropout{dropout_rate}_epochepoch=00.ckpt'
    checkpoint_path = os.path.join('checkpoints', checkpoint_filename)

    ligand_tensor = load('tensors/final_tensor.pt')
    ligand_tensor['names'] = [name[:-3] for name in ligand_tensor['names']]

    train_df = turn_protein_column_to_path_columns(train_df)
    val_df = turn_protein_column_to_path_columns(val_df)

    print('Added paths to dfs')

    train_dataset = DockingDataset(ligand_tensor, train_df.protein_paths.tolist(), train_df.ligand.tolist(), train_df.score1.tolist())
    val_dataset = DockingDataset(ligand_tensor, val_df.protein_paths.tolist(), val_df.ligand.tolist(), val_df.score1.tolist())

    print('Turned dfs into datasets')

    model = DockingModel(protein_model_dim=protein_model_dim, hidden_dim=hidden_dim, ligand_model_dim=ligand_model_dim, dropout_rate=dropout_rate)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False, collate_fn=custom_collate_fn, num_workers=3, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=3, pin_memory=True)

    print('Loaded data')

    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename=f'protein{protein_model_dim}_hidden{hidden_dim}_ligand{ligand_model_dim}_dropout{dropout_rate}_epoch{{epoch:02d}}',
        save_top_k=-1,
        verbose=True,
        every_n_epochs=1
    )
    trainer = Trainer(
        max_epochs=3, 
        accumulate_grad_batches=4,
        devices=1,
        callbacks=[RMSECallback(val_loader, device)])

    trainer.fit(model, train_loader, ckpt_path=checkpoint_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DockingModel with given hyperparameters')
    parser.add_argument('--protein_model_dim', type=int, required=True, help='Dimension of the protein model')
    parser.add_argument('--hidden_dim', type=int, required=True, help='Dimension of the hidden layers')
    parser.add_argument('--ligand_model_dim', type=int, required=True, help='Dimension of the ligand model')
    parser.add_argument('--dropout_rate', type=float, required=True, help='Dropout rate for the model')

    args = parser.parse_args()

    main(args.protein_model_dim, args.hidden_dim, args.ligand_model_dim, args.dropout_rate)
