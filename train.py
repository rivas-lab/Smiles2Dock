from src.dataset import build_dataset_from_docking_scores_folder, get_list_of_proteins, turn_protein_ligand_columns_to_path_columns, DockingDataset
from src.model   import DockingModel

from pytorch_lightning import Trainer
import torch
from torch.utils.data import Subset, DataLoader, Dataset

if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print("Number of available GPUs:", num_gpus)
    gpu_name = torch.cuda.get_device_name(0)  
    print("GPU Name:", gpu_name)
else:
    print("CUDA is not available. Using CPU.")
    
proteins = get_list_of_proteins('./proteins')

print('Loading dataset...')
dataset = build_dataset_from_docking_scores_folder(proteins, './input')
print('Loaded dataset...')

df = turn_protein_ligand_columns_to_path_columns(dataset)

dock_d = DockingDataset(df.ligand_paths.tolist(), df.protein_paths.tolist(), df.score1.tolist())

model = DockingModel()

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

indices = torch.randperm(len(dataset)).tolist()

train_indices = indices[:train_size]
test_indices = indices[train_size:]

train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

trainer = Trainer(max_epochs=3)  # Adjust max_epochs & gpus according to your needs

trainer.fit(model, train_loader)