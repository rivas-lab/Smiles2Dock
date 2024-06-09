from torch import cat, optim, nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torch

import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F

LIGAND_INPUT_DIM  = 768
PROTEIN_INPUT_DIM = 1280

class DockingModel(pl.LightningModule):
    def __init__(self, dropout_rate=0.5, ligand_model_dim=512, protein_model_dim=768, hidden_dim=256):
        super().__init__()
        self.ligand_model = nn.Sequential(
            nn.Linear(LIGAND_INPUT_DIM, ligand_model_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ligand_model_dim, hidden_dim),
            nn.ReLU())
        
        self.lstm = nn.LSTM(input_size=PROTEIN_INPUT_DIM, hidden_size=protein_model_dim, batch_first=True)
        self.protein_fc = nn.Linear(protein_model_dim, hidden_dim)  # To match ligand model output size
        
        self.regressor = nn.Linear(hidden_dim * 2, 1)  # Combining both embeddings and outputting one value for regression
    
    def forward(self, ligand_emb, protein_emb):
        # Ligand model forward pass
        ligand_features = self.ligand_model(ligand_emb)
        
        # LSTM forward pass for proteins
        packed_output, (hidden, cell) = self.lstm(protein_emb)
        # We use the last hidden state to represent the entire sequence
        protein_features = self.protein_fc(hidden[-1])
        
        # Combine features and regress
        combined_features = torch.cat((ligand_features, protein_features), dim=1)
        output = self.regressor(combined_features)
        return output
    
    def training_step(self, batch, batch_idx):
        ligand_emb, protein_emb, score = batch['ligand_emb'], batch['protein_emb'], batch['scores']  # Use 'scores' to match the custom collate function
        score_pred = self(ligand_emb, protein_emb)
        loss = nn.functional.mse_loss(score_pred.squeeze(), score)  # Make sure dimensions align, and use MSE for regression
        self.log('train_loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

def custom_collate_fn(batch):
    # Assume batch is a list of dictionaries
    ligand_embs = [item['ligand_emb'] for item in batch]
    protein_embs = [item['protein_emb'] for item in batch]
    scores = [item['score'] for item in batch]

    # Stack ligand embeddings into a tensor (assuming they are of the same size)
    ligand_embs = torch.stack(ligand_embs)

    # Convert scores list to tensor
    scores = torch.tensor(scores)

    # Determine the maximum sequence length and feature size across all protein embeddings
    max_length = max(emb.shape[0] for emb in protein_embs)
    max_size = max(emb.shape[1] for emb in protein_embs)

    # Pad each tensor to the maximum sequence length and feature size
    padded_protein_embs = [
        F.pad(emb, (0, 0, 0, max_size - emb.shape[1], 0, max_length - emb.shape[0])) 
        for emb in protein_embs
    ]

    # Stack the padded tensors into a single tensor
    protein_embs_padded = torch.stack(padded_protein_embs, dim=0)

    # Ensure the shape is (batch_size, seq_len, feature_size)
    protein_embs_padded = protein_embs_padded.squeeze(dim=1) if protein_embs_padded.dim() == 4 else protein_embs_padded

    # Return a dictionary that matches the input structure
    return {
        'ligand_emb': ligand_embs, 
        'protein_emb': protein_embs_padded, 
        'scores': scores
    }    
