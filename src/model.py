from torch import cat, optim, nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

class DockingModel(pl.LightningModule):
    def __init__(self, ligand_input_dim=768, protein_input_dim=1280, hidden_dim=256):
        super().__init__()
        # Ligand sub-model: a simple feedforward network
        self.ligand_model = nn.Sequential(
            nn.Linear(ligand_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # Protein sub-model: LSTM
        self.lstm = nn.LSTM(input_size=protein_input_dim, hidden_size=hidden_dim, batch_first=True)
        self.protein_fc = nn.Linear(hidden_dim, 256)  # To match ligand model output size
        
        # Final regression layer
        self.regressor = nn.Linear(512, 1)  # Combining both embeddings and outputting one value for regression
    
    def forward(self, ligand_emb, protein_emb):
        # Ligand model forward pass
        ligand_features = self.ligand_model(ligand_emb)
        
        # LSTM forward pass for proteins
        packed_output, (hidden, cell) = self.lstm(protein_emb)
        # We use the last hidden state to represent the entire sequence
        protein_features = self.protein_fc(hidden[-1])
        
        # Combine features and regress
        combined_features = cat((ligand_features, protein_features), dim=1)
        output = self.regressor(combined_features)
        return output
    
    def training_step(self, batch, batch_idx):
        ligand_emb, protein_emb, score = batch['ligand_emb'], batch['protein_emb'], batch['score']
        score_pred = self(ligand_emb, protein_emb)
        loss = nn.functional.mse_loss(score_pred.squeeze(), score)  # Make sure dimensions align, and use MSE for regression
        self.log('train_loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer