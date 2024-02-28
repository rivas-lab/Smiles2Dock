from torch import cat, optim, nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

class DockingModel(pl.LightningModule):
    def __init__(self, ligand_input_dim=768, protein_input_dim=1280, hidden_dim=256, num_classes=6):
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
        
        # Final classifier
        self.classifier = nn.Linear(512, num_classes)  # Combining both embeddings
    
    def forward(self, ligand_emb, protein_emb):
        # Ligand model forward pass
        ligand_features = self.ligand_model(ligand_emb)
        
        # LSTM forward pass for proteins
        # Assuming protein_emb is packed with pack_padded_sequence for variable lengths
        packed_output, (hidden, cell) = self.lstm(protein_emb)
        # We use the last hidden state to represent the entire sequence
        protein_features = self.protein_fc(hidden[-1])
        
        # Combine features and classify
        combined_features = cat((ligand_features, protein_features), dim=1)
        scores = self.classifier(combined_features)
        return scores
    
    def training_step(self, batch, batch_idx):
        ligand_emb, protein_emb, scores = batch['ligand_emb'], batch['protein_emb'], batch['score']
        # Prepare protein_emb for LSTM processing if not already packed
        scores_pred = self(ligand_emb, protein_emb)
        loss = nn.functional.cross_entropy(scores_pred, scores)
        self.log('train_loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer