def train_model(dataframe):
    # Split your dataset
    train_df, test_df = train_test_split(dataframe, test_size=0.2, random_state=42)
    
    # Convert DataFrame to Dataset
    train_dataset = DockingDataset(
        ligand_embs=torch.tensor(train_df['ligand_emb'].tolist(), dtype=torch.float),
        protein_embs=torch.stack(train_df['protein_emb'].tolist()),  # Assuming these are already tensors
        scores=torch.tensor(train_df['scores'].tolist(), dtype=torch.long)
    )
    
    test_dataset = DockingDataset(
        ligand_embs=torch.tensor(test_df['ligand_emb'].tolist(), dtype=torch.float),
        protein_embs=torch.stack(test_df['protein_emb'].tolist()),  # Assuming these are already tensors
        scores=torch.tensor(test_df['scores'].tolist(), dtype=torch.long)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize the model
    model = DockingModel()
    
    # Initialize PyTorch Lightning trainer
    trainer = pl.Trainer(max_epochs=10, gpus=1)  # Adjust as per your setup
    trainer.fit(model, train_loader)
    