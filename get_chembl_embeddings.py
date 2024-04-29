from argparse import Namespace
from os import path
from time import time
from fast_transformers.masking import LengthMask as LM
from rdkit import Chem
from transformers import BertTokenizer

import pandas as pd
import regex as re
import torch
import yaml

from molformer.train_pubchem_light import LightningModule

with open('./molformer/hparams.yaml', 'r') as f:
    config = Namespace(**yaml.safe_load(f))

tokenizer = MolTranBertTokenizer('./molformer/bert_vocab.txt')

ckpt = './molformer/checkpoints/N-Step-Checkpoint_3_30000.ckpt'
lm = LightningModule.load_from_checkpoint(ckpt, config=config, vocab=tokenizer.vocab)

def batch_split(data, batch_size=64):
    i = 0
    while i < len(data):
        yield data[i:min(i+batch_size, len(data))]
        i += batch_size

def embed(model, smiles, tokenizer, batch_size=64):
    model.eval()
    embeddings = []
    for batch in batch_split(smiles, batch_size=batch_size):
        batch_enc = tokenizer.batch_encode_plus(batch, padding=True, add_special_tokens=True)
        idx, mask = torch.tensor(batch_enc['input_ids']), torch.tensor(batch_enc['attention_mask'])
        with torch.no_grad():
            token_embeddings = model.blocks(model.tok_emb(idx), length_mask=LM(mask.sum(-1)))
        # average pooling over tokens
        input_mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embedding = sum_embeddings / sum_mask
        embeddings.append(embedding.detach().cpu())
    return torch.cat(embeddings)

filepath        = 'data/chembl_33_chemreps.txt'  
data = pd.read_table(filepath, sep='\t')

smiles = data.canonical_smiles.tolist()

def canonicalize(s):
    return Chem.MolToSmiles(Chem.MolFromSmiles(s), canonical=True, isomericSmiles=False)

smiles = df.canonical_smiles.apply(canonicalize)
X = embed(lm, smiles, tokenizer).numpy()
print('Embedded for a small batch...')
def batch_split(data, batch_size=64):
    """
    Generator to yield batches of data.
    """
    i = 0
    while i < len(data):
        yield data[i:min(i+batch_size, len(data))]
        i += batch_size

def canonicalize(s):
    """
    Canonicalize SMILES strings.
    """
    return Chem.MolToSmiles(Chem.MolFromSmiles(s), canonical=True, isomericSmiles=False)

def embed(model, smiles, tokenizer, batch_size=64):
    model.eval()
    embeddings = []
    for batch in batch_split(smiles, batch_size=batch_size):
        start_time = time() 
        
        batch_enc = tokenizer.batch_encode_plus(batch, padding=True, add_special_tokens=True, return_tensors="pt")
        idx, mask = batch_enc['input_ids'], batch_enc['attention_mask']
        with torch.no_grad():
            token_embeddings = model.blocks(model.tok_emb(idx), length_mask=LM(mask.sum(-1)))
        embeddings.append(token_embeddings.detach().cpu())

        end_time = time() 
        print(f"Processed batch in {end_time - start_time:.2f} seconds.") 

    return torch.cat(embeddings)

def process_and_save(chembl_df, model, tokenizer, batch_size=64):
    for batch in batch_split(chembl_df, batch_size=batch_size):
        chembl_ids  = batch['chembl_id'].tolist()
        smiles_list = batch['canonical_smiles'].apply(canonicalize).tolist()
        embeddings = embed(model, smiles_list, tokenizer, batch_size=len(smiles_list))

        for chembl_id, embedding in zip(chembl_ids, embeddings):
            torch.save(embedding, path.join('ligands', f"{chembl_id}.pt"))

        return embeddings

embs = process_and_save(data, lm, tokenizer, batch_size=64)


















