from logging import getLogger, basicConfig, info, ERROR, INFO
from torch import save, clamp, sum, no_grad, cat
from regex import compile
from fast_transformers.masking import LengthMask as LM
from rdkit import Chem
from transformers import BertTokenizer
from time import time
from os import path

getLogger("deepchem").setLevel(ERROR)
basicConfig(format='%(asctime)s - %(levelname)s - %(message)s\n', level=INFO)

PATTERN = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"

class MolTranBertTokenizer(BertTokenizer):
    def __init__(self, vocab_file: str = '',
                 do_lower_case=False,
                 unk_token='<pad>',
                 sep_token='<eos>',
                 pad_token='<pad>',
                 cls_token='<bos>',
                 mask_token='<mask>',
                 **kwargs):
        super().__init__(vocab_file,
                         unk_token=unk_token,
                         sep_token=sep_token,
                         pad_token=pad_token,
                         cls_token=cls_token,
                         mask_token=mask_token,
                         **kwargs)

        self.regex_tokenizer = compile(PATTERN)
        self.wordpiece_tokenizer = None
        self.basic_tokenizer = None

    def _tokenize(self, text):
        split_tokens = self.regex_tokenizer.findall(text)
        return split_tokens

    def convert_tokens_to_string(self, tokens):
        out_string = "".join(tokens).strip()
        return out_string

def process_and_save(chembl_df, model, tokenizer, batch_size=64):
    for batch in batch_split(chembl_df, batch_size=batch_size):
        chembl_ids  = batch['chembl_id'].tolist()
        smiles_list = batch['canonical_smiles'].apply(canonicalize).tolist()
        
        embeddings = embed(model, smiles_list, tokenizer, batch_size=len(smiles_list))

        for chembl_id, embedding in zip(chembl_ids, embeddings):
            save(embedding, path.join('ligands', f"{chembl_id}.pt"))

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
        start_time = time()  # Start timing here
        
        batch_enc = tokenizer.batch_encode_plus(batch, padding=True, add_special_tokens=True, return_tensors="pt")
        idx, mask = batch_enc['input_ids'], batch_enc['attention_mask']
        with no_grad():
            token_embeddings = model.blocks(model.tok_emb(idx), length_mask=LM(mask.sum(-1)))
            
        input_mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        sum_embeddings = sum(token_embeddings * input_mask_expanded, 1)
        sum_mask       = clamp(input_mask_expanded.sum(1), min=1e-9)
        
        embedding = sum_embeddings / sum_mask
        embeddings.append(embedding.detach().cpu())

        end_time = time()  # End timing here
        info(f"Processed batch in {end_time - start_time:.2f} seconds.")  

    return cat(embeddings)