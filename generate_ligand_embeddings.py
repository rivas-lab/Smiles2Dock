from argparse import Namespace, ArgumentParser
from os import path, cpu_count, path, remove, listdir, makedirs
from time import time

from logging                       import getLogger, basicConfig, info, ERROR, INFO

getLogger("deepchem").setLevel(ERROR)
basicConfig(format='%(asctime)s - %(levelname)s - %(message)s\n', level=INFO)

from pandas import read_table 
from yaml import safe_load
from molformer.train_pubchem_light import LightningModule

from src.molformer import MolTranBertTokenizer, process_and_save

def parse_args():
    parser = ArgumentParser(description="Your script description here")
    parser.add_argument("ligand_dir", type=str, help="Directory path containing ligands")
    return parser.parse_args()

start_time = time()

info('Number of CPU cores available: ' + str(cpu_count()))

args = parse_args()

directory_path = args.ligand_dir

with open('./molformer/hparams.yaml', 'r') as f:
    config = Namespace(**safe_load(f))

tokenizer = MolTranBertTokenizer('./molformer/bert_vocab.txt')

ckpt = './molformer/checkpoints/N-Step-Checkpoint_3_30000.ckpt'
lm = LightningModule.load_from_checkpoint(ckpt, config=config, vocab=tokenizer.vocab)

txt_files = [f for f in listdir(directory_path) if f.endswith('.txt') and f.startswith('chembl')]
info(txt_files)

try:
    file_path = path.join(directory_path, txt_files[0])
    info('File path:')
    info(file_path)
except Exception as e:
    info(e)
    info('Error reading the data')
    exit(1)

chembl = read_table(file_path, sep='\t')
process_and_save(chembl, lm, tokenizer, batch_size=64)

end_time = time()
duration = end_time - start_time

info(f"Code execution took {duration} seconds.")