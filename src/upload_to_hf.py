import os
import shutil
from datasets import load_dataset, DatasetDict

# Load datasets from CSV files
train_dataset = load_dataset('csv', data_files='datasets/smiles2dock_train.csv')
test_dataset = load_dataset('csv', data_files='datasets/smiles2dock_test.csv')
val_dataset = load_dataset('csv', data_files='datasets/smiles2dock_val.csv')

# Create a DatasetDict
dataset_dict = DatasetDict({
    'train': train_dataset['train'],
    'test': test_dataset['train'],
    'val': val_dataset['train']
})

# Save the dataset locally in a temporary directory
temp_dir = "temp_smiles2dock_dataset"
if os.path.exists(temp_dir):
    shutil.rmtree(temp_dir)
os.makedirs(temp_dir)

dataset_dict.save_to_disk(temp_dir)

# Hugging Face token
hf_token = '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'

# Initialize the git repository manually
os.system(f"cd {temp_dir} && git init")
os.system(f"cd {temp_dir} && git remote add origin https://huggingface.co/datasets/tlemenestrel/Smiles2Dock")

# Set git user information
os.system(f"cd {temp_dir} && git config user.email 'tlmenest@stanford.edu' && git config user.name 'Thomas Le Menestrel'")

# Track large files with Git LFS
os.system(f"cd {temp_dir} && git lfs install")
os.system(f"cd {temp_dir} && git lfs track '*.arrow'")
os.system(f"cd {temp_dir} && git add .gitattributes")

# Add and commit files
os.system(f"cd {temp_dir} && git add . && git commit -m 'Initial commit of the dataset'")

# Set up credentials for pushing to Hugging Face
os.system(f"cd {temp_dir} && git remote set-url origin https://{hf_token}:x-oauth-basic@huggingface.co/datasets/tlemenestrel/Smiles2Dock")

# Push to Hugging Face Hub
os.system(f"cd {temp_dir} && git push origin master")

# Clean up temporary directory
shutil.rmtree(temp_dir)

print("Dataset uploaded successfully.")