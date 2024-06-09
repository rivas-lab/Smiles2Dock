# Smiles2Dock

![GitHub](https://img.shields.io/github/license/rivas-lab/Smiles2Dock)
![Github](https://img.shields.io/badge/status-under_development-yellow)

![Biobank Image](https://github.com/rivas-lab/Smiles2Dock/blob/main/images/project_diagram.jpg)

## Abstract
Docking is a crucial component in drug discovery aimed at predicting the binding conformation and affinity between small molecules and target proteins. ML-based docking has recently emerged as a prominent approach, outpacing traditional methods like DOCK and AutoDock Vina in handling the growing scale and complexity of molecular libraries. However, the availability of comprehensive and user-friendly datasets for training and benchmarking ML-based docking algorithms remains limited. We introduce Smiles2Dock, an open large-scale multi-task dataset for molecular docking. We created a framework combining P2Rank and AutoDock Vina to dock 1.7 million ligands from the ChEMBL database against 15 AlphaFold proteins, giving us more than 25 million protein-ligand binding scores. The dataset leverages a wide range of high-accuracy AlphaFold protein models, encompasses a diverse set of biologically relevant compounds and enables researchers to benchmark all major approaches for ML-based docking such as Graph, Transformer and CNN-based methods. We also introduce a novel Transformer-based architecture for docking scores prediction and set it as an initial benchmark for our dataset. Our [dataset](https://huggingface.co/datasets/tlemenestrel/Smiles2Dock) and [code](https://github.com/rivas-lab/Smiles2Dock) are publicly available to support the development of novel ML-based methods for molecular docking to advance scientific research in this field.

## Getting Started
To use this code, clone the repository and ensure you have the required Python packages installed.

### Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/rivas-lab/Smiles2Dock.git
```
Once you have cloned the Smiles2Dock repository, cd into it:

```
cd Smiles2Dock 
```

Create the conda environment for docking with:
```
conda env create -f docking.yml
```

Then, activate your conda environment with:
```
conda activate docking
```

## Docking

To get started, follow the steps below. The main docking file is called run.py, which docks an input protein (as a .pdb file) with the CHeMBL database. The script is designed to be run on an HPC Slurm cluster.

### Installation

1. Install the P2Rank binaries from the official GitHub repository:

[Download P2Rank](https://github.com/rdk/p2rank)

### Preparing the PDB File

2. To obtain the PDB file from AlphaFold, go to the AlphaFold website and click on copy link on the PDB file button. Then, navigate to the proteins folder and run:
```bash
wget https://alphafold.ebi.ac.uk/entry/O14791 (your file name instead)
mv {alpha fold name - should start by AF.pdb} {name of your protein.pdb} 
```

### Setting Up Environment on Sherlock

3. Before running P2Rank, make sure you have Java loaded in your environment. This is done by the following command:

```bash
ml java
```

### Predicting the Pockets

4. To predict the binding pockets, use the following command:

```bash
./prank predict -f ../proteins/{name of your protein}.pdb  
```

### Running the Prediction

5. Once you have the PDB file, navigate back to the LLMChemCreator directory and activate your conda env by doing:

```bash
conda activate docking
```

6. To run the code for one folder of the `chembl_split_dir` (for testing purposes), use:

```bash
python run.py ./input/chembl_split_dir_1 {name of your protein}
```

7. If the test run is successful, execute the full docking process with:

```bash 
bash ./run_sherlock.sh {name of your protein}
```

7. Say your protein is called scn10a. You would do:

```bash 
bash ./run_sherlock.sh scn10a
```

This will spawn 2000 jobs to dock all the ChEMBL ligands against the selected protein.

## Training

![Biobank Image](https://github.com/rivas-lab/Smiles2Dock/blob/main/images/architecture.png)

Create the conda environment for training with:
```
conda env create -f docking.yml
```

Then, activate your conda environment with:
```
conda activate training
```

To train a given model with a specific architecture, run:
```
python train.py --protein_model_dim <PROTEIN_MODEL_DIM> --hidden_dim <HIDDEN_DIM> --ligand_model_dim <LIGAND_MODEL_DIM> --dropout_rate <DROPOUT_RATE>
```

To evaluate this model on the test set, run:
```
python eval.py --protein_model_dim <PROTEIN_MODEL_DIM> --hidden_dim <HIDDEN_DIM> --ligand_model_dim <LIGAND_MODEL_DIM> --dropout_rate <DROPOUT_RATE>
```
