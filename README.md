# Smiles2Dock

![GitHub](https://img.shields.io/github/license/rivas-lab/Smiles2Dock)
![Github](https://img.shields.io/badge/status-under_development-yellow)

![Biobank Image](https://github.com/rivas-lab/Smiles2Dock/raw/main/images/diagram_project.jpg)

## How to Use P2Rank to Predict Binding Pockets

P2Rank is a powerful tool based on Java for predicting protein-ligand binding pockets. To get started, follow the steps below.

### Installation

1. Install the P2Rank binaries from the official GitHub repository:

[Download P2Rank](https://github.com/rdk/p2rank)

### Preparing the PDB File

2. To obtain the PDB file from AlphaFold, go to the AlphaFold website and click on copy link on the PDB file button. Then, navigate to the proteins folder in Sherlock and run:
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

