# LLMChemCreator

How to use P2Rank to predict binding pockets

P2Rank is based on Java. You simply need to install the binaries at:

https://github.com/rdk/p2rank

and then run on Sherlock (to load java - need to have your conda env activated first):

ml java

The command to predic the pockets is:

./prank predict -f test_data/1fbl.pdb  

To get the pdb file, go to the AlphaFold website and click on copy link on the PDB file button. Then, in Sherlock, do:

wget https://alphafold.ebi.ac.uk/entry/O14791 (your file name instead)

