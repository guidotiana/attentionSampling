# Transformer Training for Amino Acid Sequences
## Overview
This repository contains scripts and data for training a transformer model on amino acid sequences. The training process is managed in main.py, which loads the preprocessed dataset before starting model training.


## Running the Training
To start the training process, run:

    bash run.sh

## Data Format

- SequencesMasked.txt:
    - Column 1: Original amino acid sequences.
    - Column 2: Sequences with 15% of amino acids masked.
    - Column 3: Information about masked positions and original amino acids.
- sequences_t=127.dat: 
    Contains original sequences from simulations based on real contact maps.
