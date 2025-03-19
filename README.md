# Transformer Training for Amino Acid Sequences
## Overview
This repository contains scripts and data for training a transformer model on amino acid sequences. The training process is managed in main.py, which loads the preprocessed dataset before starting model training.
## Repository Structure

    main.py - The main script to launch transformer training.
    SequencesMasked.txt - Contains amino acid sequences in three columns:
        Original sequences.
        Sequences with 15% masked amino acids.
        Masking details (position and masked amino acid).
    Sequences.dat - Contains original sequences generated from simulations using real contact maps.
    trainer.py
    model.py
    load_.py
    parms.txt

## Running the Training
To start the training process, run:

bash run.sh

## Data Format

    SequencesMasked.txt
        Column 1: Original amino acid sequences.
        Column 2: Sequences with 15% of amino acids masked.
        Column 3: Information about masked positions and original amino acids.
    Sequences.dat
        Contains original sequences from simulations based on real contact maps.
