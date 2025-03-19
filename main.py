from sklearn.model_selection import train_test_split
from trainer import TransformerTrainer_masked
from torch.utils.data import DataLoader
import torch
import numpy as np
import os
from load_data import  CustomDataset, load_seq_mask, load_inputs
from models import Transformer, Model

# print(os.getpid())



filepath = "data/SequencesMasked.txt"
sequences_masked_number, sequences_number, masked_positions = load_seq_mask(filepath )

pars = load_inputs("pars.txt")

index_val = int(len(sequences_number) - len(sequences_number) * pars['validation_rate'])


val_dataset = CustomDataset(
    input_ids=sequences_masked_number[index_val:],
    labels=sequences_number[index_val:],
)

train_dataset = CustomDataset(
    input_ids=sequences_masked_number[:index_val],
    labels=sequences_number[:index_val],
)

train_loader = DataLoader(train_dataset, batch_size= pars['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size= pars['batch_size'], shuffle=False)

VOCAB_SIZE = 21
d = pars['d']
H = pars['H']
m = pars['m']
L = pars['L']
n = pars['n']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transformer = Transformer(VOCAB_SIZE, d, H, m, L, n, device)
model = Model(transformer, d, VOCAB_SIZE, device)
model.to(device)


trainer = TransformerTrainer_masked(model, train_loader, d, pars['betas'], pars['warmup_steps'], pars['eps'] , val_dataloader = val_loader)

step = 500
epoch = 0
accuracy = 0.
while accuracy < 0.95:
    trainer.train(epoch)
    epoch = epoch + 1
    accuracy = trainer.accuracy_train

    if epoch % step == 0:
        torch.save(model, f"out_train/{pars['file_model']}_w={pars['warmup_steps']}_bs={pars['batch_size']}.pt")
        trainer.results.to_csv(f"out_train/{pars['file_out']}_w={pars['warmup_steps']}_bs={pars['batch_size']}.csv", index=False)
