from torch.utils.data import Dataset
import re
import torch
import ast



# Read desired inputs from <f> file
def load_inputs(f):
    with open(f, 'r') as ff:
        lines = ff.readlines()
    
    inputs = {}
    for line in lines:
        if (line == '\n' or line[0] == '#'): continue
        line = line.split('\t')[0]
        key_end_idx, value_start_idx = None, None
        idx = 0
        while value_start_idx is None:
            if key_end_idx is None:
                if line[idx] == ' ': key_end_idx = idx
            else:
                if (value_start_idx is None) and (line[idx] != ' '): value_start_idx = idx
            idx += 1
        key, value = line[0:key_end_idx], line[value_start_idx:]
        value = value[:-1] if value[-1]=='\n' else value
        if (value[0] == '[') and (value[-1] == ']'):
            values_list = value[1:-1].split(',')
            try: inputs[key] = [eval(el) for el in values_list]
            except: inputs[key] = [el.replace(' ', '') for el in values_list]
        else:
            try: inputs[key] = eval(value)
            except: inputs[key] = value
    return inputs


    
    
# load: sequence masked in number, seqeunces non masked in number and mask (position of mask and masked number)
def load_seq_mask(filename, n=None):
    sequences_masked_number = []
    sequences_number = []
    masked_positions = []

    with open(filename, "r") as file:
        next(file)  
        for i, line in enumerate(file):
            if n is not None and i >= n:
                break
            smn, sn, mp = line.strip().split("\t")
            # string in list of number
            sequences_masked_number.append(ast.literal_eval(smn))  
            sequences_number.append(ast.literal_eval(sn))  
            masked_positions.append(ast.literal_eval(mp))  

    return sequences_masked_number, sequences_number, masked_positions

# make dataset structure for model
class CustomDataset(Dataset):
    def __init__(self, input_ids, labels):
        self.input_ids = torch.tensor(input_ids, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'labels': self.labels[idx]
        }
