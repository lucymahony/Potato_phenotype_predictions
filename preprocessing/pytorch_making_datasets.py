import numpy as np
import torch
from torch.utils.data import Dataset, random_split
from torch import nn
import pandas as pd


def generate_phenotypes_dict(file_path):
    with open (file_path, "r") as f:
        phenotypes = {}
        for i, line in enumerate(f):
            # Genotype,Flesh Colour,Genotype,tubershape,Genotype,decol5min,Genotype,DSC_Onset 
            if i == 0:
                continue # skip the header
            parts = line.strip().split(",")
            # if any parts = '' or ' ' or 'nan' or 'NaN' or 'N/A' or 'NA' then skip this line
            if any(part == '' or part == ' ' or part.lower() in ['nan', 'n/a', 'na'] for part in parts):
                continue
            Genotype, Colour, Genotype, tubershape, Genotype, decol5min, Genotype, DSC_Onset = line.strip().split(",")
            print('Genotype', Genotype, 'Colour', Colour, 'tubershape', tubershape, 'decol5min', decol5min, 'DSC_Onset', DSC_Onset)
            phenotypes[Genotype] = {'colour': float(Colour), 'tubershape': float(tubershape), 'decol5min': float(decol5min), 'DSC_Onset': float(DSC_Onset)}
    return phenotypes


class MultiOmicDataset(Dataset):
    def __init__(self, lc_ms_path, gc_ms_path, transcriptomics_path, phenotypes_dict, target_keys):

        self.lc_ms = pd.read_csv(lc_ms_path, index_col=0).T # Transpose to have genotypes as rows
        self.gc_ms = pd.read_csv(gc_ms_path, index_col=0)
        self.gc_ms.index = 'CE' + self.gc_ms.index.astype(str)
        self.trans = pd.read_csv(transcriptomics_path, index_col=0).T

        # Ensure common genotypes
        common_genos = set(self.lc_ms.index) & set(self.gc_ms.index) & set(self.trans.index) & set(phenotypes_dict.keys())
        self.genotypes = sorted(list(common_genos))

        # Reduce to common genotypes
        self.lc_ms = self.lc_ms.loc[self.genotypes]
        self.gc_ms = self.gc_ms.loc[self.genotypes]
        self.trans = self.trans.loc[self.genotypes]
        self.phenotypes = [phenotypes_dict[geno] for geno in self.genotypes]
        self.target_keys = target_keys

    def __len__(self):
        return len(self.genotypes)

    def __getitem__(self, idx):
        x_lc = torch.tensor(self.lc_ms.iloc[idx].values, dtype=torch.float32)
        x_gc = torch.tensor(self.gc_ms.iloc[idx].values, dtype=torch.float32)
        x_trans = torch.tensor(self.trans.iloc[idx].values, dtype=torch.float32)
        
        targets = [self.phenotypes[idx][key] for key in self.target_keys]
        y = torch.tensor(targets, dtype=torch.float32)

        return (x_lc, x_gc, x_trans), y
    

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    phenotypes_file_path = "../input_data/phenotypes.csv"
    phenotypes = generate_phenotypes_dict(phenotypes_file_path)
    
    lc_ms_path = "../input_data/LC_MS.csv"
    gc_ms_path = "../input_data/GC_MS.csv"
    transcriptomics_path = '../input_data/transcriptomics.csv'

    target_keys = ['colour', 'tubershape', 'decol5min', 'DSC_Onset']

    dataset = MultiOmicDataset(
        lc_ms_path=lc_ms_path,
        gc_ms_path=gc_ms_path,
        transcriptomics_path=transcriptomics_path,
        phenotypes_dict=phenotypes,
        target_keys=target_keys)

    all_inputs = []
    all_pheno_dicts = []

    for i in range(len(dataset)):
        (x_lc, x_gc, x_trans), y = dataset[i]
        all_inputs.append((x_lc, x_gc, x_trans))
        all_pheno_dicts.append(dataset.phenotypes[i])


    # Convert to tensors and save
    lc_tensor = torch.stack([x[0] for x in all_inputs])
    gc_tensor = torch.stack([x[1] for x in all_inputs])
    trans_tensor = torch.stack([x[2] for x in all_inputs])

    torch.save({'lc': lc_tensor, 'gc': gc_tensor, 'trans': trans_tensor, 'phenotypes': all_pheno_dicts, 'genotypes': dataset.genotypes,},
               '../intermediate_data/preprocessed_dataset.pt')
    print("Dataset saved to ../intermediate_data/preprocessed_dataset.pt")