import torch
import numpy as np
import scipy

def transformer_collate(batch):
    # batch sample consists of [claimID, claim, label, snippets]

    claimIDs = [t[0] for t in batch]
    claims = [t[1] for t in batch]
    labels = [t[2] for t in batch]
    snippets = [t[3] for t in batch]

    return  claimIDs, claims, labels, snippets

class TransformerDataset(torch.utils.data.Dataset):
    def __init__(self, sub_main_data, sub_snippets_data, label_order):
        self.sub_main_data = sub_main_data
        self.sub_snippets_data = sub_snippets_data
        self.label_order = label_order

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.sub_main_data)

    def __getitem__(self, index):
        main_data_sample = self.sub_main_data[index]
        sub_snippets_data_sample = self.sub_snippets_data[index]

        claimID = main_data_sample[0]
        claim = main_data_sample[1]
        label = main_data_sample[2]
        label = self.label_order.index(label)
        snippets = sub_snippets_data_sample[1:]

        return claimID, claim, label, snippets