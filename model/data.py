import pickle
from pathlib import Path
from queue import Queue

import torch
from torch.utils.data import Dataset, DataLoader

class Data:
    def __init__(self, data_root, train_dataset,val_dataset, test_dataset, batch_size=128, num_workers=0):
        self.data_root = Path(data_root)

        self.batch_size = batch_size
        self.num_workers = num_workers

        

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        
    def train_loader(self):
        return DataLoader(self.train_dataset, collate_fn=self.collator,
                          batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_loader(self):
        return DataLoader(self.val_dataset, collate_fn=self.collator,
                          batch_size=self.batch_size, num_workers=self.num_workers)

    def test_loader(self):
        return DataLoader(self.test_dataset, collate_fn=self.collator,
                          batch_size=self.batch_size, num_workers=self.num_workers)