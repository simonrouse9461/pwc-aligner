from .imports import *


class DataMixer(IterableDataset):
    '''
    Utility to iterate over multiple datasets
    '''
    def __init__(self, *args):
        self.datasets = args
        
    def __iter__(self):
        for dataset in self.datasets:
            for data in dataset:
                yield data
    
    def __len__(self):
        return sum(map(len, self.datasets))