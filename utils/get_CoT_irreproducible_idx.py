import json
import numpy as np
import pandas as pd
import glob
from .load_results import load_results
from .sample_StatQA import sample_StatQA
from .load_data import load_datasets
import os
import sys

def get_irreproducible_idx(model_name, dataset_name, agent_type):
    results = load_results(model_name, dataset_name, agent_type)
    indices = np.arange(len(results['reproducibility_scores']))
    if dataset_name == 'StatQA':
        collection = load_datasets()
        dataset = collection.get_dataset('StatQA')
        indices = np.array(sorted(list(sample_StatQA(dataset))))
    irreproducible_idx = indices[np.array(results['reproducibility_scores'])!=1]
    print('Found', len(irreproducible_idx), 'irreproducible samples.')
    return irreproducible_idx

if __name__ == '__main__':
    model_name, dataset_name, agent_type = 'llama-3.3', 'DiscoveryBench', 'COT'
    irreproducible_idx = get_irreproducible_idx(model_name, dataset_name, agent_type)
    print(irreproducible_idx)
    print('Found', len(irreproducible_idx), 'irreproducible samples.')