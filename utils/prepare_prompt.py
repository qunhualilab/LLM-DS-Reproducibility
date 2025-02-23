from .data_class import DataSample
import os
import pandas as pd
import pdb

def prepare_prompt(sample: DataSample):
    name = sample.name
    if name == 'DiscoveryBench':
        prepared_prompt = prepare_discoverybench_prompt(sample)
    elif name == 'QRData':
        prepared_prompt = prepare_qrdata_prompt(sample)
    elif name == 'StatQA':
        prepared_prompt = prepare_statqa_prompt(sample)
    return {
        "file_paths": prepared_prompt["file_paths"], 
        "descriptions": prepared_prompt["descriptions"], 
        "question": prepared_prompt["question"]
    }

def prepare_discoverybench_prompt(sample):
    question = sample.question
    file_paths = [os.path.basename(file_path) for file_path in sample.file_paths]

    descriptions = ''
    
    descriptions += 'Below are the descriptions of the datasets and dataset columns.\n'
    for file_path, description, _column_metadata in zip(file_paths, sample.description, sample.column_metadata):
        descriptions += 'Dataset '+file_path+': '+description+'\n'
        descriptions += 'Descriptions for the columns:\n'
        column_description = ''
        for column in _column_metadata['columns']:
            column_description += 'Column name \''+column['name']+'\': '+column['description']+'\n'
        descriptions += column_description
        descriptions += '\n'

    if sample.domain_knowledge:
        descriptions += 'Domain Knowledge: '+sample.domain_knowledge
    
    return {
        "file_paths": file_paths,
        "descriptions": descriptions,
        "question": question
    }

def prepare_qrdata_prompt(sample):
    question = sample.question
    file_paths = [os.path.basename(file_path) for file_path in sample.file_paths]

    descriptions = ''
    descriptions += 'Below is the description of the dataset.\n'+sample.description+'\n\n'
    #for file_path in sample.file_paths:
    #    dataset = pd.read_csv(file_path)
    #    descriptions += 'Below are the column names of the dataset ' + os.path.basename(file_path) + ':\n'
    #    descriptions += ', '.join(dataset.columns) + '\n'

    for file_path in sample.file_paths:
        dataset = pd.read_csv(file_path)
        descriptions += 'Below is a statistical summary of the dataset '+os.path.basename(file_path)+':\n'
        descriptions += dataset.describe().to_string() + '\n'

    return {
        "file_paths": file_paths,
        "descriptions": descriptions,
        "question": question
    }

def prepare_statqa_prompt(sample):
    question = sample.question
    file_paths = [os.path.basename(file_path) for file_path in sample.file_paths]

    descriptions = ''
    if sample.description != '':
        descriptions += 'Below is the description of the dataset.\n'+sample.description+'\n\n'

    descriptions += 'Below are the descriptions of the columns.\n'
    for _column_metadata in sample.column_metadata[0]['columns']:
        descriptions += 'Column name \''+_column_metadata['name']+'\''
        if isinstance(_column_metadata['description'], str):
            descriptions += '. Description: '+_column_metadata['description']
        descriptions += '. Data type: '+_column_metadata['data_type']+'\n'

    return {
        "file_paths": file_paths,
        "descriptions": descriptions,
        "question": question
    }