import pandas as pd
import json
import os
import copy
from .data_class import DataSample, Dataset, DatasetCollection
from .time_print import time_print

def load_datasets():
    current_path = os.path.abspath(__file__)
    data_path = '/'.join(current_path.split('/')[:-2])+'/data/'

    ### QRData
    with open(data_path+'QRData/QRData.json', 'r') as f:
        QRData = json.load(f)
    data_samples = []
    for sample in QRData:
        data_sample = DataSample(name='QRData',
                                file_paths=[data_path+'QRData/data/'+data_file for data_file in sample['data_files']],
                                description=sample['data_description'],
                                question=sample['question'],
                                answer=sample['answer'],
                                reference=sample['meta_data']['reference'],
                                keywords=sample['meta_data']['keywords'],
                                question_type=sample['meta_data']['question_type'])
        data_samples.append(data_sample)
    QRData_dataset = Dataset(name='QRData', samples=data_samples, description=f"This is QRData")



    ### StatQA
    dataset_metadata = pd.read_csv(data_path+'StatQA/dataset_metadata.csv')
    dataset_to_description = {}
    for dataset, dataset_description in zip(dataset_metadata['dataset'], dataset_metadata['dataset_description']):
        if not isinstance(dataset_description, str):
            dataset_description = ''
        dataset_to_description[dataset] = dataset_description

    def restructure_metadata(data):
        # Extract the columns metadata
        columns = [
            {
                "name": row["column_header"],
                "description": row["column_description"],
                "data_type": row["data_type"]
            }
            for _, row in data.iterrows()
        ]
        
        # Extract num_of_rows and is_normality (assuming they are uniform across the dataset)
        num_of_rows = data["num_of_rows"].iloc[0]
        is_normality = data["is_normality"].iloc[0]
        
        # Create the final structure
        structured_data = {
            "columns": columns,
            "num_of_rows": num_of_rows,  # Please ignore this feature
            "is_normality": is_normality # Please ignore this feature
        }
        
        # Convert to JSON format
        return structured_data
    dataset_to_col_meta = {}
    for filename in os.listdir(data_path+'StatQA/column_metadata'):
        file_path = os.path.join(data_path+'StatQA/column_metadata', filename)
        col_meta = pd.read_csv(file_path)
        col_meta = restructure_metadata(col_meta)
        dataset_to_col_meta[filename.split('_')[0]] = col_meta

    with open(data_path+'StatQA/mini-StatQA.json', 'r') as f:
        StatQA = json.load(f)
    data_samples = []
    for sample in StatQA:
        answer = '\n'.join(['Using method '+option['method']+', the conclusion is: '+option['conclusion'] for option in json.loads(sample['results'])])
        #answer += "\nAll the above conclusions are considered correct."
        data_sample = DataSample(name='StatQA',
                                file_paths=[data_path+'StatQA/processed_dataset/'+sample['dataset'].replace(' ', '_')+'.csv'],
                                question=sample['refined_question'],
                                relevant_column=sample['relevant_column'],
                                answer=answer,
                                results=sample['results'],
                                question_type=sample['task'],
                                difficulty=sample['difficulty'],
                                description=dataset_to_description[sample['dataset']],
                                column_metadata=[dataset_to_col_meta[sample['dataset']]]
                                )
        data_samples.append(data_sample)
    StatQA_dataset = Dataset(name='StatQA', samples=data_samples, description=f"This is StatQA")



    ### DiscoveryBench
    id_to_metadata = {}
    for subject_name in os.listdir(data_path+'DiscoveryBench'):
        file_dir = os.path.join(data_path+'DiscoveryBench', subject_name)
        if file_dir.endswith('.csv'): continue
        for file_name in os.listdir(file_dir):
            file_path = os.path.join(file_dir, file_name)
            if not file_path.endswith('.json'): continue
            with open(file_path, 'r') as f:
                metadata = json.load(f)
            file_path_split = file_path.split('/')
            # subject
            subject = subject_name
            # metadata id
            metadata_id = file_path_split[-1].split('_')[1].split('.')[0]
            queries = metadata['queries']
            if len(queries) != 1: print('Check what is in queries.')
            for query in queries[0]:
                metadata['question'] = query['question']
                metadata['qid'] = query['qid']
                metadata['question_type'] = query['question_type']
                id_to_metadata[subject+'_'+metadata_id+'_'+str(query['qid'])] = copy.deepcopy(metadata)

    DiscoveryBench = pd.read_csv(data_path+'DiscoveryBench/answer_key_real.csv')
    data_samples = []
    for sample in DiscoveryBench.iterrows():
        sample = sample[1]
        sample_id = sample['dataset']+'_'+str(sample['metadataid'])+'_'+str(sample['query_id'])
        metadata = id_to_metadata[sample_id]
        domain_knowledge = metadata['domain_knowledge'] if 'domain_knowledge' in metadata else None
        for dataset in metadata['datasets']:
            if len(dataset['columns']) > 1: print('Check what is in the dataset columns')
        column_metadata = [{'columns': dataset['columns']['raw']} for dataset in metadata['datasets']]
        data_sample = DataSample(name='DiscoveryBench',
                                file_paths=[data_path+'DiscoveryBench/'+sample['dataset']+'/'+dataset['name'] for dataset in metadata['datasets']],
                                subject=sample['dataset'],
                                question=metadata['question'],
                                answer=sample['gold_hypo'],
                                question_type=metadata['question_type'],
                                domain_knowledge=domain_knowledge,
                                workflow=metadata['workflow_tags'],
                                description=[dataset['description'] for dataset in metadata['datasets']],
                                column_metadata=column_metadata,
                                )
        data_samples.append(data_sample)
    DiscoveryBench_dataset = Dataset(name='DiscoveryBench', samples=data_samples, description=f"This is DiscoveryBench")

    collection = DatasetCollection(names=['QRData', 'StatQA', 'DiscoveryBench'], 
                                   datasets=[QRData_dataset, StatQA_dataset, DiscoveryBench_dataset])
    time_print('Loaded QRData, StatQA, and DiscoveryBench datasets.')
    return collection

if __name__ == '__main__':
    pass
