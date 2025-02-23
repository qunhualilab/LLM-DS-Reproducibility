import os
import json
from enum import Enum
import warnings
import click
from collections import OrderedDict
from utils.load_data import load_datasets
from utils.time_print import time_print
from utils.sample_StatQA import sample_StatQA
from utils.get_CoT_irreproducible_idx import get_irreproducible_idx
from .get_agent import AgentType, get_agent, get_agent_instruction, get_agent_type

warnings.filterwarnings("ignore")


@click.command()
@click.option('--dataset_name', required=True, help='Name of the dataset to run experiments on.')
@click.option('--model_name', required=True, help='Name of the model to use.')
@click.option('--agent_type', required=True, type=click.Choice(['COT', 'ROT', 'REACT', 'REFLEXION']), help='Type of the agent to use.')
@click.option('--overwrite', is_flag=True, help='Flag to overwrite results file if it exists.')
def run_experiment(dataset_name, model_name, agent_type, overwrite):
    parent_dir = os.path.abspath("")

    # Load the datasets
    collection = load_datasets()
    dataset = collection.get_dataset(dataset_name)

    # Convert agent_type string to AgentType enum
    agent_type_enum = get_agent_type(agent_type)

    results = {}
    output_dir = os.path.join(parent_dir, 'results')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(parent_dir, 'results', f'{model_name}_{dataset_name}_{agent_type}.json')

    # Load existing results if not overwriting
    if not overwrite and os.path.exists(output_file):
        with open(output_file, 'r') as f:
            results = json.load(f)

    indices = set(range(len(dataset.samples)))
    if dataset_name == 'StatQA':
        indices = sample_StatQA(dataset)

    if agent_type == 'REFLEXION':
        with open(os.path.join(parent_dir, 'results', f'{model_name}_{dataset_name}_COT.json'), 'r') as f:
            results = json.load(f)
        indices = get_irreproducible_idx(model_name=model_name, dataset_name=dataset_name, agent_type='COT')
        for i in range(len(dataset.samples)):
            if str(i) in results and i not in indices:
                # Remove reproducible samples
                results.pop(str(i))

    # Iterate through the dataset samples
    for i, sample in enumerate(dataset.sample_generator()):
        if i not in indices:
            results[str(i)] = {"final_answer": "Sample not included"}
            continue

        agent = get_agent(
            agent_type=agent_type_enum, 
            model_config=os.path.join(parent_dir, 'config', 'model_config.json'),
            api_config=os.path.join(parent_dir, 'config', 'api_config.json'),
            model_name=model_name
        )

        response = agent.run(sample, get_agent_instruction(agent_type_enum), deepseek=(model_name=='deepseek-r1'))
        response.update({"answer": sample.answer})
        if response['final_answer'] != "No final answer reached":
            time_print(f'Sample {i} finished.')
        else:
            time_print(f'Sample {i} failed.')
        results[str(i)] = response

        # Save results incrementally
        results = OrderedDict(sorted(results.items(), key=lambda x: int(x[0])))
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

    time_print("Experiment completed.")

if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()
    print("Running experiment...")
    run_experiment()
