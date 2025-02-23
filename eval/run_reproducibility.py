import os
import numpy as np
import json
from collections import Counter
from datetime import datetime
from utils.load_data import load_datasets
from utils.time_print import time_print
from utils.output_parser import extract_python_code
from utils.sample_StatQA import sample_StatQA
from utils.get_CoT_irreproducible_idx import get_irreproducible_idx
from .reproducibility import Reproducibility
import click
import glob
import warnings

warnings.filterwarnings("ignore")

def get_accuracy_by_reproducibility(accuracy_scores, reproducibility_scores):
    accuracy_scores, reproducibility_scores = np.array(accuracy_scores), np.array(reproducibility_scores)
    accuracy_scores_1, accuracy_scores_0 = accuracy_scores[reproducibility_scores==1], accuracy_scores[reproducibility_scores==0]
    return len(accuracy_scores_1), len(accuracy_scores_0), np.mean(accuracy_scores_1), np.mean(accuracy_scores_0)

@click.command()
@click.option('--dataset_name', required=True, help='Name of the dataset to run experiments on.')
@click.option('--model_name', required=True, help='Name of the model to use.')
@click.option('--agent_type', required=True, help='Type of the agent to use.')
@click.option('--accuracy', is_flag=True, help='Flag to compute accuracy.')
@click.option('--reproducibility', is_flag=True, help='Flag to compute reproducibility.')
@click.option('--all_metrics', is_flag=True, help='Flag to compute all metrics.')
def main(dataset_name, model_name, agent_type, accuracy, reproducibility, all_metrics):
    collection = load_datasets()
    dataset = collection.get_dataset(dataset_name)
    if all_metrics:
        accuracy, reproducibility = True, True

    if not accuracy and not reproducibility:
        time_print('No metrics selected. Exiting.')
        return
    
    parent_dir = os.path.abspath("")
    reproducibility_evaluator = Reproducibility()
    results_path = os.path.join(parent_dir, 'results', f'{model_name}_{dataset_name}_{agent_type}.json')
    with open(results_path, 'r') as f:
        dataset_results = json.load(f)

    input_tokens, output_tokens = 0, 0
    for idx in dataset_results:
        if 'steps' not in dataset_results[idx]: continue
        for step in dataset_results[idx]['steps']:
            input_tokens += step['usage_metadata']['input_tokens']
            output_tokens += step['usage_metadata']['output_tokens']
    time_print(f"Experiment {model_name}_{dataset_name}_{agent_type} cost {input_tokens} input tokens and {output_tokens} output tokens.")

    indices = set(range(len(dataset.samples)))
    if dataset_name == 'StatQA':
        indices = sample_StatQA(dataset)

    if agent_type == 'REFLEXION':
        indices = list(get_irreproducible_idx(model_name=model_name, dataset_name=dataset_name, agent_type='COT'))

    accuracy_scores = []
    reproducibility_scores = []
    reproducibility_reasons = []

    for j, (idx, sample) in enumerate(zip(dataset_results, dataset.sample_generator())):
        if int(idx) not in indices:
            continue

        j = int(idx)
        sample = dataset.get_sample(j)
            
        result = dataset_results[idx]

        if agent_type == 'REACT' or agent_type == 'REFLEXION':
            action_input = next(
                (step['action_input'] for step in result['steps'][::-1] if step['action_input']),
                ''
            )
            workflow = next(
                (step['workflow'] for step in result['steps'][::-1] if step.get('workflow', '')),
                ''
            )
        else:
            action_input = next(
                (step['action_input'] for step in result['steps'] if step['action_input']),
                ''
            )
            workflow = next(
                (step['workflow'] for step in result['steps'] if step.get('workflow', '')),
                ''
            )

        if model_name == 'deepseek-r1':
            if agent_type == 'REACT' or agent_type == 'REFLEXION':
                workflow = next(
                    (step['content'][:step['content'].find('```python')] for step in result['steps'][::-1] if step['content'].find('```python') != -1),
                    ''
                )
            else:
                workflow = next(
                    (step['content'][:step['content'].find('```python')] for step in result['steps']),
                    ''
                )

        if accuracy:
            dataset_specific_prompt = ''
            if sample.name == 'DiscoveryBench':
                dataset_specific_prompt = "\nFor numerical questions, any predicted answer within 1% of the ground truth answer is considered correct. Please compare abs(predicted-ground_truth)/abs(ground_truth) with 1% to make your decision.\n"
            if sample.name == 'QRData':
                dataset_specific_prompt = "\nFor numerical questions, any result within 3% of the ground truth answer is considered correct. Please compare abs(predicted-ground_truth)/abs(ground_truth) with 3% to make your decision.\n"
            if sample.name == 'StatQA':
                dataset_specific_prompt = "\nPlease evaluate the accuracy of the predicted answer. As long as the predicted method aligns with the objective, it is acceptable. Then you should score based on the conclusion. If the predicted conclusion matches with the ground truth conclusion, score 1. If there are conflicting conclusions in the ground truth answers, as long as the predicted answer mentions any conclusion, it should be scored 1. For numerical questions, any result within 1% of the ground truth answer is considered correct. Please compare abs(predicted-ground_truth)/abs(ground_truth) with 1% to make your decision.\n"
            accuracy_score = reproducibility_evaluator.accuracy(
                question=sample.question,
                predicted_answer=result['final_answer'],
                true_answer=sample.answer,
                dataset_specific_prompt=dataset_specific_prompt
            )
            accuracy_scores.append(accuracy_score)
            time_print(f'Sample {idx}: Accuracy score: {accuracy_score}')
            if len(accuracy_scores) % 40 == 0:
                time_print(f'Running {len(accuracy_scores)} average accuracy score: {np.mean(accuracy_scores)}')
        else:
            old_files_pattern = os.path.join(parent_dir, 'results', f'{model_name}_{dataset_name}_{agent_type}_results_*.json')
            for old_file in glob.glob(old_files_pattern):
                old_data = json.load(open(old_file))
                if 'accuracy_scores' in old_data:
                    accuracy_scores = old_data['accuracy_scores']

        if reproducibility:
            reproducibility_score, reason = reproducibility_evaluator.llm_reproducibility(
                sample=sample,
                code=action_input,
                workflow=workflow,
                final_answer=result['final_answer']
            )
            reproducibility_scores.append(reproducibility_score)
            reproducibility_reasons.append(reason)
            time_print(f'Sample {idx}: Reproducibility score: {reproducibility_score}')
            if len(reproducibility_scores) % 20 == 0:
                num_1, num_0, acc_1, acc_0 = get_accuracy_by_reproducibility(accuracy_scores, reproducibility_scores)
                time_print(f'Accuracy (reproducibility=1): {acc_1:.4f} (num: {num_1}); Accuracy (reproducibility=0): {acc_0:.4f} (num: {num_0})')

        if reproducibility_evaluator.converted_codes:
            with open(os.path.join(parent_dir, 'results', 'converted_codes.json'), 'w') as f:
                json.dump([{"index": i, "converted_code": converted_code, "original_code": reproducibility_evaluator.original_codes[i], "reason": reproducibility_reasons[i]} for i, converted_code in enumerate(reproducibility_evaluator.converted_codes)], f, indent=4)
        if reproducibility_evaluator.converted_workflows:
            with open(os.path.join(parent_dir, 'results', 'converted_workflows.json'), 'w') as f:
                json.dump([{"index": i, "converted_workflow": converted_workflow, "original_workflow": reproducibility_evaluator.original_workflows[i], "reason": reproducibility_reasons[i]} for i, converted_workflow in enumerate(reproducibility_evaluator.converted_workflows)], f, indent=4)
        if reproducibility_evaluator.human_prompts:
            with open(os.path.join(parent_dir, 'results', 'human_prompts.json'), 'w') as f:
                json.dump([{"index": i, "human_prompts": human_prompts} for i, human_prompts in enumerate(reproducibility_evaluator.human_prompts)], f, indent=4)

    time_print(f"Accuracy score of {len(accuracy_scores)} samples: {np.mean(accuracy_scores)}")
    results = {
        "accuracy_scores": accuracy_scores,
        "mean_accuracy_score": np.mean(accuracy_scores),
    }
    if reproducibility:
        time_print(f"Reproducibility score of {len(reproducibility_scores)} samples: {np.mean(reproducibility_scores)}")
        results.update({
            "reproducibility_scores": reproducibility_scores,
            "mean_reproducibility_score": np.mean(reproducibility_scores),
            "reproducibility_reasons": reproducibility_reasons
        })

    old_files_pattern = os.path.join(parent_dir, 'results', f'{model_name}_{dataset_name}_{agent_type}_results_*.json')
    for old_file in glob.glob(old_files_pattern):
        os.remove(old_file)

    results.update({
        "input_tokens": reproducibility_evaluator.input_tokens,
        "output_tokens": reproducibility_evaluator.output_tokens,
        "cost": reproducibility_evaluator.input_tokens*2.5/1000000+reproducibility_evaluator.output_tokens*10/1000000
    })
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(parent_dir, 'results', f'{model_name}_{dataset_name}_{agent_type}_results_{timestamp}.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    if reproducibility:
        if reproducibility_evaluator.converted_codes:
            with open(os.path.join(parent_dir, 'results', f'{model_name}_{dataset_name}_{agent_type}_converted_codes.json'), 'w') as f:
                json.dump([{"index": i, "converted_code": converted_code, "original_code": reproducibility_evaluator.original_codes[i], "reason": reproducibility_reasons[i]} for i, converted_code in enumerate(reproducibility_evaluator.converted_codes)], f, indent=4)
        if reproducibility_evaluator.converted_workflows:
            with open(os.path.join(parent_dir, 'results', f'{model_name}_{dataset_name}_{agent_type}_converted_workflows.json'), 'w') as f:
                json.dump([{"index": i, "converted_workflow": converted_workflow, "original_workflow": reproducibility_evaluator.original_workflows[i], "reason": reproducibility_reasons[i]} for i, converted_workflow in enumerate(reproducibility_evaluator.converted_workflows)], f, indent=4)
        if reproducibility_evaluator.human_prompts:
            with open(os.path.join(parent_dir, 'results', f'{model_name}_{dataset_name}_{agent_type}_human_prompts.json'), 'w') as f:
                json.dump([{"index": i, "human_prompts": human_prompts} for i, human_prompts in enumerate(reproducibility_evaluator.human_prompts)], f, indent=4)
    
    time_print(f'Results saved to {output_path}')
    

if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()
    main()
