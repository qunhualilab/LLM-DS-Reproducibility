import os
import glob
import json

def load_results(model_name, dataset_name, agent_type):
    # Get the absolute path of the script
    script_path = os.path.dirname(os.path.abspath(__file__))
    parent_folder = os.path.dirname(script_path)
    pattern = parent_folder+f'/results/{model_name}_{dataset_name}_{agent_type}_results_*.json'
    matching_files = glob.glob(pattern)

    if not matching_files:
        raise FileNotFoundError(f"No results file found matching pattern: {pattern}")

    filename = max(matching_files)
    print(f"Using results file: {filename}")

    with open(filename, 'r') as f:
        results = json.load(f)
    print(f"Input tokens: {results['input_tokens']}; Output tokens: {results['output_tokens']}")
    return results

if __name__ == '__main__':
    pass
