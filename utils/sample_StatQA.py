import numpy as np

def sample_StatQA(dataset, samples_per_category=80):
    """
    Samples a specified number of samples for each category from the dataset.

    Args:
        dataset: The dataset object with a sample generator.
        samples_per_category: Number of samples to select for each category.
    
    Returns:
        List of indices of the selected samples.
    """
    np.random.seed(42)
    selected_indices = []
    
    # Group indices by question type
    question_type_indices = {}
    for idx, sample in enumerate(dataset.sample_generator()):
        if sample.question_type not in question_type_indices:
            question_type_indices[sample.question_type] = []
        question_type_indices[sample.question_type].append(idx)
    
    # Sample indices for each category
    for q_type, indices in question_type_indices.items():
        if len(indices) >= samples_per_category:
            selected_indices.extend(np.random.choice(indices, samples_per_category, replace=False))
        else:
            print(f"Warning: Not enough samples for question type '{q_type}'. Only {len(indices)} samples available.")
            selected_indices.extend(indices)
    
    return set(sorted(selected_indices))

if __name__ == '__main__':
    pass
