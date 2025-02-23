from dataclasses import dataclass, field
from typing import Dict, Optional, List, Union

@dataclass
class DataSample:
    """
    Represents an individual data sample with metadata and analysis context.
    """
    name: str  # Name of the dataset
    file_paths: List[str]  # File paths of the datasets
    question: str  # The question addressed by this dataset sample
    question_type: str  # The type/category of the question
    answer: Optional[Union[str, float, bool]] = None  # The answer to the question
    results: List[Dict[str, str]] = field(default_factory=list) # Answer structure in StatQA

    description: Union[List[str], str] = ''  # Description of the datasets
    column_metadata: Optional[List[Dict[str, str]]] = field(default_factory=list)  # Metadata for columns of datasets (e.g., column descriptions)
    domain_knowledge: str = ''  # Domain knowledge or context provided by the dataset authors
    keywords: List[str] = field(default_factory=list)  # Keywords or tags describing the dataset
    reference: str = '' # Reference of the dataset
    subject: str = '' # Subject of the dataset

    workflow: List[str] = field(default_factory=list)  # Recommended analysis workflow provided by the authors
    relevant_column: List = field(default_factory=list)  # Columns relevant to answering the question
    difficulty: str = ''  # Difficulty level of the question

class Dataset:
    """
    A class representing a collection of data samples.
    """
    def __init__(self, name, samples=None, description=''):
        """
        Initialize a Dataset instance.

        :param name: Name of the dataset
        :param samples: List of data samples in the dataset (default: empty list)
        :param description: Optional description of the dataset
        """
        self.name = name
        self.samples = samples if samples is not None else []
        self.description = description

    def sample_generator(self):
        """
        A generator function to iterate over the samples in the dataset.
        """
        for sample in self.samples:
            yield sample
    
    def get_sample(self, identifier: Union[int, str]) -> DataSample:
        """
        Retrieve a specific sample from the dataset by index or name.
        
        :param identifier: Either an integer index or the name of the sample
        :return: The requested DataSample
        :raises IndexError: If the index is out of range
        :raises ValueError: If the sample name is not found or if the identifier type is invalid
        """
        if isinstance(identifier, int):
            if 0 <= identifier < len(self.samples):
                return self.samples[identifier]
            raise IndexError(f"Sample index {identifier} is out of range. Valid range: 0-{len(self.samples)-1}")
            
        elif isinstance(identifier, str):
            for sample in self.samples:
                if sample.name == identifier:
                    return sample
            raise ValueError(f"No sample found with name '{identifier}'")
            
        else:
            raise ValueError(f"Invalid identifier type. Expected int or str, got {type(identifier)}")

class DatasetCollection:
    """
    A class for managing multiple datasets.
    """
    def __init__(self, names: str, datasets: List[Dataset]):
        self.datasets: Dict[str, Dataset] = {}  # Datasets stored by their name as a key
        for name, dataset in zip(names, datasets):
            self.add_dataset(name, dataset)

    def add_dataset(self, name: str, dataset: Dataset):
        """Adds a new dataset to the collection."""
        self.datasets[name] = dataset

    def get_dataset(self, name: str) -> Optional[Dataset]:
        """Retrieves a dataset by name."""
        return self.datasets.get(name)

    def remove_dataset(self, name: str):
        """Removes a dataset by name."""
        if name in self.datasets:
            del self.datasets[name]