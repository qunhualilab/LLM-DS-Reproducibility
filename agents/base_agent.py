from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseAgent(ABC):
    """Base class for all agents."""
    
    @abstractmethod
    def run(self, task: str) -> Dict[str, Any]:
        """Run the agent on a task."""
        pass
    
