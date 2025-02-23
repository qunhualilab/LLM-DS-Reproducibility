from enum import Enum
from agents.cot_agent import ChainOfThoughtAgent
from agents.react_agent import ReActAgent
from agents.reflexion_agent import ReflexionAgent

class AgentType(Enum):
    COT = "chain_of_thought"
    ROT = "reproducibility_of_thought"
    REACT = "react"
    REFLEXION = 'reflexion'

def get_agent(agent_type: AgentType, model_config: str, api_config: str, model_name: str):
    """Factory function to get the appropriate agent."""
    if agent_type == AgentType.COT:
        return ChainOfThoughtAgent(model_config, api_config, model_name)
    elif agent_type == AgentType.ROT:
        return ChainOfThoughtAgent(model_config, api_config, model_name)
    elif agent_type == AgentType.REFLEXION:
        return ReflexionAgent(model_config, api_config, model_name)
    elif agent_type == AgentType.REACT:
        return ReActAgent(model_config, api_config, model_name)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

def get_agent_instruction(agent_type: AgentType):
    """
        Get the instruction for the agent based on the agent type.
        This instruction is used in agents/prompt_template.py as the `agent_instruction` within the larger prompt.
    """
    if agent_type == AgentType.COT:
        return "\nLet's think step by step."
    elif agent_type == AgentType.ROT:
        return "\nLet's think step by step. Make sure a person can replicate the action input by only looking at the workflow and the action input reflects every step of the workflow."
    elif agent_type == AgentType.REACT:
        return "\nLet's think step by step."
    elif agent_type == AgentType.REFLEXION:
        return "\nLet's think step by step."
        
def get_agent_type(agent_type_str: str) -> AgentType:
    """Convert string representation to AgentType enum.
    
    Args:
        agent_type_str: String representation of agent type
        
    Returns:
        AgentType enum value
        
    Raises:
        ValueError: If agent_type_str doesn't match any enum value
    """
    try:
        return AgentType[agent_type_str]
    except KeyError:
        raise ValueError(f"Unknown agent type string: {agent_type_str}")
