from typing import Dict, Any
import os
import json
from langchain_ollama import ChatOllama
from langchain_aws import ChatBedrock
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from utils.output_parser import ReActOutputParser, extract_python_code, CoTOutputParser
from utils.code_execution import CustomPythonAstREPLTool
from utils.data_class import DataSample
from utils.prepare_prompt import prepare_prompt
from .prompt_template import react_template, ReActPromptTemplate
from .base_agent import BaseAgent

class ReActAgent(BaseAgent):
    """ReAct agent implementation."""

    def __init__(self, model_config: str, api_config: str, model_name: str):
        """
        Initialize the BaseAgent with configuration files and the model name.

        Args:
            model_config (str): Path to the model configuration file.
            api_config (str): Path to the API keys configuration file.
            model_name (str): Name of the model to initialize.
        """
        self.model_config = self._load_config(model_config, "model configuration")
        self.api_config = self._load_config(api_config, "API configuration")

        if model_name not in self.model_config:
            raise ValueError(f"Model name '{model_name}' not found in the configuration.")

        self.model_name = self.model_config[model_name]['model_name']
        self.model_type = self.model_config[model_name]['model_type']
        self.api_key = self.api_config.get(self.model_type, '')

        self.llm = self.get_model()
        self.parser = ReActOutputParser()
        self.python_repl = CustomPythonAstREPLTool(max_runs=3)
        self.prompt = ReActPromptTemplate(
            template=react_template, 
            tools=[self.python_repl], 
            input_variables=["file_paths", "descriptions", "question"])
        self.history = []

    def _load_config(self, path: str, config_name: str) -> dict:
        """Load a JSON configuration file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"The {config_name} file was not found at path: {path}")
        
        with open(path, 'r') as f:
            return json.load(f)

    def get_model(self, temperature=0):
        """Initialize the appropriate LLM model based on the configuration."""
        if self.model_type == 'deepseek':
            return ChatOllama(model=self.model_name, base_url=self.api_key, temperature=temperature, num_ctx=8196*2, num_predict=6000)
        elif self.model_type == 'meta':
            return ChatOllama(
                model=self.model_name, 
                base_url=self.api_key, 
                temperature=temperature, 
                repetition_penalty=1.18, 
                num_ctx=8196*2, 
                num_predict=2048
            )
        elif self.model_type == 'anthropic':
            return ChatBedrock(
                model_id=self.model_name, 
                model_kwargs=dict(temperature=temperature), 
                region_name='us-east-1',
                max_tokens=2048
            )
        elif self.model_type == 'openai':
            return AzureChatOpenAI(
                azure_deployment=self.model_name, 
                api_version="2024-10-01-preview", 
                temperature=temperature,
                max_tokens=2048
            )
        elif self.model_type == 'openai-o':
            return AzureChatOpenAI(azure_deployment='o3-mini-2025-01-31', api_version="2024-12-01-preview", temperature=1, reasoning_effort='low', max_completion_tokens=4000)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def run(self, sample: DataSample, agent_instruction: str = "\nLet's think step by step.", max_steps: int = 4, deepseek=False) -> Dict[str, Any]:
        # The maximum number of max_steps is determined by the allowable number of Python code executions.
        change_dir = os.path.dirname(sample.file_paths[0])
        current_input = prepare_prompt(sample)
        current_input.update({"agent_instruction": agent_instruction})
        if 'conversation' not in current_input:
            current_input['conversation'] = []
        steps_taken = 0

        while steps_taken <= max_steps:
            prompt_result = self.prompt.format(**current_input).strip()
            llm_output = self.llm.invoke([('human', prompt_result)])
            try:
                parsed_output = self.parser.parse(llm_output.content)
            except Exception as e:
                return self._format_result()

            parsed_output.update({
                "content": llm_output.content, 
                "usage_metadata": llm_output.usage_metadata})
            self.history.append(parsed_output)

            if (
                (parsed_output.get("final_answer") and steps_taken >= 1 and parsed_output.get("task_done_flag"))
                or not parsed_output.get("action_input")
                or not parsed_output.get("action")
            ):
                return self._format_result()
            action = parsed_output.get('action')
            action_input = parsed_output.get('action_input')
            workflow = parsed_output.get('workflow')
            if deepseek:
                workflow = llm_output.content[:llm_output.content.find('```python')]
            if action != self.python_repl.name or not action_input:
                raise ValueError(f"Invalid action or action_input: action={action}, action_input={action_input}")

            observation = self.python_repl._run(extract_python_code(action_input), change_dir=change_dir)
            if steps_taken == max_steps - 3:
                observation = observation + '\nYou have one last chance to regenerate the workflow, action, and action input for the complete analysis.'
            self.history[-1].update({'observation': observation})
            current_input['conversation'].append((action, action_input, workflow, observation))
            steps_taken += 1

        return self._format_result(max_steps_reached=True)

    def _format_result(self, max_steps_reached: bool = False, error: str = None) -> Dict[str, Any]:
        """Format the final result of the agent's execution."""
        steps = []
        final_answer, workflow = None, None
        
        for step in self.history:
            if 'action' in step:
                step_info = {
                    "action": step["action"] if step["action"] else "",
                    "action_input": step["action_input"] if step["action_input"] else "",
                    "workflow": step["workflow"] if step["workflow"] else "",
                    "observation": step.get("observation"),
                    "content": step.get("content"),
                    "usage_metadata": step.get("usage_metadata")
                }
                steps.append(step_info)
            if step.get("final_answer"):
                final_answer = step["final_answer"]
                if len(steps) > 0:
                    steps[-1].update({"final_answer": final_answer})
                
        
        return {
            "steps": steps,
            "final_answer": final_answer or "No final answer reached",
            "max_steps_reached": max_steps_reached
        }

    
