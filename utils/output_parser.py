from langchain.agents import AgentOutputParser
from typing import Dict, Any, Optional, Union
import re

def extract_python_code(text):
    """
    Extract Python code from a string that contains code blocks marked with ```python
    
    Args:
        text (str): Input text containing Python code blocks
        
    Returns:
        str: Extracted Python code without the markdown markers
    """
    # Pattern to match Python code blocks
    # Matches anything between ```python and ``` 
    pattern = r'```python\n(.*?)```'
    
    # Find all matches in the text using re.DOTALL flag to match across lines
    matches = re.findall(pattern, text, re.DOTALL)
    
    # Return the first match if found, otherwise return empty string
    return matches[0] if matches else ''

class CoTOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Dict[str, Any]:
        """
        Parses the LLM output and extracts action, action_input, workflow, and final_answer.
        """
        llm_output = llm_output.strip()
        
        # Internal function to parse action and action_input
        def parse_action_and_input(output):
            """
            pattern = re.compile(
                r'(?i)'                                   # Case-insensitive
                r'(?:###\s*)?Action:\s*(.*?)\s*'              # Group 1: Action
                r'(?:###\s*)?Action\s*Input:\s*(```python.*?```)', # Group 2: Action Input
                re.DOTALL
            )
            """
            pattern = re.compile(r'(```python\s.*?```)', re.DOTALL)
            match = pattern.search(output)
            if match:
                #return match.group(1).strip(), match.group(2).strip()
                return 'python_repl_ast', match.group(1).strip()
            return '', ''

        # Internal function to parse workflow
        def parse_workflow(output):
            pattern = re.compile(
                r'(?i)'                     # Case-insensitive
                r'(?:###\s*)?Workflow:\s*(.*?)'
                r'(?=\n\nAction|\n\n```python|\nAction|\n```python|\n\n###|\n###|$)',   
                re.DOTALL
            )
            match = pattern.search(output)
            if match:
                return match.group(1).strip()
            return ''

        # Internal function to parse final answer
        def parse_final_answer(output):
            final_answer_match = re.search(
                r'(?i)'
                r'(?:###\s*)?Final Answer:?\s*(.*)$',
                output,
                re.DOTALL
            )
            if final_answer_match:
                return final_answer_match.group(1).strip()
            return None

        # Parse individual components
        action, action_input = parse_action_and_input(llm_output)
        workflow = parse_workflow(llm_output)
        final_answer = parse_final_answer(llm_output)

        # Return results as a dictionary
        return {
            "action": action,
            "action_input": action_input,
            "workflow": workflow,
            "final_answer": final_answer
        }

class ReActOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Dict[str, Any]:
        """
        Parses the LLM output and extracts action, action_input, workflow, and final_answer.
        """
        llm_output = llm_output.strip()
        
        # Internal function to parse action and action_input
        def parse_action_and_input(output):
            """
            pattern = re.compile(
                r'(?i)'                                   # Case-insensitive
                r'(?:###\s*)?Action:\s*(.*?)\s*'              # Group 1: Action
                r'(?:###\s*)?Action\s*Input:\s*(```python.*?```)', # Group 2: Action Input
                re.DOTALL
            )
            """
            pattern = re.compile(r'(```python\s.*?```)', re.DOTALL)
            match = pattern.search(output)
            if match:
                #return match.group(1).strip(), match.group(2).strip()
                return 'python_repl_ast', match.group(1).strip()
            return '', ''

        # Internal function to parse workflow
        def parse_workflow(output):
            pattern = re.compile(
                r'(?i)'                     # Case-insensitive
                r'(?:###\s*)?Workflow:\s*(.*?)'
                r'(?=\n\nAction|\n\n```python|\nAction|\n```python|\n\n###|\n###|$)',   
                re.DOTALL
            )
            match = pattern.search(output)
            if match:
                return match.group(1).strip()
            return ''

        # Internal function to parse final answer
        def parse_final_answer(output):
            final_answer_match = re.search(
                r'(?i)'
                r'(?:###\s*)?Final Answer:?\s*(.*)$',
                output,
                re.DOTALL
            )
            if final_answer_match:
                return final_answer_match.group(1).strip()
            return None

        # Parse individual components
        action, action_input = parse_action_and_input(llm_output)
        workflow = parse_workflow(llm_output)
        final_answer = parse_final_answer(llm_output)
        if final_answer is not None and 'Task done' in final_answer: 
            final_answer = final_answer[:final_answer.find('Task done')]
        task_done_flag = 'Task done' in llm_output

        # Return results as a dictionary
        return {
            "action": action,
            "action_input": action_input,
            "workflow": workflow,
            "final_answer": final_answer,
            "task_done_flag": task_done_flag
        }

class ScoreParser:
    def __init__(self):
        self.accuracy_score_prefix = "The accuracy score is:"
        self.similarity_score_prefix = "The similarity score is:"
        self.thoughts_prefix = "Thoughts:"

    def extract_accuracy_score(self, text):
        start_pos = text.find(self.accuracy_score_prefix)
        if start_pos == -1:
            print(text)
            return 0
            
        # Extract the substring after the prefix
        score_text = text[start_pos + len(self.accuracy_score_prefix):].strip()
        
        # Find the first number in the remaining text
        match = re.search(r'-?\d*\.?\d+', score_text)
        if match:
            return float(match.group())
        
        print(text)
        return 0

    def extract_similarity_score_and_category(self, text):
        # Initialize default values
        thoughts = ""
        score = -1
        
        # Find thoughts section
        thoughts_start = text.find(self.thoughts_prefix)
        if thoughts_start != -1:
            thoughts_start += len(self.thoughts_prefix)
            thoughts_end = text.find(self.similarity_score_prefix, thoughts_start)
            if thoughts_end != -1:
                thoughts = text[thoughts_start:thoughts_end].strip()
            else:
                thoughts = text[thoughts_start:].strip()
        
        # Extract score
        score_start = text.find(self.similarity_score_prefix)
        if score_start != -1:
            score_text = text[score_start + len(self.similarity_score_prefix):].strip()
            score_match = re.search(r'-?\d*\.?\d+', score_text)
            if score_match:
                score = float(score_match.group())
        
        if score == -1:
            print(text)
        return score, thoughts
