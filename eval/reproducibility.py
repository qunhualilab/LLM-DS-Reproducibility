from typing import List
import openai
import os
import re
from .eval_prompt import accuracy_prompt, workflow_to_code_prompt, llm_reproducibility_prompt
from utils.output_parser import ScoreParser, extract_python_code
from langchain_openai import AzureChatOpenAI
from utils.code_execution import CustomPythonAstREPLTool

class Reproducibility:
    def __init__(self):
        self.llm = AzureChatOpenAI(azure_deployment='gpt-4o-2024-11-20', api_version="2024-10-01-preview", temperature=0)
        self.parser = ScoreParser()
        self.original_workflows = []
        self.converted_workflows = []
        self.original_codes = []
        self.converted_codes = []
        self.human_prompts = []
        self.input_tokens = 0
        self.output_tokens = 0

    def add_up_tokens(self, response):
        self.input_tokens += response.usage_metadata['input_tokens']
        self.output_tokens += response.usage_metadata['output_tokens']

    def accuracy(self, question, predicted_answer, true_answer, dataset_specific_prompt=''):
        human_prompt = accuracy_prompt.format(
            question=question,
            dataset_specific_prompt=dataset_specific_prompt,
            predicted_answer=predicted_answer,
            true_answer=true_answer)
        try:
            messages = [
                ("system", "You are a data scientist grading the accuracy of a predicted answer to a question."),
                ("human", human_prompt),
                ]
            response = self.llm.invoke(messages)
            #print(response.content)
            self.add_up_tokens(response)
            score = self.parser.extract_accuracy_score(response.content)
            return score
        except openai.BadRequestError as e:
            print('The prompt triggers Azure OpenAI\'s content management policy. Please manually check the accuracy.')
            return 1

    def workflow_to_code(self, question: str, datasets: List, workflow: str):
        human_prompt = workflow_to_code_prompt.format(
            question=question,
            datasets=datasets,
            workflow=workflow
        )
        #print(human_prompt)
        messages = [
            ("system", "You are a data scientist translating a workflow into code."),
            ("human", human_prompt),
            ]
        response = self.llm.invoke(messages)
        self.add_up_tokens(response)
        extracted_code = extract_python_code(response.content)
        self.converted_codes.append(extracted_code)
        return extracted_code

    def generate_conclusion(self, question: str, code: str, code_output: str):
        messages = [
            ("system", "You are a data scientist answering a question based on the code and code output."),
            ("human", f"Question: {question}\n\nCode:\n{code}\n\nCode output: {code_output}\n\nPlease generate the answer based only on the code output in this format: \n'Thought:\n\nConclusion:'"),
            ]
        response = self.llm.invoke(messages)
        self.add_up_tokens(response)
        pattern = re.compile(r'Conclusion:\s*(.*)', re.DOTALL)
        match = pattern.search(response.content)
        if match:
            return match.group(1).strip()  # Remove any trailing/leading spaces/newlines
        return response.content

    def llm_reproducibility(self, sample, code: str, workflow: str, final_answer: str=''):
        change_dir = os.path.dirname(sample.file_paths[0])
        self.original_codes.append(code)
        if extract_python_code(code):
            code_1 = extract_python_code(code)
        else:
            code_1 = code
        code_1_output = CustomPythonAstREPLTool()._run(code_1, change_dir=change_dir)
        if 'Error [' in code_1_output:
            self.human_prompts.append("")
            self.converted_codes.append("")
            return -1, 'Code not run'
        if code_1_output.find('You cannot generate code anymore.') != -1:
            code_1_output = code_1_output[:code_1_output.find('You cannot generate code anymore.')]
        question = sample.question
        # code_1_conclusion = self.generate_conclusion(question=question, code=code_1, code_output=code_1_output)
        code_1_conclusion = final_answer

        code_2 = self.workflow_to_code(
            question=question,
            datasets=[os.path.basename(file_path) for file_path in sample.file_paths],
            workflow=workflow
        )
        if extract_python_code(code_2):
            code_2 = extract_python_code(code_2)
        else:
            code_2 = code_2
        code_2_output = CustomPythonAstREPLTool()._run(code_2, change_dir=change_dir)
        if code_2_output.find('You cannot generate code anymore.') != -1:
            code_2_output = code_2_output[:code_2_output.find('You cannot generate code anymore.')]
        code_2_conclusion = self.generate_conclusion(question=question, code=code_2, code_output=code_2_output)
        
        human_prompt = llm_reproducibility_prompt.format(
            question = question,
            code_1 = code_1,
            code_1_output = code_1_output,
            code_1_conclusion = code_1_conclusion,
            code_2 = code_2,
            code_2_output = code_2_output,
            code_2_conclusion = code_2_conclusion,
        )
        self.human_prompts.append(human_prompt)
        messages = [
            #("system", "You are a data scientist analyzing the functional similarity between code chunks."),
            ("human", human_prompt),
            ]
        response = self.llm.invoke(messages)
        self.add_up_tokens(response)
        score, reason = self.parser.extract_similarity_score_and_category(response.content)
        return score, reason
