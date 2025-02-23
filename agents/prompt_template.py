from langchain.prompts import StringPromptTemplate
from langchain.tools import BaseTool
from typing import List, Dict
from pydantic import Field

####################### Chain of Thought
cot_template = """You are a statistician trying to answer a question based on one or more datasets.

You have access to the following tools: 
{tool_description}

In your output, please strictly follow the format outlined below, maintaining the specified order and structure.
Question: the input question you must answer
Workflow: a plan to tackle the problem
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action in the format of (```python\s.*?```). Use print to show the results.
Observation: the result of performing the action with the action input (please do not generate)
Final Answer: an answer to the original question

NOTE: You will need to generate the complete action input in one code snippet to solve the query in one attempt. 
If no observation is provided, you need to generate the workflow, action, and action input. You don't need to provide the final answer. 
If an observation is provided, you should generate the answer starting with "Final Answer:"{agent_instruction}

Begin!

You need to load all datasets in python using the specified paths: 
{file_paths}

Dataset descriptions: 
{descriptions}

Question: 
{question}
{conversation}
"""

class CoTPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[BaseTool]

    def format(self, **kwargs: Dict) -> str:
        """
        Format the prompt based on the template type.

        Args:
            **kwargs (Dict): Input values for the variables.

        Returns:
            str: The formatted prompt.
        """
        # Get the conversation (action, action input, observation tuples)
        conversation = kwargs.pop("conversation")
        thoughts = ""
        for action, action_input, workflow, observation in conversation[-1:]:
            thoughts += f"Workflow: {workflow}\n\nAction: {action}\nAction input: \n{action_input}\n\nObservation: {observation}"
        # Set the conversation variable to that value
        kwargs["conversation"] = thoughts

        kwargs["tool_description"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)

####################### Reflexion (reproducibility)
reflexion_template = """You are a statistician trying to answer a question based on one or more datasets.

You have access to the following tools: 
{tool_description}

In your output, please strictly follow the format outlined below, maintaining the specified order and structure.
Question: the input question you must answer
Workflow: a plan to tackle the problem
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action in the format of (```python\s.*?```). Use print to show the results.
Observation: the result of performing the action with the action input (please do not generate)
Final Answer: an answer to the original question

NOTE: You will need to generate the complete action input in one code snippet to solve the query in one attempt. 
If no observation is provided, you need to generate the workflow, action, and action input. You don't need to provide the final answer. 
If an observation is provided, you should generate the answer starting with "Final Answer:"{agent_instruction}

Begin!

You need to load all datasets in python using the specified paths: 
{file_paths}

Dataset descriptions: 
{descriptions}

Question: 
{question}
{conversation}{reflexion}
"""

class ReflexionPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[BaseTool]

    def format(self, **kwargs: Dict) -> str:
        """
        Format the prompt based on the template type.

        Args:
            **kwargs (Dict): Input values for the variables.

        Returns:
            str: The formatted prompt.
        """
        # Get the conversation (action, action input, observation tuples)
        conversation = kwargs.pop("conversation")
        thoughts = ""
        for action, action_input, workflow, observation in conversation[-1:]:
            thoughts += f"Workflow: {workflow}\n\nAction: {action}\nAction input: \n{action_input}\n\n"
            if observation:
                thoughts += f"Observation: {observation}"
        # Set the conversation variable to that value
        kwargs["conversation"] = thoughts

        kwargs["tool_description"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)

####################### ReAct
react_template = """You are a statistician trying to answer a question based on one or more datasets.

You have access to the following tools: 
{tool_description}

Below is the structure of the agent-environment interaction. Your task is to generate only the agent's responses.
Question: the input question you must answer
<Agent>
Workflow: a plan to tackle the problem
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action in the format of (```python\s.*?```). Use print to show the results.
<Environment>
Observation: the result of performing the action with the action input (please do not generate)
<Agent>
Workflow: a plan to tackle the problem
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action in the format of (```python\s.*?```). Use print to show the results.
<Environment>
Observation: the result of performing the action with the action input (please do not generate)
... 
<Agent>
Final Answer: an answer to the original question
Task done!

NOTE: You will need to generate the complete action input in one code snippet. We will execute the code for you and provide the observation to you.
If no observation is provided or the observation is insufficient to answer the question, you need to (re)generate the complete workflow, action, and action input. Do not generate final answer and 'Task done'.
If the observation is sufficient to answer the question, generate the final answer starting with 'Final Answer:' and indicate completion by ending with 'Task done!'. Do not generate workflow any more.{agent_instruction}

Example
Question: 
What is the linear coefficient that describes the positive relationship between the rate of maximum body length evolution and spatial variation in speciation rates, where the former emerges as the most influential factor?
<Agent>
Workflow: First, I need to load the dataset 'body-size-evolution-in-south-american-freshwater-fishes.csv' into a Pandas DataFrame, and print the first few rows of the data.
Action: python_repl_ast
Action Input: ```python\nimport pandas as pd\n\ndf = pd.read_csv('body-size-evolution-in-south-american-freshwater-fishes.csv')\ndf.head()\n```
<Environment>
Observation: 
     HYBAS_ID       long       lat  BAMM_speciation  BAMM_extinction  BAMM_NetDiv  ...   aet    Elevation  sgr  soil_div     area  diversity
0  6050000010 -76.477422  7.742693         0.137392         0.026807     0.110585  ...  1387   330.150088  166  0.482402  72363.7         68
1  6050000740 -74.628725  9.803586         0.117235         0.025796     0.091438  ...  1082    69.475294   23  0.457436  17944.3         35
2  6050068100 -75.295995  8.448815         0.119381         0.023826     0.095555  ...  1312   143.032178   74  0.378793  17105.5         44
3  6050068110 -74.608408  8.922863         0.132477         0.027777     0.104700  ...  1445    14.724138    3  0.468328    610.1         48
4  6050070260 -75.591588  5.770093         0.120127         0.022940     0.097187  ...  1371  1378.729945  421  0.158870  61901.9         81
[5 rows x 21 columns]
<Agent>
Workflow: 1. Import Libraries: Import `LinearRegression` from `sklearn` and `numpy`.
2. Prepare Data: Extract predictor 'BAMM_speciation' and response 'BAMM_NetDiv'.
3. Create and Fit Model: Instantiate `LinearRegression()` and fit the model using `model.fit(X, y)`.
4. Retrieve Coefficient: Get the slope using `model.coef_[0]`.
5. Output Result: Print the coefficient with `print(coefficient)`.
Action: python_repl_ast
Action Input: ```python\nfrom sklearn.linear_model import LinearRegression\nimport numpy as np\n\n# Prepare the data for regression\nX = df[['BAMM_speciation']].values.reshape(-1, 1)  # Predictor\ny = df['BAMM_NetDiv'].values  # Response\n\n# Create a linear regression model\nmodel = LinearRegression()\nmodel.fit(X, y)\n\n# Get the coefficient\ncoefficient = model.coef_[0]\nprint(coefficient)\n```
<Environment>
Observation: 0.5175306498596297
<Agent>
Final Answer:
The linear coefficient that describes the positive relationship between the rate of maximum body length evolution ('BAMM_speciation') and spatial variation in speciation rates ('BAMM_NetDiv') is approximately 0.518.
Task done!

Begin!

You need to load all datasets in python using the specified paths: 
{file_paths}

Dataset descriptions: 
{descriptions}

Question: 
{question}
<Agent>
{conversation}
"""

class ReActPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[BaseTool]

    def format(self, **kwargs: Dict) -> str:
        """
        Format the prompt based on the template type.

        Args:
            **kwargs (Dict): Input values for the variables.

        Returns:
            str: The formatted prompt.
        """
        conversation = kwargs.pop("conversation")
        thoughts = ""
        for action, action_input, workflow, observation in conversation:
            thoughts += f"Workflow: {workflow}\n\nAction: {action}\nAction input: \n{action_input}\n\n<Environment>\nObservation: {observation}\n<Agent>\n"
        # Set the conversation variable to that value
        kwargs["conversation"] = thoughts

        kwargs["tool_description"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)

if __name__ == '__main__':
    pass
