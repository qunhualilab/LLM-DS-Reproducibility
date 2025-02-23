accuracy_prompt = """Evaluate the correctness (0 for incorrect, 1 for correct) of the predicted answer to the question: {question}
{dataset_specific_prompt}
Predicted answer: {predicted_answer}

Ground truth answer: {true_answer}

Please reply in this format: 
"Thoughts: 

The accuracy score is:"
"""

workflow_to_code_prompt = """Question: {question}
These are the dataset paths: {datasets}

Develop a Python script that precisely converts the provided narrative summary into executable code. Ensure that each component of the analysis process is correctly implemented, closely following the steps outlined in the summary. Maintain consistency by using the exact variable names specified in the narrative. Below is the code summary to translate:
{workflow}"""

llm_reproducibility_prompt = """Your task is to determine whether Code 1 and Code 2 arrive at the same conclusion regarding the question: {question}

Code 1:
{code_1}

Code 1 execution output:
{code_1_output}

Code 1 conclusion:
{code_1_conclusion}

Code 2:
{code_2}

Code 2 execution output:
{code_2_output}

Code 2 conclusion:
{code_2_conclusion}

If the output of the two code snippets provide the same values for the same statistics and lead to the same conclusion to the question, please score 1. Otherwise, score 0. Note that if code 2 runs into error, please score 0.
Please reply in this format: 
"Thoughts: 

The similarity score is:"
"""