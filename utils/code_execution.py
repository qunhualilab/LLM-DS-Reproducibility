import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import resample
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests
from scipy import stats
from scipy.stats import pearsonr, ttest_ind, norm
from sklearn.preprocessing import LabelEncoder
from statsmodels.formula.api import logit, ols
from statsmodels.stats.anova import anova_lm
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from scipy.stats import chi2_contingency
import seaborn as sns
from statsmodels.regression.linear_model import OLS
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import linregress
from econml.dml import DML
from statsmodels.stats.proportion import proportion_confint
from statsmodels.tsa.stattools import grangercausalitytests
import statsmodels.api as sm
from scipy.stats import bootstrap
from scipy.stats import chisquare
from sklearn.neighbors import NearestNeighbors
from pingouin import partial_corr
import pingouin as pg
from Bio import Phylo

import io
import os
import sys
from contextlib import redirect_stdout
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from pydantic import Field

class CustomPythonAstREPLTool(PythonAstREPLTool):
    """Custom Python REPL tool that limits the number of code executions."""
    
    max_runs: int = Field(default=1, description="Maximum number of allowed code executions")
    max_turns: int = Field(default=0, description="Current number of code executions")
    reach_limit_message: str = Field(default='', description="Warning message of reaching tool use limitation.")

    def __init__(self, max_runs: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.max_runs = max_runs
        self.max_turns = 0
        self.reach_limit_message = (
            "You cannot generate code anymore. "
            "Please provide the final answer directly "
            "based on the information you have so far."
        )


    def _run(self, query: str, change_dir: str = None) -> str:
        """
        Override _run to capture print statements as well as the final evaluated result.
        """
        if change_dir:
            import os
            os.chdir(change_dir)

        # Initialize a stream to capture output
        output_stream = io.StringIO()
        self.max_turns += 1

        # Define custom exit and quit functions
        def fake_exit(*args):
            print("Intercepted exit/quit call.")

        # Redirect stdout to capture print statements
        with redirect_stdout(output_stream):
            try:
                # Use exec to execute the query, as it allows print outputs
                shared_namespace = {"exit": fake_exit, "quit": fake_exit}
                exec(query, shared_namespace, shared_namespace)
                result = ""  # No errors, so no explicit result
            except Exception as e:
                # Capture errors
                error_category = type(e).__name__
                result = f"Error [{error_category}]: {str(e)}"

        # Combine print output and any explicit result
        captured_output = output_stream.getvalue()
        output_stream.close()
        final_output = (captured_output + result).strip()
        
        if self.max_turns >= self.max_runs:
            final_output += '\n' + self.reach_limit_message

        return final_output

if __name__ == '__main__':
    pass
