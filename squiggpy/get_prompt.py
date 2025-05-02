import sys

user_prompt = sys.argv[1]

# prompt = f"""Your task is to build models for complex and open ended questions.
# You have a few tools available to you that you should use in this process:
# - web search: use this to do a literature review or to find up-to-date information
# - jupyter: use jupyter to run analysis and build quantitative models

# Start with a conceptual analysis and research how to best model to problem before you start quantitative modeling.

# Your final output should be a jupyter notebook with a probabilistic squiggpy model and background research in markdown blocks. Cite sources, state assumptions and motivate your choices. Prioritize thoroughness and correctness over speed - you can work for hours if needed, and you can do hundreds of web searches or jupyter runs.

# Squiggpy models decompose estimation problems into smaller subproblems. Often, those subproblems are also estimation problems. You can build high-level models first and then refine them based on additional (web) research, and/or build nested models - you can go pretty deep here.

# Here are the quiggpy docs:
# <squiggpy.md>
# {open('README.md').read()}
# </squiggpy.md>

# The task for today is: "{user_prompt}"
# """

prompt = f"""Your task is to build models for complex and open ended questions.

Start with a conceptual analysis and research how to best model to problem before you start quantitative modeling.

Your final output should be a python script with a probabilistic squiggpy model and background research in extensive comments. Cite sources, state assumptions and motivate your choices. Prioritize thoroughness and correctness over speed - you can work for hours if needed, and you can do hundreds of web searches or jupyter runs. The script will be run by the user in an environment that has squiggpy installed.

Squiggpy models decompose estimation problems into smaller subproblems. Often, those subproblems are also estimation problems. You can build high-level models first and then refine them based on additional (web) research, and/or build nested models - you can go pretty deep here.

Here are the quiggpy docs:
<squiggpy.md>
{open('README.md').read()}
</squiggpy.md>

The task for today is: "{user_prompt}"
"""

print(prompt)