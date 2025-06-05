from typing import Final

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.runnables import RunnableConfig

from ai_scientist_v2.pipelines.tree_based_experimentation import (
    EvaluationMetrics,
    TreeBasedExperimentationState,
)

SYSTEM_PROMPT: Final[str] = """\
# Instructions

## Response Format

Your response should be a brief outline/sketch of your proposed solution in natural language (7-10 sentences), followed by a single markdown code block (using the format ```python ... ```) which implements this solution and prints out the evaluation metric(s) if applicable. There should be no additional headings or text in your response. Just natural language text followed by a newline and then the markdown code block. Make sure to write concise code.

## Plotting Code Guideline

### Abaliable Data

- Experiment data: `experiment_data.npy`

### Requirements

- The code should start with:

    ```python
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    working_dir = os.path.join(os.getcwd(), 'working')
    ```

- Create standard visualizations of experiment results
- Save all plots to `working_dir`
- Include training/validation curves if available
- ONLY plot data that exists in experiment_data.npy - DO NOT make up or simulate any values
- Use basic matplotlib without custom styles
- Each plot should be in a separate try-except block
- Always close figures after saving
- Always include a title for each plot, and be sure to use clear subtitles—such as 'Left: Ground Truth, Right: Generated Samples' —- while also specifying the type of dataset being used.
- Make sure to use descriptive names for figures when saving e.g. always include the dataset name and the type of plot in the name.
- When there are many similar figures to plot (e.g. generated samples at each epoch), make sure to plot only at a suitable interval of epochs so that you only plot at most 5 figures.
- Use the following experiment code to infer the data to plot

    ```python
    {code}
    ```

- Example to extract data from experiment_data: `experiment_data["dataset_name_1"]["metrics"]["train"]`

### Example Data Loading and Plot Saving Code

```python
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {{e}}")

try:
    # First plot
    plt.figure()
    # ... plotting code ...
    plt.savefig("working_dir/[plot_name_1].png")
    plt.close()
except Exception as e:
    print(f"Error creating plot1: {{e}}")
    plt.close()  # Always close figure even if error occurs

try:
    # Second plot
    plt.figure()
    # ... plotting code ...
    plt.savefig("working_dir/[plot_name_2].png")
    plt.close()
except Exception as e:
    print(f"Error creating plot2: {{e}}")
    plt.close()
```
"""

STAGE3_GUIDELINE: Final[str] = """\
# IMPORTANT

Use the following base plotting code as a starting point.

Base plotting code:  

{plot_code_from_prev_stage}

Modify the base plotting code to:

1. Keep the same numpy data structure and plotting style
2. Add comparison plots between different datasets
3. Add dataset-specific visualizations if needed
4. Include clear labels indicating which plots are from which dataset
5. Use consistent naming conventions for saved files
"""

STAGE4_GUIDELINE: Final[str] = """\
# IMPORTANT

This is an ablation study. Use the following base plotting code as a starting point.

Base plotting code: 

{plot_code_from_prev_stage}

Modify the base plotting code to:

1. Keep the same numpy data structure and plotting style
2. Add comparison plots between ablation and baseline results
3. Add ablation-specific visualizations if needed
4. Include clear labels indicating which plots are from ablation vs baseline
5. Use consistent naming conventions for saved files
"""


def generate_plotting_code_node(
    state: TreeBasedExperimentationState, config: RunnableConfig
):
    raise NotImplementedError
