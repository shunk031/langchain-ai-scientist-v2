from typing import Final

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import RunnableConfig

from ai_scientist_v2.config import BFTSConfig
from ai_scientist_v2.pipelines.tree_based_experimentation import (
    TreeBasedExperimentationState,
)

SYSTEM_PROMPT: Final[str] = """\
# Introduction

You are an AI researcher who is looking to publish a paper that will contribute significantly to the field. Your first task is to write a python code to implement a solid baseline based on your research idea provided below, from data preparation to model training, as well as evaluation and visualization. Focus on getting a simple but working implementation first, before any sophisticated improvements. We will explore more advanced variations in later stages.
"""

CREATE_DRAFT_PROMPT: Final[str] = """\
# Research Idea

{research_idea}

# Memory

{memory_summary}

# Instructions

## Response Format

Your response should be a brief outline/sketch of your proposed solution in natural language (7-10 sentences), followed by a single markdown code block (using the format ```python ... ```) which implements this solution and prints out the evaluation metric(s) if applicable. There should be no additional headings or text in your response. Just natural language text followed by a newline and then the markdown code block. Make sure to write concise code.

## Experiment Design Sketch Guideline

This first experiment design should be relatively simple, without extensive hyper-parameter optimization. Take the Memory section into consideration when proposing the design. The solution sketch should be 6-10 sentences. Don't suggest to do EDA. Make sure to create synthetic data if needed.

## Evaluation Metric(s)

{evaluation_metrics}

## Implementation Guidline

### General

CRITICAL GPU REQUIREMENTS - Your code MUST include ALL of these:

- At the start of your code, add these lines to handle GPU/CPU:

    ```python
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {{device}}')
    ```

- ALWAYS move models to device using the `.to(device)` method
- ALWAYS move input tensors to device using the `.to(device)` method
- ALWAYS move model related tensors to device using the `.to(device)` method
- For optimizers, create them AFTER moving model to device
- When using DataLoader, move batch tensors to device in training loop:

    ```python
    batch = {{k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}}
    ```

CRITICAL MODEL INPUT GUIDELINES:
- Always pay extra attention to the input to the model being properly normalized
- This is extremely important because the input to the model's forward pass directly affects the output, and the loss function is computed based on the output

### Synthetic Dataset

{synthetic_dataset_instructions}

### Generative Modeling

For generative modeling tasks, you must:

- Generate a set of samples from your model
- Compare these samples with ground truth data using appropriate visualizations
- When saving plots, always use the 'working_dir' variable that will be defined at the start of the script
- Make sure to give each figure a unique and appropriate name based on the dataset it represents, rather than reusing the same filename.

### Important Code Structure Requirements

- Do NOT put any execution code inside 'if __name__ == "__main__":' block
- All code should be at the global scope or in functions that are called from the global scope
- The script should execute immediately when run, without requiring any special entry point

The code should start with:

```python
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
```

The code should be a single-file python program that is self-contained and can be executed as-is. No parts of the code should be skipped, don't terminate the code execution before finishing the script. Your response should only contain a single code block. Be aware of the running time of the code, it should complete within {timeout}. You can also use the "./working" directory to store any temporary files that your code needs to create.

### Data Saving Requirements

- Save all plottable data (metrics, losses, predictions, etc.) as numpy arrays using np.save()
- Use the following naming convention for saved files:

    ```python
    # At the start of your code
    experiment_data = {{
        "dataset_name_1": {{
            "metrics": {{"train": [], "val": []}},
            "losses": {{"train": [], "val": []}},
            "predictions": [],
            "ground_truth": [],
            # Add other relevant data
        }},
        # Add additional datasets as needed:
        "dataset_name_2": {{
            "metrics": {{"train": [], "val": []}},
            "losses": {{"train": [], "val": []}},
            "predictions": [],
            "ground_truth": [],
            # Add other relevant data
        }},
    }}
    # During training/evaluation:
    experiment_data["dataset_name_1"]["metrics"]["train"].append(train_metric)
    experiment_data["dataset_name_1"]["metrics"]["val"].append(val_metric)
    ```
    
- Include timestamps or epochs with the saved metrics
- For large datasets, consider saving in chunks or using np.savez_compressed()

CRITICAL EVALUATION REQUIREMENTS - Your code MUST include ALL of these:

1. Track and print validation loss at each epoch or at suitable intervals:

    ```python
    print(f"Epoch {{epoch}}: validation_loss = {{val_loss:.4f}}")
    ```

2. Track and update ALL these additional metrics: {evaluation_metrics}
3. Update metrics at EACH epoch:
4. Save ALL metrics at the end:

    ```python
    np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
    ```

### K-Fold Cross-Validation

{k_fold_cross_validation_instructions}

## Installed Packages

Your solution can use any relevant machine learning packages such as: {pkg_str}. Feel free to use any other packages too (all packages are already installed!). For neural networks we suggest using PyTorch rather than TensorFlow.

## Data Overview

{data_overview}
"""

SYNTHETIC_DATASET_INSTRUCTIONS: Final[str] = """\
You MUST evaluate your solution on at least {num_syn_datasets} different synthetic datasets to ensure robustness:

- Use standard benchmark datasets when available
- If using synthetic data, generate at least {num_syn_datasets} variants with different characteristics
- Report metrics separately for each dataset
- Compute and report the average metric across all datasets
"""

K_FOLD_CROSS_VALIDATION_INSTRUCTIONS: Final[str] = """\
The evaluation should be based on {k_fold_validation}-fold cross-validation but only if that's an appropriate evaluation for the task at hand.
"""


def create_draft_node(state: TreeBasedExperimentationState, config: RunnableConfig):
    conf = BFTSConfig()

    llm = init_chat_model(
        model=conf.agent.code.model_name,
        temperature=conf.agent.code.temperature,
    )

    num_syn_datasets = conf.experiment.num_syn_datasets
    synthetic_dataset_instructions = (
        SYNTHETIC_DATASET_INSTRUCTIONS.format(num_syn_datasets=num_syn_datasets)
        if num_syn_datasets > 1
        else "No specific instructions for synthetic datasets are provided."
    )

    k_fold_cv = conf.agent.k_fold_validation
    kfold_cv_instructions = (
        K_FOLD_CROSS_VALIDATION_INSTRUCTIONS.format(k_fold_validation=k_fold_cv)
        if k_fold_cv > 1
        else "No specific instructions for k-fold cross-validation are provided."
    )

    system_prompt = SystemMessage(SYSTEM_PROMPT)
    human_prompt = HumanMessagePromptTemplate.from_template(CREATE_DRAFT_PROMPT)
    prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
    chain = prompt | llm

    breakpoint()

    input_dict = {
        "research_idea": config["configurable"]["idea"][0]["idea"],
        "memory_summary": state.memory_summary,
        "evaluation_metrics": state.evaluation_metrics.to_string(),
        "synthetic_dataset_instructions": synthetic_dataset_instructions,
        "k_fold_cross_validation_instructions": kfold_cv_instructions,
        "timeout": conf.agent.code.timeout,
        "pkg_str": ", ".join(conf.agent.code.packages),
        "data_overview": state.data_overview,
    }
    output = chain.invoke(input_dict)

    breakpoint()
