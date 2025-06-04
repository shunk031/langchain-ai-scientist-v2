from typing import Final

SYSTEM_PROMPT: Final[str] = """\
# Introduction
You are an experienced AI researcher. You are provided with a previously developed baseline implementation. Your task is to implement hyperparameter tuning for the following idea:

{hyperprameter_idea_name}

{hyperparameter_idea_description}

# Base Code

Base code you are working on:

```python
{base_code}
```

# Instructions

## Implementation Guidline

The code should be a single-file python program that is self-contained and can be executed as-is. No parts of the code should be skipped, don't terminate the code execution before finishing the script.

Data saving requirements:

- Save all plottable data (metrics, losses, predictions, etc.) as numpy arrays using np.save()
- Use the following naming convention for saved files:

```python
# At the start of your code
experiment_data = {
    "hyperparam_tuning_type_1": {
        "dataset_name_1": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            # Add other relevant data
        },
        # Add additional datasets as needed:
    },
    # Add additional hyperparam tuning types as needed
}
```
  
Make sure to use a filename 'experiment_data.npy' to save the data. Do not use any other filename.
"""


def generate_hparam_tuning_idea_node(state, config):
    raise NotImplementedError


def perform_hparam_tuning_node(state, config):
    conf = AgentConfig.from_runnable_config(config)

    llm = init_chat_model(
        model=conf.feedback.model_name,
        temperature=conf.feedback.temperature,
    )
    system_prompt = SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT)

    human_prompt = HumanMessagePromptTemplate.from_template(
        "Please implement the hyperparameter tuning code."
    )

    prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

    response = llm.invoke(
        prompt.format_messages(
            hyperprameter_idea_name=state.hyperparameter_idea.name,
            hyperparameter_idea_description=state.hyperparameter_idea.description,
            base_code=state.base_code,
        )
    )

    state.hyperparameter_tuning_code = response.content

    raise NotImplementedError
