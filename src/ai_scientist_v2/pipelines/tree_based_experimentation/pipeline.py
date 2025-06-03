from typing import Final

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from .models import TreeBasedExperimentationState
from .nodes import define_global_metrics_node

STAGE1_GOAL: Final[str] = """\
- Focus on getting basic working implementation
- Use a simple dataset
- Aim for basic functional correctness
- If you are given \"Code To Use\", you can directly use it as a starting point."""

STAGE2_GOAL: Final[str] = """\
- Change hyperparameters such as learning rate, number of epochs, batch size, etc. to improve the performance
- DO NOT change the model architecture from the previous stage
- Introduce TWO more new datasets from HuggingFace test the model. Try very hard to think what Huggingface datasets can be used here for testing."""

STAGE3_GOAL: Final[str] = """\
- Explore novel improvements
- Come up with experiments to reveal new insights
- Be creative and think outside the box
- MAKE SURE you use THREE HuggingFace dataset in total to test your models."""

STAGE4_GOAL: Final[str] = """\
- Conduct systematic component analysis that reveals the contribution of each part
- Use the same datasets you used from the previous stage"""


def create_tree_based_experimentation_pipeline():
    graph = (
        StateGraph(TreeBasedExperimentationState)
        .add_node(node="define-global-metrics", action=define_global_metrics_node)
        .add_edge(start_key=START, end_key="define-global-metrics")
        .compile()
    )
    return graph


def run_tree_based_experimentation_pipeline(
    idea, code_to_use: str = "No code is provided for this stage."
):
    graph = create_tree_based_experimentation_pipeline()
    output = graph.invoke(
        {
            "idea": idea,
            "main_stage_name": "Initial Implementation",
            "sub_stage_name": "Preliminary Investigation",
            "sub_stage_num": 1,
            "sub_stage_goals": STAGE1_GOAL,
            "code_to_use": code_to_use,
        },
        config={
            "configurable": {
                "idea": idea,
            }
        },
    )
    return output
