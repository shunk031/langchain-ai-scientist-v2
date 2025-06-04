from typing import Final, Literal

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from .models import TreeBasedExperimentationState
from .nodes import (
    analyze_plots_with_vlm_node,
    create_draft_node,
    debug_code_node,
    define_global_metrics_node,
    extract_metrics_node,
    generate_ablation_node,
    generate_hparam_tuning_idea_node,
    generate_plotting_code_node,
    generate_substage_goal_node,
    improve_idea_node,
    parse_exec_result_node,
    perform_hparam_tuning_node,
)

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


def should_debug(state, config) -> Literal["debug-code", "parse-exec-result"]:
    raise NotImplementedError


# def should_analyze_plots(
#     state, config
# ) -> Literal[
#     "analyze-plots-with-vlm",
#     "stage 2 - generate-hparam-tuning-idea",
#     "stage 3 - improve-idea",
#     "stage 4 - generate-ablation",
# ]:
#     raise NotImplementedError


def create_tree_based_experimentation_pipeline():
    graph = (
        StateGraph(TreeBasedExperimentationState)
        #
        # -----
        # Nodes
        # -----
        #
        # Preliminary Investigation Nodes
        #
        .add_node("define-global-metrics", define_global_metrics_node)
        .add_node("stage 1 - create-draft", create_draft_node)
        .add_node("debug-code", debug_code_node)
        .add_node("parse-exec-result", parse_exec_result_node)
        .add_node("generate-plotting-code", generate_plotting_code_node)
        .add_node("extract-metrics", extract_metrics_node)
        .add_node("analyze-plots-with-vlm", analyze_plots_with_vlm_node)
        #
        # Hyperparameter Tuning Node
        #
        .add_node(
            "stage 2 - generate-hparam-tuning-idea", generate_hparam_tuning_idea_node
        )
        .add_node("stage 2 - perform-hparam-tuning", perform_hparam_tuning_node)
        #
        # Improvement Nodes
        #
        .add_node("stage 3 - improve-idea", improve_idea_node)
        #
        # Ablation Node
        #
        .add_node("stage 4 - generate-ablation", generate_ablation_node)
        #
        .add_node("generate-substage-goal", generate_substage_goal_node)
        #
        # -----
        # Edges
        # -----
        #
        .add_edge(START, "define-global-metrics")
        .add_edge("define-global-metrics", "stage 1 - create-draft")
        .add_conditional_edges("stage 1 - create-draft", path=should_debug)
        .add_edge("debug-code", "parse-exec-result")
        .add_edge("parse-exec-result", "extract-metrics")
        .add_edge("extract-metrics", "generate-plotting-code")
        # .add_conditional_edges("generate-plotting-code", should_analyze_plots)
        .add_edge("generate-plotting-code", "analyze-plots-with-vlm")
        .add_edge("analyze-plots-with-vlm", "stage 2 - generate-hparam-tuning-idea")
        .add_edge("analyze-plots-with-vlm", "stage 3 - improve-idea")
        .add_edge("analyze-plots-with-vlm", "stage 4 - generate-ablation")
        .add_edge(
            "stage 2 - generate-hparam-tuning-idea", "stage 2 - perform-hparam-tuning"
        )
        .add_edge("stage 2 - perform-hparam-tuning", "parse-exec-result")
        .add_edge("stage 3 - improve-idea", "parse-exec-result")
        .add_edge("stage 4 - generate-ablation", "parse-exec-result")
        .add_edge("stage 4 - generate-ablation", "generate-substage-goal")
        .add_edge("generate-substage-goal", "define-global-metrics")
        # .add_edge("parse-exec-result", "generate-substage-goal")
        .add_edge("stage 4 - generate-ablation", END)
        #
        # Compile the graph
        #
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
