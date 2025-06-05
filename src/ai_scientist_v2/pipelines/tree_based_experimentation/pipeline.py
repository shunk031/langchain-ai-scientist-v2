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
    select_generated_plots,
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


def should_debug(state, config) -> Literal["DebugCode", "ParseExecResult"]:
    raise NotImplementedError


# def should_analyze_plots(
#     state, config
# ) -> Literal[
#     "AnalyzePlotsWithVlm",
#     "Stg. 2 - GenerateHparamTuningIdea",
#     "Stg. 3 - ImproveIdea",
#     "Stg. 4 - generate-ablation",
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
        .add_node("DefineGlobalMetrics", define_global_metrics_node)
        .add_node("Stg. 1 - CreateDraft", create_draft_node)
        .add_node("DebugCode", debug_code_node)
        .add_node("ParseExecResult", parse_exec_result_node)
        .add_node("GeneratePlottingCode", generate_plotting_code_node)
        .add_node("ExtractMetrics", extract_metrics_node)
        .add_node("AnalyzePlotsWithVlm", analyze_plots_with_vlm_node)
        .add_node("SelectGeneratedPlots", select_generated_plots)
        #
        # Hyperparameter Tuning Node
        #
        .add_node("Stg. 2 - GenerateHparamTuningIdea", generate_hparam_tuning_idea_node)
        .add_node("Stg. 2 - PerformHparamTuning", perform_hparam_tuning_node)
        #
        # Improvement Nodes
        #
        .add_node("Stg. 3 - ImproveIdea", improve_idea_node)
        #
        # Ablation Node
        #
        .add_node("Stg. 4 - generate-ablation", generate_ablation_node)
        #
        .add_node("generate-substage-goal", generate_substage_goal_node)
        #
        # -----
        # Edges
        # -----
        #
        .add_edge(START, "DefineGlobalMetrics")
        .add_edge("DefineGlobalMetrics", "Stg. 1 - CreateDraft")
        .add_conditional_edges("Stg. 1 - CreateDraft", path=should_debug)
        .add_edge("DebugCode", "ParseExecResult")
        .add_edge("ParseExecResult", "ExtractMetrics")
        .add_edge("ExtractMetrics", "GeneratePlottingCode")
        # .add_conditional_edges("GeneratePlottingCode", should_analyze_plots)
        .add_edge("GeneratePlottingCode", "SelectPlots")
        .add_edge("SelectPlots", "AnalyzePlotsWithVlm")
        .add_edge("AnalyzePlotsWithVlm", "Stg. 2 - GenerateHparamTuningIdea")
        .add_edge("AnalyzePlotsWithVlm", "Stg. 3 - ImproveIdea")
        .add_edge("AnalyzePlotsWithVlm", "Stg. 4 - generate-ablation")
        .add_edge("Stg. 2 - GenerateHparamTuningIdea", "Stg. 2 - PerformHparamTuning")
        .add_edge("Stg. 2 - PerformHparamTuning", "ParseExecResult")
        .add_edge("Stg. 3 - ImproveIdea", "ParseExecResult")
        .add_edge("Stg. 4 - generate-ablation", "ParseExecResult")
        .add_edge("Stg. 4 - generate-ablation", "generate-substage-goal")
        .add_edge("generate-substage-goal", "DefineGlobalMetrics")
        # .add_edge("ParseExecResult", "generate-substage-goal")
        .add_edge("Stg. 4 - generate-ablation", END)
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
