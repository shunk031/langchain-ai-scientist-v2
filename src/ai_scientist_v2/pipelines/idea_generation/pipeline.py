from typing import cast

from langgraph.graph import END, START, StateGraph

from .nodes import (
    IdeaGenerationState,
    idea_finalization_node,
    idea_generation_node,
    idea_reflection_node,
    should_continue,
)


def create_idea_generation_pipeline():
    graph = (
        StateGraph(IdeaGenerationState)
        .add_node(node="idea-generation", action=idea_generation_node)
        .add_node(node="idea-reflection", action=idea_reflection_node)
        .add_node(node="idea-finalization", action=idea_finalization_node)
        .add_edge(start_key=START, end_key="idea-generation")
        .add_edge(start_key="idea-generation", end_key="idea-reflection")
        .add_conditional_edges("idea-reflection", path=should_continue)
        # .add_edge(start_key="idea-reflection", end_key="idea-finalization")
        .add_edge(start_key="idea-finalization", end_key=END)
        .compile()
    )
    return graph


def run_idea_generation_pipeline(
    ideas,
    workshop_description: str,
    idea_generation_model_name: str,
    idea_reflection_model_name: str,
    max_reflections: int,
    current_round: int = 0,
) -> IdeaGenerationState:
    graph = create_idea_generation_pipeline()

    return cast(
        IdeaGenerationState,
        graph.invoke(
            {
                "prev_ideas": ideas,
                "current_round": current_round,
            },
            config={
                "configurable": {
                    "thread_id": "run-idea-generation-pipeline",
                    "workshop_description": workshop_description,
                    "prev_ideas": ideas,
                    "max_reflections": max_reflections,
                    "idea_generation_model_name": idea_generation_model_name,
                    "idea_reflection_model_name": idea_reflection_model_name,
                }
            },
        ),
    )
