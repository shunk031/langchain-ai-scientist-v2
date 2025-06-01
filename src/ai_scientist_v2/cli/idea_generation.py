import pathlib

from ai_scientist_v2.pipelines import run_idea_generation_pipeline
from ai_scientist_v2.pipelines.idea_generation.utils import (
    load_ideas,
    load_workshop_description,
)


def idea_generation_runner(
    ideas_file: pathlib.Path,
    workshop_file: pathlib.Path,
    idea_generation_model_name: str,
    idea_reflection_model_name: str,
    max_reflections: int,
) -> None:
    ideas = load_ideas(ideas_file)
    workshop_description = load_workshop_description(workshop_file)

    output = run_idea_generation_pipeline(
        ideas=ideas,
        workshop_description=workshop_description,
        idea_generation_model_name=idea_generation_model_name,
        idea_reflection_model_name=idea_reflection_model_name,
        max_reflections=max_reflections,
    )
