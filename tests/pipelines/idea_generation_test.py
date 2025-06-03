import pathlib
from typing import Dict, List

import pytest

from ai_scientist_v2.models import Idea
from ai_scientist_v2.pipelines import (
    create_idea_generation_pipeline,
    run_idea_generation_pipeline,
)


@pytest.fixture
def max_reflections() -> int:
    return 2


@pytest.fixture
def idea_generation_model_name() -> str:
    return "openai:gpt-4o"


@pytest.fixture
def idea_reflection_model_name() -> str:
    return "openai:gpt-4o"


def test_create_idea_generation_pipeline(root_dir: pathlib.Path):
    import io

    from langchain_core.runnables.graph import CurveStyle
    from PIL import Image

    graph = create_idea_generation_pipeline()

    image = Image.open(
        io.BytesIO(graph.get_graph().draw_mermaid_png(curve_style=CurveStyle.LINEAR)),
    )
    image.save(root_dir / ".github" / "assets" / "graph-idea-generation.png")


def test_run_idea_generation(
    idea_generation_model_name: str,
    idea_reflection_model_name: str,
    ideas: List[Dict[str, str]],
    workshop_description: str,
    max_reflections: int,
):
    output = run_idea_generation_pipeline(
        ideas=ideas,
        workshop_description=workshop_description,
        idea_generation_model_name=idea_generation_model_name,
        idea_reflection_model_name=idea_reflection_model_name,
        max_reflections=max_reflections,
    )
    idea = output.generated_idea
    assert idea is not None
    assert isinstance(idea, Idea)
