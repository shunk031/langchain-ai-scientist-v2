import json
import pathlib
from typing import Dict, List

import pytest

from ai_scientist_v2.pipelines import (
    create_idea_generation_pipeline,
    run_idea_generation_pipeline,
)
from ai_scientist_v2.pipelines.idea_generation import Idea


@pytest.fixture
def max_reflections() -> int:
    return 2


@pytest.fixture
def ideas_file(test_fixtures_dir: pathlib.Path) -> pathlib.Path:
    return test_fixtures_dir / "i-cant-believe-its-not-better.jsonl"


@pytest.fixture
def workshop_description_file(test_fixtures_dir: pathlib.Path) -> pathlib.Path:
    return test_fixtures_dir / "i-cant-believe-its-not-better.md"


@pytest.fixture
def ideas(ideas_file: pathlib.Path) -> List[Dict[str, str]]:
    with open(ideas_file, "r") as f:
        ideas = [json.loads(line) for line in f]
    return ideas


@pytest.fixture
def workshop_description(workshop_description_file: pathlib.Path):
    with open(workshop_description_file, "r") as f:
        workshop_description = f.read()
    return workshop_description


@pytest.fixture
def idea_generation_model_name() -> str:
    return "openai:gpt-4o"


@pytest.fixture
def idea_reflection_model_name() -> str:
    return "openai:gpt-4o"


def test_create_idea_generation_pipeline():
    import io

    from PIL import Image

    graph = create_idea_generation_pipeline()

    image = Image.open(
        io.BytesIO(graph.get_graph().draw_mermaid_png()),
    )
    image.save("graph.png")


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
    idea_dict = output["generated_idea"]
    assert idea_dict is not None
    assert Idea.model_validate(idea_dict)
