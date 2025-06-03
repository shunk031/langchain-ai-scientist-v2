import pathlib
from typing import Dict, List

import pytest

from ai_scientist_v2.pipelines.idea_generation.utils import (
    load_ideas,
    load_workshop_description,
)


@pytest.fixture
def root_dir() -> pathlib.Path:
    return pathlib.Path(__file__).parents[1]


@pytest.fixture
def test_fixtures_dir(root_dir: pathlib.Path) -> pathlib.Path:
    test_fixtures_dir = root_dir / "test_fixtures"
    assert test_fixtures_dir.exists()
    return test_fixtures_dir


@pytest.fixture
def ideas_file(test_fixtures_dir: pathlib.Path) -> pathlib.Path:
    return test_fixtures_dir / "i-cant-believe-its-not-better.jsonl"


@pytest.fixture
def workshop_description_file(test_fixtures_dir: pathlib.Path) -> pathlib.Path:
    return test_fixtures_dir / "i-cant-believe-its-not-better.md"


@pytest.fixture
def ideas(ideas_file: pathlib.Path) -> List[Dict[str, str]]:
    return load_ideas(ideas_file)


@pytest.fixture
def workshop_description(workshop_description_file: pathlib.Path):
    return load_workshop_description(workshop_description_file)
