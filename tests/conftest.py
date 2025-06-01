import pathlib

import pytest


@pytest.fixture
def root_dir() -> pathlib.Path:
    return pathlib.Path(__file__).parents[1]


@pytest.fixture
def test_fixtures_dir(root_dir: pathlib.Path) -> pathlib.Path:
    test_fixtures_dir = root_dir / "test_fixtures"
    assert test_fixtures_dir.exists()
    return test_fixtures_dir
