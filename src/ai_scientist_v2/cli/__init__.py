import pathlib
from typing import Union

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import CliApp, CliSubCommand

from .idea_generation import idea_generation_runner


class IdeaGeneration(BaseModel):
    workshop_file: pathlib.Path = Field(
        description="Path to the workshop description file."
    )
    ideas_file: pathlib.Path = Field(
        description="Path to the JSONL file containing initial ideas."
    )
    idea_generation_model_name: str = Field(
        default="openai:gpt-4o",
        description="Model name for generating new ideas.",
    )
    idea_reflection_model_name: str = Field(
        default="openai:gpt-4o",
        description="Model name for reflecting on ideas.",
    )
    max_reflections: int = Field(
        default=5,
        description="Maximum number of reflection rounds for each idea.",
    )

    @field_validator("workshop_file", mode="before")
    @classmethod
    def validate_workshop_file(cls, value: Union[pathlib.Path, str]) -> pathlib.Path:
        value = pathlib.Path(value)  # Ensure value is a Path object
        if value.suffix != ".md":
            raise ValueError(f"Workshop file {value} must be a Markdown (.md) file.")
        return value

    @field_validator("ideas_file", mode="before")
    @classmethod
    def validate_ideas_file(cls, value: Union[pathlib.Path, str]) -> pathlib.Path:
        value = pathlib.Path(value)  # Ensure value is a Path object
        if value.suffix != ".jsonl":
            raise ValueError(f"Ideas file {value} must be a JSON Lines (.jsonl) file.")
        return value

    def cli_cmd(self) -> None:
        idea_generation_runner(
            workshop_file=self.workshop_file,
            ideas_file=self.ideas_file,
            idea_generation_model_name=self.idea_generation_model_name,
            idea_reflection_model_name=self.idea_reflection_model_name,
            max_reflections=self.max_reflections,
        )


class AiScientistV2(BaseModel):
    idea_generation: CliSubCommand[IdeaGeneration]

    def cli_cmd(self) -> None:
        CliApp.run_subcommand(self)


def run() -> None:
    CliApp.run(AiScientistV2)
