import os
import pathlib
from typing import Literal, Optional

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)
from typing_extensions import Self


class BFTSCodeExecConfig(BaseModel):
    timeout: int = 3600
    agent_file_name: str = "runfile.py"
    format_tb_ipython: bool = False


class BFTSReportConfig(BaseModel):
    model_name: str = "gpt-4o"
    temperature: float = Field(default=1.0, ge=0.0, le=1.0)


class BFTSExperimentConfig(BaseModel):
    num_syn_datasets: int = 1


class BFTSDebugConfig(BaseModel):
    stage4: bool = False


class BFTSAgentStagesConfig(BaseModel):
    stage1_max_iters: int = 20
    stage2_max_iters: int = 12
    stage3_max_iters: int = 12
    stage4_max_iters: int = 18


class BFTSAgentSeedConfig(BaseModel):
    num_seeds: int = 3


class BFTSAgentCodeConfig(BaseModel):
    model_name: str = "gpt-4o"
    temperature: float = Field(default=1.0, ge=0.0, le=1.0)
    max_tokens: int = 12000


class BFTSAgentFeedbackConfig(BaseModel):
    model_name: str = "gpt-4o"
    templerature: float = Field(default=0.5, ge=0.0, le=1.0)
    max_tokens: int = 8192


class BFTSAgentVLMFeedbackConfig(BaseModel):
    model_name: str = "gpt-4o"
    temperature: float = Field(default=0.5, ge=0.0, le=1.0)
    max_tokens: Optional[int] = None


class BFTSAgentSearchConfig(BaseModel):
    max_debug_depth: int = 3
    debug_proba: float = 0.5
    num_drafts: int = 3


class BFTSAgentConfig(BaseModel):
    agent_type: Literal["parallel", "sequential"] = "parallel"
    num_workers: int = 4
    stages: BFTSAgentStagesConfig = Field(
        default_factory=BFTSAgentStagesConfig,
        description="Configuration for the agent stages.",
    )
    steps: int = Field(
        default=5,
        description="How many improvement iterations to run. If stage-specific max_iters are not provided, the agent will use this value for all stages.",
    )
    k_fold_validation: int = Field(
        default=1,
        description="Whether to instruct the agent to use CV (set to 1 to disable CV).",
    )
    multi_seed_eval: BFTSAgentSeedConfig = Field(
        default_factory=BFTSAgentSeedConfig,
        description="Configuration for multi-seed evaluation.",
    )
    expose_prediction: bool = Field(
        default=False,
        description="Whether to instruct the agent to generate a prediction function.",
    )
    data_preview: bool = Field(
        default=False,
        description="Whether to provide the agent with a preview of the data.",
    )
    code: BFTSAgentCodeConfig = Field(
        default=BFTSAgentCodeConfig(),
        description="Settings for code execution.",
    )
    feedback: BFTSAgentFeedbackConfig = Field(
        default=BFTSAgentFeedbackConfig(),
        description="Settings for feedback generation.",
    )
    vlm_feedback: BFTSAgentVLMFeedbackConfig = Field(
        default=BFTSAgentVLMFeedbackConfig(),
        description="Settings for VLM feedback generation.",
    )
    search: BFTSAgentSearchConfig = Field(
        default=BFTSAgentSearchConfig(),
        description="Settings for the agent search.",
    )


class BestFirstTreeSearchConfig(BaseSettings):
    data_dir: pathlib.Path = Field(
        description="Directory containing the data files",
    )
    log_dir: pathlib.Path = Field(
        description="Directory where logs will be stored",
    )
    workspace_dir: pathlib.Path = Field(
        description="Directory where the workspace will be created",
    )

    goal: Optional[str] = None
    eval_option: Optional[str] = None

    is_processed_data: bool = Field(
        default=False,
        description="Whether the data is already processed",
    )
    is_copy_data: bool = Field(
        default=True,
        description="Whether to the data to the workspace directory (otherwise it will be symlinked). Copying is recommended to prevent the agent from accidentally modifying the original data.",
    )

    exp_name: Optional[str] = Field(
        default=None,
        description="Experiment name. A random experiment name will be generated if not provided.",
    )

    code_exec: BFTSCodeExecConfig = Field(
        default=BFTSCodeExecConfig(),
        description="Settings for code execution.",
    )
    report: Optional[BFTSReportConfig] = Field(
        default=BFTSReportConfig(),
        description="Settings for report generation.",
    )
    experiment: BFTSExperimentConfig = Field(
        default=BFTSExperimentConfig(),
        description="Settings for the experiment.",
    )
    debug: BFTSDebugConfig = Field(
        default=BFTSDebugConfig(),
        description="Settings for debugging.",
    )
    agent: BFTSAgentConfig = Field(
        default=BFTSAgentConfig(),
        description="Settings for the agent.",
    )

    # Configuration for Pydantic settings
    model_config = SettingsConfigDict(
        toml_file="config/best-first-tree-search.yaml",
    )

    @field_validator("data_dir", "log_dir", "workspace_dir", mode="before")
    @classmethod
    def convert_to_pathlib(cls, value: str | pathlib.Path) -> pathlib.Path:
        return pathlib.Path(value)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (YamlConfigSettingsSource(settings_cls),)


BFTSConfig = BestFirstTreeSearchConfig


class Settings(BaseSettings):
    OPENAI_API_KEY: SecretStr = Field(
        description="OpenAI API key",
    )
    LANGSMITH_TRACING: bool = Field(
        default=False, description="Enable LangSmith tracing"
    )
    LANGSMITH_ENDPOINT: Optional[str] = Field(
        default="https://api.smith.langchain.com", description="LangSmith API endpoint"
    )
    LANGSMITH_PROJECT: Optional[str] = Field(
        default="ai-scientist-v2", description="LangSmith project name"
    )
    LANGSMITH_API_KEY: Optional[SecretStr] = Field(
        default=None, description="LangSmith API key"
    )

    # Configuration for Pydantic settings
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    @model_validator(mode="after")
    def export_to_env(self) -> Self:
        for field_name in self.model_fields_set:
            field_value = getattr(self, field_name)

            if field_value is None:
                # Skip fields that are None
                continue

            if isinstance(field_value, SecretStr):
                field_value = field_value.get_secret_value()
            elif isinstance(field_value, bool):
                field_value = str(field_value).lower()

            # Export the field to the environment variables
            # !! Do not print the value to avoid exposing secrets !!
            os.environ[field_name] = field_value

        return self


settings = Settings()
