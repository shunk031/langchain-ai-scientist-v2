import pathlib
from typing import Optional

from pydantic import BaseModel
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

# class Settings(BaseSettings):
#     pass


class BFTSAgentConfig(BaseModel):
    agent_type: str = "parallel"


class BestFirstTreeSearchConfig(BaseSettings):
    data_dir: pathlib.Path
    preprocess_data: bool = False

    goal: Optional[str] = None
    eval_option: Optional[str] = None

    agent: BFTSAgentConfig

    model_config = SettingsConfigDict(toml_file="config/best-first-tree-search.yaml")

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
