import os
from typing import Optional

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing_extensions import Self


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
