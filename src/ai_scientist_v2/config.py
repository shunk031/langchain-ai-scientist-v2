import os

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing_extensions import Self


class Settings(BaseSettings):
    LANGSMITH_TRACING: bool
    LANGSMITH_ENDPOINT: str
    LANGSMITH_PROJECT: str
    LANGSMITH_API_KEY: SecretStr

    OPENAI_API_KEY: SecretStr

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    @model_validator(mode="after")
    def export_to_env(self) -> Self:
        for field_name in self.model_fields_set:
            field_value = getattr(self, field_name)

            if isinstance(field_value, SecretStr):
                field_value = field_value.get_secret_value()
            elif isinstance(field_value, bool):
                field_value = str(field_value).lower()

            # Export the field to the environment variables
            # !! Do not print the value to avoid exposing secrets !!
            os.environ[field_name] = field_value

        return self


settings = Settings()
