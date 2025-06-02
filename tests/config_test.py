import os
from unittest.mock import patch

import pytest
from pydantic import SecretStr

from ai_scientist_v2.config import Settings, settings


@pytest.fixture
def mock_env_vars():
    """Fixture to mock environment variables"""
    env_vars = {
        "LANGSMITH_TRACING": "True",
        "LANGSMITH_ENDPOINT": "https://api.langsmith.com",
        "LANGSMITH_PROJECT": "test-project",
        "LANGSMITH_API_KEY": "langsmith-api-key",
        "OPENAI_API_KEY": "openai-api-key",
    }
    return env_vars


def test_settings_initialization(mock_env_vars):
    """Test that Settings is correctly initialized from environment variables"""
    with patch.dict(os.environ, mock_env_vars):
        # Create a new Settings instance
        test_settings = Settings.model_validate({})

        # Verify that each setting value is correctly loaded
        assert test_settings.LANGSMITH_TRACING is True
        assert test_settings.LANGSMITH_ENDPOINT == "https://api.langsmith.com"
        assert test_settings.LANGSMITH_PROJECT == "test-project"
        assert isinstance(test_settings.LANGSMITH_API_KEY, SecretStr)
        assert test_settings.LANGSMITH_API_KEY.get_secret_value() == "langsmith-api-key"
        assert isinstance(test_settings.OPENAI_API_KEY, SecretStr)
        assert test_settings.OPENAI_API_KEY.get_secret_value() == "openai-api-key"


def test_settings_secret_str(mock_env_vars):
    """Test that SecretStr values are handled correctly"""
    with patch.dict(
        os.environ,
        {
            **mock_env_vars,
            "LANGSMITH_API_KEY": "test-api-key",
            "OPENAI_API_KEY": "test-openai-key",
        },
    ):
        test_settings = Settings.model_validate({})

        # Verify that SecretStr values are hidden in string representation
        assert str(test_settings.LANGSMITH_API_KEY) == "**********"
        assert str(test_settings.OPENAI_API_KEY) == "**********"

        # Verify that actual values can be retrieved using get_secret_value() method
        assert test_settings.LANGSMITH_API_KEY.get_secret_value() == "test-api-key"
        assert test_settings.OPENAI_API_KEY.get_secret_value() == "test-openai-key"


def test_settings_boolean_parsing(mock_env_vars):
    """Test that boolean environment variables are correctly parsed"""
    # Cases that should be parsed as True
    for true_value in ["True", "true", "1", "yes", "y", "on"]:
        with patch.dict(
            os.environ,
            {
                **mock_env_vars,
                "LANGSMITH_TRACING": true_value,
            },
        ):
            test_settings = Settings.model_validate({})
            assert test_settings.LANGSMITH_TRACING is True, (
                f"'{true_value}' should be parsed as True"
            )

    # Cases that should be parsed as False
    for false_value in ["False", "false", "0", "no", "n", "off"]:
        with patch.dict(
            os.environ,
            {
                **mock_env_vars,
                "LANGSMITH_TRACING": false_value,
            },
        ):
            test_settings = Settings.model_validate({})
            assert test_settings.LANGSMITH_TRACING is False, (
                f"'{false_value}' should be parsed as False"
            )


def test_settings_model_config():
    """Test that Settings class model_config is correctly configured"""
    assert Settings.model_config.get("env_file") == ".env"
    assert Settings.model_config.get("env_file_encoding") == "utf-8"


def test_settings_singleton():
    """Test that settings instance is correctly created"""
    assert settings is not None
    assert isinstance(settings, Settings)


def test_export_to_env(mock_env_vars):
    """Test that export_to_env method correctly exports environment variables"""
    # Clear environment variables and set only test values
    with patch.dict(os.environ, {}, clear=True):
        with patch.dict(os.environ, mock_env_vars):
            # Create Settings instance (export_to_env is automatically executed at this point)
            Settings.model_validate({})

            # Verify that SecretStr fields are correctly exported
            assert os.environ["LANGSMITH_API_KEY"] == "langsmith-api-key"
            assert os.environ["OPENAI_API_KEY"] == "openai-api-key"

            # Verify that boolean fields are converted to lowercase strings
            assert os.environ["LANGSMITH_TRACING"] == "true"

            # Verify that regular string fields are exported as-is
            assert os.environ["LANGSMITH_ENDPOINT"] == "https://api.langsmith.com"
            assert os.environ["LANGSMITH_PROJECT"] == "test-project"


def test_export_to_env_with_different_types():
    """Test that different types of values are correctly exported to environment variables"""
    test_values = {
        "LANGSMITH_TRACING": False,
        "LANGSMITH_ENDPOINT": "https://custom.endpoint.com",
        "LANGSMITH_PROJECT": "custom-project",
        "LANGSMITH_API_KEY": SecretStr("custom-api-key"),
        "OPENAI_API_KEY": SecretStr("custom-openai-key"),
    }

    # Clear environment variables and set only test values
    with patch.dict(os.environ, {}, clear=True):
        # Create Settings instance with direct values (export_to_env is automatically executed)
        Settings.model_validate(test_values)

        # Verify that each type of value is correctly exported
        assert os.environ["LANGSMITH_TRACING"] == "false"
        assert os.environ["LANGSMITH_ENDPOINT"] == "https://custom.endpoint.com"
        assert os.environ["LANGSMITH_PROJECT"] == "custom-project"
        assert os.environ["LANGSMITH_API_KEY"] == "custom-api-key"
        assert os.environ["OPENAI_API_KEY"] == "custom-openai-key"
