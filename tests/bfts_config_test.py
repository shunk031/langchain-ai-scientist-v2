import pathlib
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from ai_scientist_v2.config import (
    BestFirstTreeSearchConfig,
    BFTSAgentCodeConfig,
    BFTSAgentConfig,
    BFTSAgentFeedbackConfig,
    BFTSAgentSearchConfig,
    BFTSAgentSeedConfig,
    BFTSAgentStagesConfig,
    BFTSAgentVLMFeedbackConfig,
    BFTSCodeExecConfig,
    BFTSConfig,
    BFTSDebugConfig,
    BFTSExperimentConfig,
    BFTSReportConfig,
)


class TestBFTSCodeExecConfig:
    def test_default_values(self):
        """Test that BFTSCodeExecConfig has correct default values"""
        config = BFTSCodeExecConfig()
        assert config.timeout == 3600
        assert config.agent_file_name == "runfile.py"
        assert config.format_tb_ipython is False


class TestBFTSReportConfig:
    def test_default_values(self):
        """Test that BFTSReportConfig has correct default values"""
        config = BFTSReportConfig()
        assert config.model_name == "gpt-4o"
        assert config.temperature == 1.0

    def test_temperature_validation(self):
        """Test that temperature is validated to be between 0.0 and 1.0"""
        # Valid values
        assert BFTSReportConfig(temperature=0.0).temperature == 0.0
        assert BFTSReportConfig(temperature=0.5).temperature == 0.5
        assert BFTSReportConfig(temperature=1.0).temperature == 1.0

        # Invalid values
        with pytest.raises(ValidationError):
            BFTSReportConfig(temperature=-0.1)
        with pytest.raises(ValidationError):
            BFTSReportConfig(temperature=1.1)


class TestBFTSExperimentConfig:
    def test_default_values(self):
        """Test that BFTSExperimentConfig has correct default values"""
        config = BFTSExperimentConfig()
        assert config.num_syn_datasets == 1


class TestBFTSDebugConfig:
    def test_default_values(self):
        """Test that BFTSDebugConfig has correct default values"""
        config = BFTSDebugConfig()
        assert config.stage4 is False


class TestBFTSAgentStagesConfig:
    def test_default_values(self):
        """Test that BFTSAgentStagesConfig has correct default values"""
        config = BFTSAgentStagesConfig()
        assert config.stage1_max_iters == 20
        assert config.stage2_max_iters == 12
        assert config.stage3_max_iters == 12
        assert config.stage4_max_iters == 18


class TestBFTSAgentSeedConfig:
    def test_default_values(self):
        """Test that BFTSAgentSeedConfig has correct default values"""
        config = BFTSAgentSeedConfig()
        assert config.num_seeds == 3


class TestBFTSAgentCodeConfig:
    def test_default_values(self):
        """Test that BFTSAgentCodeConfig has correct default values"""
        config = BFTSAgentCodeConfig()
        assert config.model_name == "gpt-4o"
        assert config.temperature == 1.0
        assert config.max_tokens == 12000

    def test_temperature_validation(self):
        """Test that temperature is validated to be between 0.0 and 1.0"""
        # Valid values
        assert BFTSAgentCodeConfig(temperature=0.0).temperature == 0.0
        assert BFTSAgentCodeConfig(temperature=0.5).temperature == 0.5
        assert BFTSAgentCodeConfig(temperature=1.0).temperature == 1.0

        # Invalid values
        with pytest.raises(ValidationError):
            BFTSAgentCodeConfig(temperature=-0.1)
        with pytest.raises(ValidationError):
            BFTSAgentCodeConfig(temperature=1.1)


class TestBFTSAgentFeedbackConfig:
    def test_default_values(self):
        """Test that BFTSAgentFeedbackConfig has correct default values"""
        config = BFTSAgentFeedbackConfig()
        assert config.model_name == "gpt-4o"
        assert config.templerature == 0.5  # Note: There's a typo in the field name
        assert config.max_tokens == 8192

    def test_temperature_validation(self):
        """Test that temperature is validated to be between 0.0 and 1.0"""
        # Valid values
        assert BFTSAgentFeedbackConfig(templerature=0.0).templerature == 0.0
        assert BFTSAgentFeedbackConfig(templerature=0.5).templerature == 0.5
        assert BFTSAgentFeedbackConfig(templerature=1.0).templerature == 1.0

        # Invalid values
        with pytest.raises(ValidationError):
            BFTSAgentFeedbackConfig(templerature=-0.1)
        with pytest.raises(ValidationError):
            BFTSAgentFeedbackConfig(templerature=1.1)


class TestBFTSAgentVLMFeedbackConfig:
    def test_default_values(self):
        """Test that BFTSAgentVLMFeedbackConfig has correct default values"""
        config = BFTSAgentVLMFeedbackConfig()
        assert config.model_name == "gpt-4o"
        assert config.temperature == 0.5
        assert config.max_tokens is None

    def test_temperature_validation(self):
        """Test that temperature is validated to be between 0.0 and 1.0"""
        # Valid values
        assert BFTSAgentVLMFeedbackConfig(temperature=0.0).temperature == 0.0
        assert BFTSAgentVLMFeedbackConfig(temperature=0.5).temperature == 0.5
        assert BFTSAgentVLMFeedbackConfig(temperature=1.0).temperature == 1.0

        # Invalid values
        with pytest.raises(ValidationError):
            BFTSAgentVLMFeedbackConfig(temperature=-0.1)
        with pytest.raises(ValidationError):
            BFTSAgentVLMFeedbackConfig(temperature=1.1)


class TestBFTSAgentSearchConfig:
    def test_default_values(self):
        """Test that BFTSAgentSearchConfig has correct default values"""
        config = BFTSAgentSearchConfig()
        assert config.max_debug_depth == 3
        assert config.debug_proba == 0.5
        assert config.num_drafts == 3


class TestBFTSAgentConfig:
    def test_default_values(self):
        """Test that BFTSAgentConfig has correct default values"""
        config = BFTSAgentConfig()
        assert config.agent_type == "parallel"
        assert config.num_workers == 4
        assert config.steps == 5
        assert config.k_fold_validation == 1
        assert config.expose_prediction is False
        assert config.data_preview is False

    def test_nested_configs(self):
        """Test that nested configs are correctly initialized"""
        config = BFTSAgentConfig()

        # Test stages config
        assert isinstance(config.stages, BFTSAgentStagesConfig)
        assert config.stages.stage1_max_iters == 20
        assert config.stages.stage2_max_iters == 12
        assert config.stages.stage3_max_iters == 12
        assert config.stages.stage4_max_iters == 18

        # Test multi_seed_eval config
        assert isinstance(config.multi_seed_eval, BFTSAgentSeedConfig)
        assert config.multi_seed_eval.num_seeds == 3

        # Test code config
        assert isinstance(config.code, BFTSAgentCodeConfig)
        assert config.code.model_name == "gpt-4o"
        assert config.code.temperature == 1.0
        assert config.code.max_tokens == 12000

        # Test feedback config
        assert isinstance(config.feedback, BFTSAgentFeedbackConfig)
        assert config.feedback.model_name == "gpt-4o"
        assert config.feedback.templerature == 0.5
        assert config.feedback.max_tokens == 8192

        # Test vlm_feedback config
        assert isinstance(config.vlm_feedback, BFTSAgentVLMFeedbackConfig)
        assert config.vlm_feedback.model_name == "gpt-4o"
        assert config.vlm_feedback.temperature == 0.5
        assert config.vlm_feedback.max_tokens is None

        # Test search config
        assert isinstance(config.search, BFTSAgentSearchConfig)
        assert config.search.max_debug_depth == 3
        assert config.search.debug_proba == 0.5
        assert config.search.num_drafts == 3

    def test_agent_type_validation(self):
        """Test that agent_type is validated to be either 'parallel' or 'sequential'"""
        # Valid values
        assert BFTSAgentConfig(agent_type="parallel").agent_type == "parallel"
        assert BFTSAgentConfig(agent_type="sequential").agent_type == "sequential"

        # Invalid value - we expect a ValidationError at runtime
        # Pylance will show a type error here, but that's expected
        # since we're intentionally testing an invalid value
        with pytest.raises(ValidationError):
            BFTSAgentConfig(agent_type="invalid")  # type: ignore


class TestBestFirstTreeSearchConfig:
    @pytest.fixture
    def mock_yaml_source(self):
        """Fixture to mock YamlConfigSettingsSource"""
        mock_source = MagicMock()
        mock_source.get_config_dict.return_value = {
            "data_dir": "/path/to/data",
            "log_dir": "/path/to/logs",
            "goal": "test goal",
            "eval_option": "test option",
            "is_processed_data": True,
            "is_copy_data": False,
            "exp_name": "test_exp",
        }
        return mock_source

    @patch("ai_scientist_v2.config.YamlConfigSettingsSource")
    def test_settings_customise_sources(self, mock_yaml_source_class):
        """Test that settings_customise_sources returns YamlConfigSettingsSource"""
        # Use the actual class instead of a mock for the settings_cls parameter
        settings_cls = BestFirstTreeSearchConfig
        mock_init_settings = MagicMock()
        mock_env_settings = MagicMock()
        mock_dotenv_settings = MagicMock()
        mock_file_secret_settings = MagicMock()

        result = BestFirstTreeSearchConfig.settings_customise_sources(
            settings_cls,
            mock_init_settings,
            mock_env_settings,
            mock_dotenv_settings,
            mock_file_secret_settings,
        )

        mock_yaml_source_class.assert_called_once_with(settings_cls)
        assert len(result) == 1
        assert result[0] == mock_yaml_source_class.return_value

    def test_convert_to_pathlib(self):
        """Test that convert_to_pathlib correctly converts strings to Path objects"""
        # Test with string
        result = BestFirstTreeSearchConfig.convert_to_pathlib("test/path")
        assert isinstance(result, pathlib.Path)
        assert str(result) == "test/path"

        # Test with Path object
        path_obj = pathlib.Path("test/path")
        result = BestFirstTreeSearchConfig.convert_to_pathlib(path_obj)
        assert isinstance(result, pathlib.Path)
        assert result == path_obj

    @patch("ai_scientist_v2.config.YamlConfigSettingsSource")
    def test_model_config(self, _):
        """Test that model_config is correctly set"""
        # Use get() method to safely access the dictionary key
        assert (
            BestFirstTreeSearchConfig.model_config.get("toml_file")
            == "config/best-first-tree-search.yaml"
        )

    def test_field_annotations(self):
        """Test that field annotations are correctly defined"""
        # Verify that the field types are correct
        fields = BestFirstTreeSearchConfig.__annotations__
        assert "code_exec" in fields
        assert "report" in fields
        assert "experiment" in fields
        assert "debug" in fields
        assert "agent" in fields


class TestBFTSConfig:
    def test_alias(self):
        """Test that BFTSConfig is an alias for BestFirstTreeSearchConfig"""
        assert BFTSConfig == BestFirstTreeSearchConfig
