"""Tests for the config module: settings surface from a .env file, missing required values fail with a readable error."""

from pathlib import Path

import pytest

from src.config import ConfigError, load_settings

_CONFIG_ENV_VARS = (
    "OPENAI_API_KEY",
    "DOC_QA_CHAT_MODEL",
    "DOC_QA_REASONING_EFFORT",
    "DOC_QA_EMBEDDING_MODEL",
    "DOC_QA_DATA_DIR",
)


@pytest.fixture(autouse=True)
def _clean_config_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scrub config vars so tests are hermetic against the developer's real environment."""
    for key in _CONFIG_ENV_VARS:
        monkeypatch.delenv(key, raising=False)


def test_settings_surface_from_env_file(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=sk-test-123",
                "DOC_QA_CHAT_MODEL=gpt-5.6-luna",
                "DOC_QA_REASONING_EFFORT=low",
                "DOC_QA_EMBEDDING_MODEL=text-embedding-3-large",
                "DOC_QA_DATA_DIR=store",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_settings(env_file)

    assert settings.openai_api_key == "sk-test-123"
    assert settings.chat_model == "gpt-5.6-luna"
    assert settings.reasoning_effort == "low"
    assert settings.embedding_model == "text-embedding-3-large"
    assert settings.data_dir == Path("store")


def test_missing_api_key_raises_readable_error(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("DOC_QA_CHAT_MODEL=gpt-5.6-luna\n", encoding="utf-8")

    with pytest.raises(ConfigError) as exc_info:
        load_settings(env_file)

    message = str(exc_info.value)
    assert "OPENAI_API_KEY" in message
    assert ".env" in message


def test_unset_optional_keys_fall_back_to_spec_defaults(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=sk-test-123\n", encoding="utf-8")

    settings = load_settings(env_file)

    # Defaults per the spec: gpt-5.6-luna @ high reasoning, text-embedding-3-small.
    assert settings.chat_model == "gpt-5.6-luna"
    assert settings.reasoning_effort == "high"
    assert settings.embedding_model == "text-embedding-3-small"
    assert settings.data_dir == Path("data")
