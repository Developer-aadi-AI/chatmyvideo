from app import config


def test_check_config_reports_missing_required_keys(monkeypatch, capsys) -> None:
    monkeypatch.setattr(config, "GROQ_API_KEY", "")
    monkeypatch.setattr(config, "HF_TOKEN", "")
    monkeypatch.setattr(config, "SUPADATA_API_KEY", "")

    missing = config.check_config()

    assert missing == ["GROQ_API_KEY", "HF_TOKEN", "SUPADATA_API_KEY"]
    assert capsys.readouterr().out == (
        "Missing required configuration keys: GROQ_API_KEY, HF_TOKEN, SUPADATA_API_KEY\n"
    )


def test_check_config_does_not_print_key_values(monkeypatch, capsys) -> None:
    for name in ("GROQ_API_KEY", "HF_TOKEN", "SUPADATA_API_KEY"):
        monkeypatch.setattr(config, name, "test-secret-value")

    assert config.check_config() == []
    assert capsys.readouterr().out == "All required configuration keys are set.\n"


def test_optional_settings_use_defaults_when_empty(monkeypatch) -> None:
    for name in ("LLM_MODEL", "EMBED_MODEL", "RETRIEVER_K", "FULL_CONTEXT_CHAR_LIMIT"):
        monkeypatch.setenv(name, "")

    assert config._string_setting("LLM_MODEL", "openai/gpt-oss-120b") == ("openai/gpt-oss-120b")
    assert config._string_setting("EMBED_MODEL", "intfloat/multilingual-e5-large") == (
        "intfloat/multilingual-e5-large"
    )
    assert config._integer_setting("RETRIEVER_K", 4) == 4
    assert config._integer_setting("FULL_CONTEXT_CHAR_LIMIT", 24000) == 24000


def test_supadata_key_uses_backend_env_as_fallback(monkeypatch) -> None:
    monkeypatch.delenv("SUPADATA_API_KEY", raising=False)
    monkeypatch.setattr(config, "_BACKEND_ENV", {"SUPADATA_API_KEY": "backend-key"})

    assert config._supadata_api_key_setting() == "backend-key"


def test_supadata_key_prefers_environment_over_backend_fallback(monkeypatch) -> None:
    monkeypatch.setenv("SUPADATA_API_KEY", "root-key")
    monkeypatch.setattr(config, "_BACKEND_ENV", {"SUPADATA_API_KEY": "backend-key"})

    assert config._supadata_api_key_setting() == "root-key"
