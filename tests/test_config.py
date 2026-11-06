from pathlib import Path

from config import ROOT, Settings, get_settings, load_settings


def test_defaults():
    s = load_settings({})
    assert s == Settings()
    assert s.max_upload_mb == 50
    assert s.seed == 42
    assert s.storage_dir == ROOT / "storage"


def test_env_overrides_are_cast_to_the_field_type():
    s = load_settings({"LEXORA_MAX_UPLOAD_MB": "7", "LEXORA_SEED": "1",
                       "LEXORA_STORAGE_DIR": "/tmp/somewhere"})
    assert s.max_upload_mb == 7 and isinstance(s.max_upload_mb, int)
    assert s.seed == 1
    assert s.storage_dir == Path("/tmp/somewhere")


def test_empty_and_unrelated_variables_are_ignored():
    s = load_settings({"LEXORA_MAX_UPLOAD_MB": "", "MAX_UPLOAD_MB": "999"})
    assert s.max_upload_mb == 50


def test_get_settings_reads_the_environment_once(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("LEXORA_SEED", "7")
    assert get_settings().seed == 7
    monkeypatch.setenv("LEXORA_SEED", "8")
    assert get_settings().seed == 7          # cached
    get_settings.cache_clear()
    assert get_settings().seed == 8
    get_settings.cache_clear()