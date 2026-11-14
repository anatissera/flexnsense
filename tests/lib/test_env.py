from lib.env import env_get, load_env


def test_load_env_parses_and_skips(tmp_path):
    p = tmp_path / ".env"
    p.write_text('# comment\n\nROBOT_PORT="/dev/ttyACM0"\nHF_TOKEN=\nCAMERA_INDEX = 1\nnot a pair\n')
    assert load_env(p) == {"ROBOT_PORT": "/dev/ttyACM0", "HF_TOKEN": "", "CAMERA_INDEX": "1"}
    assert load_env(tmp_path / "missing.env") == {}


def test_env_get_precedence(tmp_path, monkeypatch):
    p = tmp_path / ".env"
    p.write_text("ROBOT_PORT=/dev/from-file\nHF_TOKEN=\n")
    monkeypatch.delenv("ROBOT_PORT", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert env_get("ROBOT_PORT", path=p) == "/dev/from-file"
    monkeypatch.setenv("ROBOT_PORT", "/dev/from-env")
    assert env_get("ROBOT_PORT", path=p) == "/dev/from-env"
    assert env_get("HF_TOKEN", default="none", path=p) == "none"      # empty counts as unset
    assert env_get("NOT_THERE", path=p) is None
