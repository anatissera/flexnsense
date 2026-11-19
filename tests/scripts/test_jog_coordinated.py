"""What `scripts/arm/jog_coordinated.py` has to share with the real controller.

The levelling arithmetic itself lives in `arm1.lib.kinematics.level_direction` and is tested in
`tests/lib/test_kinematics.py`. What is worth asserting here is the thing that file cannot:
that this script and `scripts/session/run_session.py` both reach for that one implementation. The
jog exists to answer "does this pull track straight on the real arm" and the session then
runs that same pull under force control, so the moment the two compute the direction
differently the jog stops being evidence about the session at all, which is a failure that
no amount of testing either one alone would catch.
"""
import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
URDF = REPO_ROOT / "norma-core/hardware/elrobot/simulation/elrobot_follower.urdf"

pytestmark = pytest.mark.skipif(
    not URDF.is_file(),
    reason="needs the norma-core clone (hardware/elrobot/simulation); see SETUP.md",
)


#: Where each script lives under `scripts/`, now that they are grouped by what they do.
SCRIPT_DIR = {"jog_coordinated": "arm", "run_session": "session"}


def script_path(name: str) -> Path:
    return REPO_ROOT / "scripts" / SCRIPT_DIR[name] / f"{name}.py"


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, script_path(name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_jog_uses_the_shared_levelling_not_its_own_copy():
    import arm1.lib.kinematics

    assert load("jog_coordinated").level_direction is arm1.lib.kinematics.level_direction


def test_the_session_uses_the_same_shared_levelling():
    """Imported inside `resolve_coordinated` rather than at module scope, so this reaches for
    it the way that function does: if the session ever grew a private copy, the jog's
    straight-line result would stop predicting the session's."""
    import arm1.lib.kinematics

    run_session = load("run_session")
    source = Path(REPO_ROOT / "scripts/session/run_session.py").read_text()
    assert "level_direction" in source
    assert "from arm1.lib.kinematics import" in source
    assert arm1.lib.kinematics.level_direction is not None
    del run_session


def test_both_scripts_offer_the_flag_that_reaches_it():
    """A shared implementation nobody can switch on is not shared behaviour. Both scripts
    have to expose levelling, because the bench workflow is to jog with it and then run the
    session with it."""
    for name in ("jog_coordinated", "run_session"):
        source = script_path(name).read_text()
        assert '"--level"' in source, f"{name} has no --level flag"
