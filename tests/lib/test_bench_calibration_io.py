import json

import pytest

from arm1.lib import bench_calibration_io as io


def test_known_keys_matches_session_calibration_bench_calibration():
    """Pinned explicitly: session.calibration.BenchCalibration.from_dict's known-key set is not
    importable from this branch (session/ lives on feature/session-orchestration), so this is
    the regression check that stands in for asserting against it directly. If either side adds
    or renames a field, update both and this list together."""
    assert io.KNOWN_KEYS == {
        "tether_n_per_m", "ticks_per_metre", "yield_sign", "excursion_m",
        "displacement_uncertainty_m", "measured_in_pose",
    }


class TestLoad:
    def test_missing_file_is_an_empty_calibration(self, tmp_path):
        data = io.load(tmp_path / "nope.json")
        assert data == {"excursion_m": {}}

    def test_reads_back_what_was_written(self, tmp_path):
        path = tmp_path / "bench.json"
        path.write_text(json.dumps({"tether_n_per_m": 56.2, "excursion_m": {"index": 0.02}}))
        assert io.load(path) == {"tether_n_per_m": 56.2, "excursion_m": {"index": 0.02}}

    def test_rejects_unknown_keys(self, tmp_path):
        path = tmp_path / "bench.json"
        path.write_text(json.dumps({"bogus": 1}))
        with pytest.raises(ValueError, match="unknown calibration key"):
            io.load(path)

    def test_missing_excursion_m_defaults_to_empty(self, tmp_path):
        path = tmp_path / "bench.json"
        path.write_text(json.dumps({"tether_n_per_m": 56.2}))
        assert io.load(path)["excursion_m"] == {}


class TestSave:
    def test_creates_the_file_on_first_write(self, tmp_path):
        path = tmp_path / "bench.json"
        result = io.save({"tether_n_per_m": 56.2}, path)
        assert result["tether_n_per_m"] == 56.2
        assert json.loads(path.read_text())["tether_n_per_m"] == 56.2

    def test_scalar_keys_are_replaced_not_merged(self, tmp_path):
        path = tmp_path / "bench.json"
        io.save({"ticks_per_metre": 1000.0}, path)
        io.save({"ticks_per_metre": 2000.0}, path)
        assert io.load(path)["ticks_per_metre"] == 2000.0

    def test_excursion_m_merges_finger_by_finger(self, tmp_path):
        path = tmp_path / "bench.json"
        io.save({"excursion_m": {"index": 0.02}}, path)
        io.save({"excursion_m": {"middle": 0.045}}, path)
        assert io.load(path)["excursion_m"] == {"index": 0.02, "middle": 0.045}

    def test_writing_one_key_does_not_disturb_another(self, tmp_path):
        path = tmp_path / "bench.json"
        io.save({"tether_n_per_m": 56.2, "displacement_uncertainty_m": 0.006}, path)
        io.save({"ticks_per_metre": 3300.0, "yield_sign": -1}, path)
        data = io.load(path)
        assert data["tether_n_per_m"] == 56.2
        assert data["displacement_uncertainty_m"] == 0.006
        assert data["ticks_per_metre"] == 3300.0
        assert data["yield_sign"] == -1

    def test_rejects_unknown_keys(self, tmp_path):
        with pytest.raises(ValueError, match="unknown calibration key"):
            io.save({"bogus": 1}, tmp_path / "bench.json")

    def test_rejects_an_unknown_key_already_in_the_file(self, tmp_path):
        path = tmp_path / "bench.json"
        path.write_text(json.dumps({"bogus": 1}))
        with pytest.raises(ValueError, match="unknown calibration key"):
            io.save({"tether_n_per_m": 1.0}, path)
