import copy
import json
import shutil
import subprocess
import sys

import numpy as np
import pytest

from phaselattice import Reconstruction, default_family, reconstruct
from phaselattice.evaluation import score
from phaselattice.family import load_family
from phaselattice.io import read_json, write_json
from phaselattice.observations import load_public


def test_unknown_waveform_and_weights(example):
    directory, result = example
    assert result.recovered
    assert score(directory, result.record)["recovered"]
    assert result.verify(directory / "observations.json")["valid"]


def test_only_public_files_enter_subprocess(example, tmp_path):
    directory, _ = example
    shutil.copy(directory / "observations.json", tmp_path / "observations.json")
    process = subprocess.run(
        [
            sys.executable,
            "-I",
            "-m",
            "phaselattice",
            "recover",
            str(tmp_path / "observations.json"),
            "--output",
            str(tmp_path / "result.json"),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "RECOVERED" in process.stdout
    result = Reconstruction.load(tmp_path / "result.json")
    assert result.verify(tmp_path / "observations.json")["valid"]
    assert score(directory, result.record)["recovered"]


def test_screening_preserves_model(example):
    directory, screened = example
    full = reconstruct(directory / "observations.json", screen=False)
    assert full.record["model"] == screened.record["model"]
    assert screened.record["screening"]["retained_candidates"] < 62


@pytest.mark.parametrize("field,value", [("theta", [1, 2, 3]), ("seed", 123), ("teacher", {})])
def test_hidden_metadata_rejected(example, tmp_path, field, value):
    directory, _ = example
    record = read_json(directory / "observations.json")
    record[field] = value
    write_json(tmp_path / "invalid.json", record)
    with pytest.raises(ValueError, match="public observation"):
        load_public(tmp_path / "invalid.json")


def test_lossless_float_access_enforced(example, tmp_path):
    directory, _ = example
    record = read_json(directory / "observations.json")
    record["train"]["X"][0][0] = 2**100 + 1
    write_json(tmp_path / "invalid.json", record)
    with pytest.raises(ValueError, match="losslessly"):
        load_public(tmp_path / "invalid.json")


def test_gzip_roundtrip(example, tmp_path):
    directory, _ = example
    record = read_json(directory / "observations.json")
    write_json(tmp_path / "observations.json.gz", record)
    assert read_json(tmp_path / "observations.json.gz") == record
    _, original = load_public(directory / "observations.json")
    _, compressed = load_public(tmp_path / "observations.json.gz")
    np.testing.assert_array_equal(original["train"].inputs, compressed["train"].inputs)


@pytest.mark.parametrize("mutation", ["lift", "weight", "short_trace", "malformed"])
def test_invalid_trace_rejected(example, mutation):
    directory, result = example
    record = copy.deepcopy(result.record)
    if mutation == "lift":
        record["trace"]["lifts"][0] += 1
    elif mutation == "weight":
        record["model"]["theta"][0] = "0"
    elif mutation == "short_trace":
        for name in ("indices", "phases", "branches", "signs", "lifts"):
            record["trace"][name] = record["trace"][name][:-1]
    else:
        record["trace"] = {"indices": None}
    assert not Reconstruction(record).verify(directory / "observations.json")["valid"]


def test_shuffled_outputs_unresolved(example, tmp_path):
    directory, _ = example
    record = read_json(directory / "observations.json")
    rng = np.random.default_rng(121)
    for name in ("train", "validation"):
        rng.shuffle(record[name]["y"])
    write_json(tmp_path / "shuffled.json", record)
    assert not reconstruct(tmp_path / "shuffled.json").recovered


def test_family_digest_rejects_changed_band(tmp_path):
    record = json.loads(default_family().read_text())
    record["bands"][0]["chi"] = 7
    write_json(tmp_path / "family.json", record)
    with pytest.raises(ValueError, match="digest"):
        load_family(tmp_path / "family.json")
