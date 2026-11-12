
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import pytest
import yaml
from hypothesis import given, settings, strategies as st

ROOT = Path(__file__).resolve().parent
CASES = {
    "typo": st.sampled_from(["epocs", "eopchs", "epochss", "epchs"]),
    "wrong_type": st.sampled_from([[], {}, "not-an-integer", 1.25]),
    "out_of_range": st.integers(-500, 0),
    "empty": st.sampled_from([None, [], {}]),
    "conflict": st.booleans(),
    "long": st.integers(260, 2048),
}


def invoke(kind, config, command=None):
    logs = ROOT / "logs" / "cases"
    logs.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix=f"{kind}-", dir=logs))
    shutil.copyfile(ROOT / "configs/train.jsonl", folder / "train.jsonl")
    (folder / "soup.yaml").write_text(config if isinstance(config, str) else yaml.safe_dump(config))
    args = command or ["soup", "plan", "-c", "soup.yaml"]
    env = dict(os.environ, NO_COLOR="1", HF_HUB_OFFLINE="1")
    code = None
    try:
        p = subprocess.run(args, cwd=folder, env=env, capture_output=True, timeout=45)
        code, stdout, stderr = p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as exc:
        stdout, stderr = exc.stdout or b"", exc.stderr or b""
    (folder / "stdout.log").write_bytes(stdout)
    (folder / "stderr.log").write_bytes(stderr)
    output = (stdout + stderr).decode(errors="replace")
    if code is None:
        outcome = "timeout"
    elif "Traceback" in output:
        outcome = "traceback"
    elif code == 0:
        outcome = "accepted"
    elif "Config validation error" in output:
        outcome = "clear config error"
    else:
        outcome = "other error"
    (folder / "result.json").write_text(json.dumps(dict(kind=kind, args=args, code=code, outcome=outcome), indent=2))
    return outcome, output


def test_control():
    assert invoke("control", yaml.safe_load((ROOT / "configs/baseline.yaml").read_text()))[0] == "accepted"


@pytest.mark.parametrize("kind", CASES)
@settings(max_examples=5, deadline=None, derandomize=True)
@given(data=st.data())
def test_mutations(kind, data):
    val = data.draw(CASES[kind], label=kind)
    config = yaml.safe_load((ROOT / "configs/baseline.yaml").read_text())
    t = config["training"]
    if kind == "typo":
        t[val] = t.pop("epochs")
    elif kind in {"wrong_type", "out_of_range"}:
        t["epochs"] = val
    elif kind == "empty":
        config["data"]["train"] = val
    elif kind == "conflict":
        t.update(packing=True, multipack=True, gradient_accumulation_steps=1 + val)
    else:
        config["experiment_name"] = "x" * val
    outcome, output = invoke(kind, config)
    if kind == "long":
        assert outcome in {"accepted", "clear config error"}, output
    else:
        assert outcome == "clear config error", output


def test_duplicate_epochs():
    config = (ROOT / "configs/baseline.yaml").read_text()
    config = config.replace("  epochs: 1\n", "  epochs: 1\n  epochs: 2\n")
    result, output = invoke("duplicate-plan", config)
    if result == "accepted":
        parsed, detail = invoke("duplicate-effective", config, [
            "python", "-c",
            'from soup_cli.config.loader import load_config; print(load_config("soup.yaml").training.epochs)',
        ])
        assert parsed == "accepted" and detail.strip() == "2", detail
        pytest.xfail("QA-01: duplicate epochs accepted and last value used")
    assert result == "clear config error" and "duplicate" in output.lower(), output
