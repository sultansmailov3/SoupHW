
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
from datetime import datetime, timezone
import traceback

import torch
import yaml
from safetensors.torch import load_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parent


def prepare_model(path):
    tokens = ["[PAD]", "[UNK]", "[EOS]", "Add", "one", "to", ".", "2", "3", "4", "5", "6"]
    backend = Tokenizer(WordLevel(dict(zip(tokens, range(len(tokens)))), unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]", eos_token="[EOS]"
    ).save_pretrained(path)
    torch.manual_seed(11)
    model = GPT2LMHeadModel(GPT2Config(
        vocab_size=len(tokens), n_positions=128, n_embd=32, n_layer=1, n_head=2,
        bos_token_id=2, eos_token_id=2, pad_token_id=0,
    ))
    model.save_pretrained(path)


def train(folder, text):
    folder.mkdir()
    shutil.copyfile(ROOT / "configs/train.jsonl", folder / "train.jsonl")
    (folder / "soup.yaml").write_text(text)
    env = dict(os.environ, SOUP_DB_PATH=str(folder / "tracker.sqlite3"),
               HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false", NO_COLOR="1")
    command = ["soup", "train", "-c", "soup.yaml", "--yes"]
    (folder / "command.json").write_text(json.dumps(command))
    with (folder / "stdout.log").open("wb") as out, (folder / "stderr.log").open("wb") as err:
        p = subprocess.run(command, cwd=folder, env=env, stdout=out, stderr=err, timeout=300)
    (folder / "exit-code.txt").write_text(str(p.returncode))
    print(folder.name, p.returncode, flush=True)
    return p.returncode


def read_artifacts(folder):
    states = [json.loads(p.read_text()) for p in folder.rglob("trainer_state.json")]
    state = max(states, key=lambda v: v["global_step"])
    loss = {v["step"]: v["loss"] for v in state["log_history"] if "loss" in v}
    with sqlite3.connect(folder / "tracker.sqlite3") as db:
        row = db.execute("SELECT status, initial_loss, final_loss, total_steps FROM runs").fetchone()
    weights = load_file(folder / "output/adapter_model.safetensors")
    return loss, dict(zip(("status", "initial_loss", "final_loss", "total_steps"), row)), weights


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA runtime is required for this test")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    base = ROOT / "logs" / ("training-" + stamp)
    base.mkdir()
    (base / "device.json").write_text(json.dumps({
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "cuda": torch.version.cuda
    }, indent=2))
    model = base / "initial-model"
    prepare_model(model)
    cfg = yaml.safe_load((ROOT / "configs/baseline.yaml").read_text())
    cfg["base"] = str(model)
    cfg["data"]["chat_template"] = "chatml"
    cfg["training"].update(save_steps=1, lora={"r": 4, "target_modules": ["c_attn"]})
    clean = yaml.safe_dump(cfg)
    duplicate = clean.replace("  epochs: 1\n", "  epochs: 1\n  epochs: 2\n")
    assert clean != duplicate
    cases = {"a": clean, "b": clean, "duplicate": duplicate}
    codes = {}
    for name, text in cases.items():
        try:
            codes[name] = train(base / name, text)
        except Exception:
            (base / name / "runner-error.log").write_text(traceback.format_exc())
            codes[name] = None
    summary = {"exit_codes": codes}
    records = {}
    for name, code in codes.items():
        if code == 0:
            try:
                records[name] = read_artifacts(base / name)
            except Exception:
                (base / name / "analysis-error.log").write_text(traceback.format_exc())
    if "a" in records and "b" in records:
        la, ma, wa = records["a"]
        lb, mb, wb = records["b"]
        assert la and la.keys() == lb.keys() and wa.keys() == wb.keys()
        assert all(wa[k].shape == wb[k].shape and wa[k].dtype == wb[k].dtype for k in wa)
        summary["repeatability"] = dict(
            loss_a=la, loss_b=lb, loss_max_delta=max(abs(la[k] - lb[k]) for k in la),
            first_different_step=next((k for k in sorted(la) if la[k] != lb[k]), None),
            weights_exact=all(torch.equal(wa[k], wb[k]) for k in wa),
            weights_max_delta=max((wa[k].double() - wb[k].double()).abs().max().item() for k in wa),
            metrics_a=ma, metrics_b=mb,
            metric_abs_delta={k: abs(ma[k] - mb[k]) for k in ("initial_loss", "final_loss", "total_steps")},
        )
    if "a" in records and "duplicate" in records:
        summary["duplicate"] = dict(control=records["a"][1], duplicate=records["duplicate"][1])
    (base / "comparison.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if len(records) != 3:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
