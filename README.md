# Soup QA take-home

Start with `report.md` and `strategy.md`. See `bug-reports/` for the defect records and `logs/` for raw evidence, including failed attempts.

Install `python -m pip install -r requirements.txt`. Run the generated config checks with one command:

```sh
python -m pytest -q -rx
```

Run the training pair and the duplicate-key control with `python repeat.py` on a CUDA runtime, ideally Colab T4. The script creates its own tiny local GPT-2 model, uses the committed synthetic data and saves separate outputs, trackers and logs for all cases. If PEFT reports the same incompatible preinstalled torchao as QA-02, use the logged workaround in a disposable runtime, then repeat the command. Compare runs using the JSON summary in the newly created `logs/training-*/` folder.

The case summary for QA-01 is intentionally marked as an expected failure while the defect is present. No external datasets, credentials or private inputs are needed. Use the dependency snapshots under logs/ to audit the executed environment.
