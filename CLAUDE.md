# Notes for Claude Code sessions in this repo

- **Keep `LEDGER.md` current.** After pulling and scoring any Kaggle session:
  1. Rerun `.venv/bin/python scripts/ledger_table.py --vs-base` and paste the output over the
     Results snapshot.
  2. Update the task row's status.
  3. Add a Run log line, including for failed or cancelled sessions.
  4. Record any change to settings, thresholds or claims under Decisions.

  Commit the ledger with the results it describes.
- **Write only to the run folders your task row owns** (LEDGER.md → Task board). Two sessions
  must never write the same folder.
- `docs/TEAM_PLAN.md` explains each task and how to launch it. `README.md` covers the pipeline
  and Kaggle.
- **Never read, print or commit Kaggle or Hugging Face tokens.** The repo is public.
- Before launching an experiment, explain in plain words what it tests and what each possible
  result would mean.
