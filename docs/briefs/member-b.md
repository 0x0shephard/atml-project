# Brief for Member B's agent: robustness, controls and measurement checks

You are working in the `atml-project` repository, a research harness that studies how compressed
reasoning LLMs fail. Your job is to make every number in the paper **defensible**:
- finish the missing controls;
- measure how noisy the results are;
- check the answer detector against human labels;
- run the remaining fix controls.

Most of this is launching well-defined Kaggle runs and careful checking, with little new code.
The paper we answer is Lotfi et al., arXiv 2606.00206 (NeurIPS 2026 oral). Reviewers will
compare our numbers with theirs line by line.

You have only this repository as context. Read before doing anything:
1. `CLAUDE.md`: the house rules.
2. `LEDGER.md`: the findings, task board (your tasks are **T07, T08, T09, T10, T11, T12,
   T14**), run log and decisions.
3. `README.md`: the pipeline, Kaggle workflow, outcome labels and design notes.
4. `docs/TEAM_PLAN.md`: why the work is split this way.

## What the project has found (so you know what you're checking)

- **Compressed models mostly fail because they can't stop.** Wrong answers are mostly chains that
  hit the token limit, often looping. Forcing an answer early recovers much of the loss.
- **The "had the right answer, then lost it" share (had-it) doesn't rise as Lotfi report.**
  Their 26% → 52% under 3-bit AWQ is not reproduced with plain rounding (RTN) on R1-1.5B: it
  stays at about 18–25%. It does rise for pruning and low rank.
- **The had-it share depends on an automatic answer detector.** The detector's loose rule is
  about 80% precise, judged so far by two Claude labellers, **not humans**. That is the weakest
  link a reviewer will attack, and your T11 fixes it.
- **Every number comes from one sample per problem.** Nobody knows yet whether a 2-point gap
  is real. Your T10 fixes that.

## Hard rules

- **Only write to your folders:**
  - `results/r1-1.5b/base-hf`;
  - `results/qwen3-1.7b/*`;
  - the `budget_*` files of `results/r1-1.5b/asvd-r95/none`;
  - `results/r1-1.5b/*/none-vote4`;
  - `results/r1-1.5b/*/pen-{mined,random,entropy}`;
  - `data/validation/human_check_*.csv` and new scripts you add.

  Teammates are writing other `results/` folders at the same time. Commit with explicit paths
  (`git add results/qwen3-1.7b LEDGER.md …`), never `git add -A`. Run `git pull --rebase`
  before every push.
- **Never read, print or commit credentials.** Leave alone `~/.kaggle`, any `access_token` and
  any Hugging Face token. Your human signs in to Kaggle (`kaggle auth login`); you only run the
  scripts. The repo is public.
- **Explain each Kaggle launch first.** Before launching, explain to your human in plain words
  what the run tests, how long it takes, and what each outcome would mean. Then wait for their go.
- **Keep the ledger.** After every pulled session, follow the ledger rules: snapshot, task
  status, a Run log line (also for failed sessions), and Decisions for any setting change.
- **Run the tests.** Run `.venv/bin/python -m pytest -q` before every commit; 42 tests pass
  today.
- **Stop and ask the lead** before changing scoring, sampling, prompts, the detector rule or any
  setting definition. Those would change every existing result.

## Setup

```bash
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest -q
uv tool install kaggle --with six      # then the human runs: kaggle auth login
kaggle quota                           # GPU hours left this week (30 per week, resets Fri 05:00 PKT)
```

The Kaggle account must be phone-verified, so it has GPU and internet access. `kaggle/push.sh
<user> <jobfile> [groups]` creates or updates two datasets under that account (`crs-code`,
`crs-results`) and the notebook `crs-run`, then starts it. Follow it with
`kaggle kernels logs -f <user>/crs-run`, and fetch it with `kaggle/pull.sh <user> crs-run`. The
pull merges chains by id and never overwrites. Score locally:

```bash
.venv/bin/python -m crs.run score --runs 'results/<model>/<setting>/*'
```

## Week 1

### T11: human check of the answer detector (P0, no GPU, about 2 h of human time)

`data/validation/human_check.csv` has **40 spans**. Each shows a problem, the reference answer,
and an excerpt of the model's reasoning with one span marked `<<< >>>`. The question for each
row:

> Does the solver state the reference answer as its answer to the whole problem here, even
> tentatively?

Not as an intermediate quantity that happens to equal it, and not a value copied from the
problem.

1. **Get the labels.**
   - Make a copy for each human labeller: `data/validation/human_check_<initials>.csv`.
   - Your human (and ideally one more teammate, independently) fills `your_label` with YES or
     NO, using `note` for doubts.
   - **Labellers must not open** `human_check_key.json`, `labels_A.json`, `labels_B.json` or
     `sample.jsonl` before finishing: they reveal the answers.
   - Your role as the agent: prepare the files, and render the rows one at a time in readable
     form if that helps your human. **Do not label them yourself**: the point is human labels.
2. **Score them.** Write `data/validation/score_human.py`. It maps each `check_id` through
   `human_check_key.json` to `(item, span)` in `sample.jsonl` and the Claude labels, then
   reports:
   - human vs Claude-labeller agreement, raw and Cohen's κ (reuse `kappa()` from
     `data/validation/agreement.py`);
   - human vs human agreement, if there are two human files;
   - precision of the detector's loose rule against the human labels. Use `agreement.py` to see
     how rule verdicts are derived for each span.
3. **Record the result.** Put the numbers in `data/validation/README.md` and in the ledger's
   findings (point 6), and mark T11 done.

**What would matter:** if humans disagree with the Claude labellers on more than about 15% of
spans, tell the lead at once. The had-it numbers may then need re-estimating.

### T07 + T08 (+ part of T09): one Kaggle session (about 8 h)

The job file is ready: `configs/member-b-week1.yaml`. Read its comments. It runs, in order:

1. **Qwen3-1.7B 30% pruned (wanda-30):** the last 87 of 360 problems, then its forced answers.
2. **Forced answers** (thinking cut at 512/1k/2k/4k/8k tokens, then an answer forced) for:
   - Qwen3 4-bit, which today covers only 338/360;
   - Qwen3 50% pruned, which has none.
3. **Forced answers for R1 low rank (asvd-r95)** on all 360 problems: only 120 have them.
4. **R1 full size on the Hugging Face engine (base-hf):** the last 171 of 360. The KV-cache
   runs can't use vLLM, so they are compared against this control. It currently shows a
   possible ~4-point engine gap that needs the full 360 to settle.

```bash
.venv/bin/python -m crs.run jobs --priority configs/member-b-week1.yaml   # sanity: lists 5 jobs
HOURS=8 kaggle/push.sh <user> member-b-week1.yaml
```

The first command is a sanity check; it should list 5 jobs.

After pulling:
1. Score the touched runs.
2. Rerun `scripts/ledger_table.py --vs-base` and paste the snapshot.
3. Update T07/T08/T09 and add the Run log line.
4. Commit the results folders and the ledger together.

**Check:**
- the `forced@4k` column for `qwen3 rtn-w4` and `r1 asvd-r95` no longer shows an `(n=…)` note;
- `base-hf` reaches n=360;
- the "vs base" interval for `base-hf` either includes 0 (no engine gap) or not. Write which
  in the findings.

## Week 2

### T09: Qwen3 low rank and AWQ (about 5 h)

**Check AWQ on Qwen3 locally first,** if the machine has at least 16 GB of RAM:

```bash
.venv/bin/python scripts/rtn_drift_check.py qwen3-1.7b --settings rtn-w4-g128,awq-w4-g128 --device cpu
```

- It downloads about 4 GB of weights and takes about 15 minutes.
- AWQ is validated on R1 only. Expect AWQ's KL to be clearly below RTN's (RTN W4 was 0.133).
- Afterwards, delete the model from the Hugging Face cache. Remove the blob files under
  `~/.cache/huggingface/hub/`, not just the snapshot links, which don't free disk space.

**Then run the job file** `configs/member-b-week2.yaml` (write it):

```yaml
groups:
  - name: qwen3-lowrank-awq
    models: [qwen3-1.7b]
    settings: [asvd-r95, awq-w4-g128]
    fix: none
```

### T10: four samples per problem (about 12–18 h over 2 sessions)

**Why:** every result so far is one sample per problem. Four samples give:
- the noise level of each accuracy and had-it number;
- majority-vote accuracy (does voting rescue compressed models?).

A group with `samples: 4` writes to `results/r1-1.5b/<setting>/none-vote4`, a separate folder,
so the existing runs are untouched.

```yaml
groups:
  - name: r1-vote4
    models: [r1-1.5b]
    settings: [base, rtn-w3-g128, wanda-50]
    samples: 4
```

- **Order:** base and rtn-w3 first (the Lotfi comparison). Add `awq-w3-g128` only after the
  lead's T01 AWQ runs are pulled and look sane; check the ledger.
- **Speed:** sampling with n > 1 uses the older chunked path, not streaming, so it is slower.
  Check whether loop stopping applies there (`Backend.generate_stream` in
  `src/crs/backends/base.py`), and note it in the ledger.
- The forced-answer post step skips vote runs, which is fine.

**Analysis** (write `scripts/vote_stats.py`, or extend `crs.analysis`; `crs.run report` already
writes a `voting.csv`):
- per-setting accuracy and had-it share for each of the 4 samples, and their spread;
- a bootstrap CI over problems and samples;
- majority-vote accuracy;
- how often a problem is right in some samples but wrong in others.

Put a short table in the ledger.

## Week 3

### T14: controls for the word-penalty fix (after the lead's T13)

The lead runs Lotfi's fix: a penalty on hesitation words like "wait" and "but"
(`fix: pen-hand`, defined in `configs/fixes.yaml`). Your controls show whether any effect comes
from those specific words.

1. Build the word lists locally: `.venv/bin/python -m crs.run wordlists --model r1-1.5b`.
2. Run:
   ```yaml
   groups:
     - name: r1-pen-controls
       models: [r1-1.5b]
       settings: [base, rtn-w3-g128]
       fixes: [pen-mined, pen-random, pen-entropy]
   ```
   - **pen-mined:** words mined from the data.
   - **pen-random:** random words, the placebo.
   - **pen-entropy:** the penalty applies only when the model is uncertain.

### T12: LLM judge (last, after all runs are final)

`crs.run judge` asks Qwen3-4B to confirm each detector hit. It writes `judge_verdicts`,
`had_correct_judged` and `outcome_judged` into every `scored.jsonl`.

- **Run it last.** Check whether `crs.run score` (`score_runs` → `extract/score.py`) keeps the
  judge fields. If it overwrites them, the judge must run after the final scoring, and again
  after any rescore. It also touches **every** run folder, including teammates', so announce
  it in the ledger (and to the lead) before running, and commit straight after.
- **It needs a GPU,** and `kaggle/run_kernel.py` has no judge job type yet. Add a `kind: judge`
  branch: in `expand_jobs` (`src/crs/run.py`) and in the scheduler. It should run
  `python -m crs.run judge --runs '<RESULTS>/*/*/*'` on one GPU after generation. Add a test
  for the job expansion.
- **Afterwards:**
  1. Run `python -m crs.run manual export`. It samples 10% of judged hits per method family
     into `reports/manual_check.csv`.
  2. Have the humans label a subset.
  3. Run `python -m crs.run manual agreement`.
  4. Report the judge's precision against humans in the ledger.

## How to report back

- **Each session:** a ledger Run log line, plus a two-line plain-words summary for your human.
- **Surprises:** flag them to the lead with the folder and numbers. Examples: humans reject
  many detector hits, a big engine gap in base-hf, or vote4 spread larger than the gaps we
  report.
- **Blockers:** if a blocker costs more than a day, write it under Decisions as "blocked: …"
  and tell the lead. Don't silently change settings.
