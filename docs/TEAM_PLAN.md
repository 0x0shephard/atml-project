# Team plan: from workshop results to a NeurIPS-level paper

Written 2026-11-26. Task IDs (T01…) match the task board in [LEDGER.md](../LEDGER.md), and the
ledger is where status lives. This file explains **why** each piece matters and **how** to run it.

## Why these tasks

We already have a clear result on 1.5–1.7B models:
- compressed reasoning models mostly fail because they **can't stop**;
- Lotfi et al.'s rise in "had it, then lost it" errors does **not** reproduce with plain rounding.

A NeurIPS reviewer will ask five things. Each workstream answers one:

| Reviewer question | Answer | Owner |
|---|---|---|
| "Is this just a small-model effect?" Lotfi go up to 32B | 7–8B models (W2) | **Member A** |
| "Your quantizer differs from theirs" | AWQ, the method Lotfi used (W3) | Lead |
| "One sample per problem, so how big is the noise?" | 4 samples on key settings, CIs everywhere (W5) | **Member B** |
| "Is your answer detector right?" | Human check + LLM judge (W6) | **Member B** |
| "What should practitioners do?" | Their word penalty vs our stopping rule (W4) | Lead (+B for controls) |

The lead keeps the AWQ comparison, the new fix and the writing, because those decide the
paper's story. Members A and B get independent, well-defined runs that never touch the same
folders.

## Compute

- Each person runs on their **own Kaggle account**: 30 GPU-h per week on T4×2, resetting
  Friday 05:00 PKT. Three accounts give about 90 GPU-h per week, against roughly 120 GPU-h of
  work left. So **about 2 weeks of runs** if everyone launches promptly after each reset.
- One account can run **one session per notebook name**. To run two sessions at once, use
  `KERNEL=crs-run-b`, and give them different groups (see `kaggle/push.sh`).
- Kaggle kills sessions at 12 h. Set `HOURS` so generation plus forced answers
  (`POST_HOURS`, default 1.2) fit.

## Setup (everyone, once)

```bash
git clone https://github.com/0x0shephard/atml-project && cd atml-project
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest -q          # 42 tests should pass
uv tool install kaggle --with six && kaggle auth login
```

Then read `README.md` (Kaggle section) and `LEDGER.md` (Rules).

The loop for every session:
1. **Push:** `HOURS=<h> kaggle/push.sh <your-kaggle-user> <jobfile> <group>`
2. **Follow:** `kaggle kernels logs -f <user>/crs-run`
3. **Pull:** `kaggle/pull.sh <your-kaggle-user> crs-run` merges by chain id and never
   overwrites.
4. **Score:** `.venv/bin/python -m crs.run score --runs 'results/<model>/<setting>/*'`
5. **Update the ledger:** `.venv/bin/python scripts/ledger_table.py --vs-base`, paste the
   snapshot, update your task row, and add a Run log line.
6. **Commit and push:** `git pull --rebase`, then commit the results and the ledger together.

Using Claude Code? Paste your brief below into it as the task. `CLAUDE.md` in the repo tells it
to keep the ledger.

---

## Member A brief: scale-up to 7–8B (W2: T03 → T04 → T05 → T06)

**Goal:** show whether the "can't stop" pattern and the flat had-it share hold at 7–8B. This is
the single biggest step from workshop to main conference.

### T03: two-GPU support (code, ~1–2 days)

A 7B model in fp16 is about 15 GB and doesn't fit on one 16 GB T4 with room for the KV cache.
Spread it over both GPUs:
- `configs/models.yaml`: add
  - `r1-7b` (`deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`, sampling as r1-1.5b)
  - `qwen3-8b` (`Qwen/Qwen3-8B`, sampling as qwen3-1.7b)
  - give both `tensor_parallel: 2`.
- `src/crs/backends/vllm_backend.py`: pass `tensor_parallel_size` to `LLM(...)`.
- `kaggle/run_kernel.py`: today each job takes one GPU (`CUDA_VISIBLE_DEVICES=<gpu>`). A job
  whose model has `tensor_parallel: 2` must get `"0,1"` and run alone.
- `src/crs/compress/apply.py` `build_checkpoint`: it loads the whole model onto one GPU
  (`.to(device)`), which won't fit at 7B. Use `device_map="auto"` across both GPUs, or the CPU
  (Kaggle has about 29 GB of RAM; slow but fine for RTN and Wanda). Check that the AWQ and
  Wanda hooks still work when layers sit on different devices.
- `HF_HOME` / disk: the 7B checkpoints are about 15 GB each. Delete each compressed checkpoint
  after its run (Kaggle's working disk is about 57 GB).
- Add a test where possible, then a **10-problem smoke run** (`--limit 10`) of r1-7b base
  before any full run. Record its tokens/s in the ledger: it sets the plan.

### T04 / T05: the runs

Write `configs/scale-7b.yaml` with groups like:

```yaml
groups:
  - name: r1-7b-main
    models: [r1-7b]
    settings: [base, rtn-w4-g128, awq-w4-g128, awq-w3-g128, rtn-w3-g128, wanda-50]
    fix: none
  - name: qwen3-8b-main
    models: [qwen3-8b]
    settings: [base, rtn-w4-g128, awq-w4-g128, awq-w3-g128, wanda-50]
    fix: none
```

- **Order:** base first (everything is compared with it), then awq-w3 and rtn-w3 (the Lotfi
  comparison), then the rest.
- **Cost:** expect about 2.5–4 h per setting. Forced answers come automatically after
  generation.
- **Owned folders:** `results/r1-7b/*`, `results/qwen3-8b/*`, `results-32k-ext/r1-7b/*`.

### T06: 32k extend

After T04, extend the cut-off chains of base and the worst setting to 32k on MATH-500. Copy the
`r1-awq-extend` group in `configs/r1-awq.yaml`: it continues saved chains, so it must run only
after the main runs are pulled.

**Done when:** r1-7b has all six settings on 360 problems with forced answers, and the ledger
snapshot shows them.

---

## Member B brief: robustness and validation (W1 leftovers, W5, W6; T07–T12, T14)

**Goal:** make every number in the paper defensible: finished controls, noise estimates and a
human-checked detector. Most of this is small runs and careful checking, with no new code.

### First week (in this order)

1. **T11: human check of the answer detector (no GPU, about 3 h of reading).**
   - Run `.venv/bin/python -m crs.run manual export`.
   - Fill `human_label` in `reports/manual_check.csv`. For each row, decide whether the
     reasoning really stated the correct answer as its conclusion at that point. Don't look at
     the detector's label first.
   - Then run `python -m crs.run manual agreement`, and put the agreement in the ledger.
   - This is P0: reviewers will not accept the had-it numbers without it.
2. **T07 + T08: one launch covers both.** It's ready in `configs/member-b-week1.yaml`:
   ```bash
   HOURS=8 kaggle/push.sh <your-kaggle-user> member-b-week1.yaml
   ```
   The launch runs:
   - Qwen3 30% pruned: 87 problems left, then its forced answers.
   - Forced answers for Qwen3 4-bit and 50% pruned.
   - Forced answers for R1 low rank on all 360 problems.
   - R1 `base-hf`: 171 problems left. This is the engine control for the KV-cache runs, which run
     on Hugging Face, not vLLM.

### Second week

4. **T09:** Qwen3 `asvd-r95` and `awq-w4-g128`, with forced answers. (The R1 `asvd-r95` forced
   answers are already in the week-1 file.)
5. **T10: 4 samples per problem** for R1 base, rtn-w3, wanda-50 and (after the lead's T01)
   awq-w3: `samples: 4` in the job group writes `results/r1-1.5b/<setting>/none-vote4`. This
   gives:
   - the noise level of every R1 number (are 2-point gaps real?);
   - the majority-vote results (P5 in the proposal).
6. **T12: LLM judge** over all runs (`python -m crs.run judge --runs 'results/*/*/*'`). It
   needs a GPU, so run it on Kaggle. It confirms or rejects each "had the answer" hit.
7. **T14: penalty controls** (pen-mined, pen-random, pen-entropy on R1 base + rtn-w3), after
   the lead's T13. Run `python -m crs.run wordlists --model r1-1.5b` locally first.

**Owned folders:**
- `results/r1-1.5b/base-hf`
- `results/qwen3-1.7b/*` (except anything the lead names)
- `results/r1-1.5b/*/none-vote4`
- `results/r1-1.5b/*/pen-{mined,random,entropy}`
- `reports/manual_check.csv`

**Done when:**
- the human check agreement is in the ledger;
- every 1.5B/1.7B setting has forced answers on all problems;
- the 4 vote4 runs are scored.

---

## Lead (with Claude): T01, T02, T13, T15, T16, T17

1. **After the 28 Nov reset:** R1 AWQ main runs (T01), then the 32k extend (T02). Together
   they decide the headline:
   - **If AWQ W3 stays flat:** "the rise doesn't replicate; the failure is not stopping".
   - **If it rises:** "AWQ and RTN fail differently".
2. **T15: our fix.** Since the failure is "can't stop", test a stopping rule: end the thinking
   once the same answer has been concluded k times, then force the answer.
   - It can be studied **offline on the saved chains first**, with zero GPU, using
     `correct_hits` and the forced-answer step.
   - Compare it with Lotfi's word penalty (T13) at equal tokens.
   - This is the paper's practical contribution.
3. **T16–T17:** CIs in every table, the paper outline, figure 1 and related work.

## Calendar (rough)

| Week starting | Lead | Member A | Member B |
|---|---|---|---|
| Fri 28 Nov | T01 AWQ main, T15 offline | T03 code + 7B smoke | T11 human check, T07, T08 |
| Fri 5 Dec | T02 AWQ extend, T13 penalty | T04 r1-7b runs | T09, T10 vote4 |
| Fri 12 Dec | T15 live, T16 | T04 finish, T05 qwen3-8b | T12 judge, T14 controls |
| Fri 19 Dec | T17 draft | T06 7B extend | Reruns / gaps |
| Jan | Full draft; decide ICML (late Jan) vs NeurIPS (May) | Gaps, ablations | Gaps, appendix tables |

**Weekly sync:** each member updates their ledger rows before the Friday reset, so the next
week's launches can be planned from the ledger alone.
