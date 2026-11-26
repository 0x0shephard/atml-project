# Brief for Member A's agent: scale the study to 7–8B models

You are working in the `atml-project` repository, a research harness that studies how compressed
reasoning LLMs fail. Your job is the **scale-up workstream (W2)**: make the pipeline run 7–8B
models on Kaggle's two T4 GPUs, then produce the 7B results. This is the single biggest step
from a workshop paper to a NeurIPS-level one. Everything so far is on 1.5–1.7B models, and the
paper we answer (Lotfi et al., arXiv 2606.00206, NeurIPS 2026 oral) went up to 32B.

You have only this repository as context. Read before doing anything:
1. `CLAUDE.md`: the house rules.
2. `LEDGER.md`: the findings, task board (your tasks are **T03, T04, T05, T06**), run log and
   decisions.
3. `README.md`: the pipeline, Kaggle workflow and design notes.
4. `docs/TEAM_PLAN.md`: why the work is split this way.

## What the project has found (so you know what you're testing)

- **Compressed models mostly fail because they can't stop.** Wrong answers are mostly chains that
  hit the token limit, often looping. Forcing an answer early recovers much of the loss.
- **The "had the right answer, then lost it" share doesn't rise as Lotfi report.** Their 26% →
  52% under 3-bit AWQ is not reproduced with plain rounding (RTN) on R1-1.5B: it stays at about
  18–25%. It does rise for pruning and low rank. AWQ runs on 1.5B (the lead's task) will test
  the method difference.
- **Qwen3-1.7B is far more fragile than R1-1.5B.** It has 5–10× larger activation outliers.

**Your question at 7–8B:** do these three patterns hold?

## Hard rules

- **Only write to your folders:** `results/r1-7b/*`, `results/qwen3-8b/*`,
  `results-32k-ext/r1-7b/*`, `results-smoke-7b/*`, plus the code you change. Teammates are
  writing other `results/` folders at the same time. Commit with explicit paths
  (`git add results/r1-7b LEDGER.md …`), never `git add -A`. Run `git pull --rebase` before
  every push.
- **Never read, print or commit credentials.** Leave alone `~/.kaggle`, any `access_token` and
  any Hugging Face token. Your human signs in to Kaggle (`kaggle auth login`); you only run the
  scripts. The repo is public.
- **Explain each Kaggle launch first.** Before launching, explain to your human in plain words
  what the run tests, how long it takes, and what each outcome would mean. Then wait for their go.
- **Keep the ledger.** After every pulled session, follow the ledger rules: snapshot, task
  status, a Run log line (also for failed sessions), and Decisions for any setting change.
- **Run the tests.** Run `.venv/bin/python -m pytest -q` before every commit; 42 tests pass
  today. Add tests for new logic.
- **Stop and ask the lead** if a change would alter results for the existing 1.5B/1.7B runs:
  sampling, scoring, prompts or the compression code paths those runs use.

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
`kaggle kernels logs -f <user>/crs-run`, and fetch it with `kaggle/pull.sh <user> crs-run`.
Score locally with:

```bash
.venv/bin/python -m crs.run score --runs 'results/r1-7b/*/*'
```

## T03: run 7–8B models across both T4 GPUs (code)

**Why:** a 7B model in fp16 is about 15 GB. One T4 has 15 GB usable, so the model alone fills
it and leaves nothing for the KV cache. vLLM's tensor parallelism splits each layer across both
GPUs (about 7.6 GB of weights each, the rest for the cache). Today every job is pinned to one
GPU.

### Code changes

1. **`configs/models.yaml`.** Add two models, copying sampling and `chat_template_kwargs` from
   their small siblings:
   - `r1-7b`: `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`, family `r1-distill`, sampling as
     `r1-1.5b`.
   - `qwen3-8b`: `Qwen/Qwen3-8B`, family `qwen3-thinking`, sampling as `qwen3-1.7b`,
     `enable_thinking: true`.

   Give both `tensor_parallel: 2` and `dtype: float16`. T4 has no bf16. See the fp16 risk below.
2. **`src/crs/config.py`.** Add `tensor_parallel: int = 1` to `ModelConfig`, and read it in
   `models()` with `v.get("tensor_parallel", 1)`.
3. **`src/crs/run.py` `open_backend` and `src/crs/backends/vllm_backend.py`.**
   - Pass `tensor_parallel_size=mcfg.tensor_parallel` through `make_backend` into `LLM(...)`.
   - On T4s, vLLM's custom all-reduce needs GPU peer-to-peer access, which T4 pairs often
     lack. If startup hangs or errors, pass `disable_custom_all_reduce=True` and/or set the
     environment variable `NCCL_P2P_DISABLE=1`.
   - Check that `generate_stream` (the streaming path) and the loop stopper work unchanged
     with TP=2.
   - The budget step (`budget_run`), `extend_run` and `bench_run` also go through
     `open_backend`, so they get TP automatically. Verify it.
4. **`kaggle/run_kernel.py`, the scheduler loop in `main()`.** Today it starts one job per GPU
   with `CUDA_VISIBLE_DEVICES=<gpu>`. A job whose model has `tensor_parallel: 2` must:
   - wait until **both** GPUs are free;
   - start with `CUDA_VISIBLE_DEVICES=0,1`;
   - occupy both slots until it ends.

   Other rules:
   - The kernel can read the model's TP from `configs/models.yaml` in the code dataset
     (`CRS_CONFIG_DIR`), or `expand_jobs` can add it to the job dict.
   - Keep the time-budget and `killpg` cleanup logic working for TP jobs. vLLM starts worker
     processes, and they must all die with the job.
   - The forced-answer post step at the end runs without `CUDA_VISIBLE_DEVICES`, so it sees
     both GPUs. Make sure it uses TP=2 for these models.
   - Factor the "which job can start on which GPUs" decision into a small pure function, and
     unit-test it.
5. **`src/crs/compress/apply.py` `build_checkpoint`.** It loads the full model with
   `.to(device)` onto one GPU, which won't fit at 7B.
   - Load with `device_map="auto"` across both GPUs (with a `max_memory` cap per GPU) whenever
     `tensor_parallel > 1`.
   - Check that every compression path still works when layers sit on different GPUs: RTN in
     `compress/quant.py`, Wanda's activation-norm hooks, AWQ in `compress/awq.py` (see
     `collect_inputs`, which sends input ids to `next(model.parameters()).device`) and ASVD.
   - After `save_pretrained`, **free both GPUs completely** (`del model`, `gc.collect()`,
     `torch.cuda.empty_cache()`) before vLLM starts in the same process. vLLM grabs 90% of each
     GPU.
   - CPU-only compression is not an option. Kaggle has about 29 GB of RAM, and AWQ on CPU for
     7B would take hours.
6. **Disk.** A compressed 7B checkpoint is about 15 GB under `/tmp/ckpt/gpu<N>`. Make sure each
   job's checkpoint is deleted when it ends (the scheduler already `rmtree`s it), and print
   `df -h /tmp` at kernel start.
7. **Tests.** Add tests for the scheduler function and the config field. The existing tiny-model
   tests must still pass.

### Smoke test (one short session, about 1 h)

Create `configs/scale-smoke.yaml`:

```yaml
groups:
  - name: smoke-7b
    models: [r1-7b]
    settings: [base, rtn-w4-g128]
    fix: none
    limit: 10
    max_tokens: 2048
    results: results-smoke-7b
```

Launch it with:
```bash
HOURS=1 POST_HOURS=0.3 kaggle/push.sh <user> scale-smoke.yaml
```
Explain it to your human first.

Check these in the `meta.json` of each run:
- `records_with_nonfinite_logprobs` must be 0;
- `tokens_per_s` (record it in the ledger);
- that the text reads normally.

**fp16 risk.** Qwen-family models have large activation outliers. Qwen3-1.7B peaked at about
17k against fp16's 65,504 limit, and 8B could overflow. If `qwen3-8b` shows non-finite
logprobs, record it in Decisions and tell the lead before trying a substitute. Candidates are
`deepseek-ai/DeepSeek-R1-Distill-Llama-8B` or keeping only `r1-7b`. Smoke-test `qwen3-8b` the
same way, with 10 problems, before T05.

**Done when:** both smoke runs produce finite, readable chains on two GPUs; the tests pass; and
the code is committed with a ledger Decision explaining the TP design.

## T04: R1-Distill-Qwen-7B main runs

Create `configs/scale-7b.yaml`:

```yaml
groups:
  # session 1: full size first: the compressed versions are calibrated on base's own chains
  - name: r1-7b-base
    models: [r1-7b]
    settings: [base, rtn-w4-g128, rtn-w3-g128]
    fix: none
  # session 2, only after session 1 is pulled and committed (AWQ / Wanda calibrate on base chains)
  - name: r1-7b-calibrated
    models: [r1-7b]
    settings: [awq-w3-g128, awq-w4-g128, wanda-50]
    fix: none
```

- **Order matters:** AWQ and Wanda read calibration text from the base run's saved chains (see
  `calibration` in `configs/settings.yaml` and `build_checkpoint`). If base isn't in the
  uploaded results yet, calibration falls back to generic text, and the runs aren't comparable
  with the 1.5B ones. So:
  1. run session 1;
  2. pull, score and commit;
  3. then push session 2.
- **Timing:** use the smoke run's tokens/s to plan `HOURS`. Expect about 2.5–4 h per setting
  for 360 problems at an 8,192-token limit. A TP=2 job uses both GPUs, so jobs run one after
  another. Kaggle kills a session at 12 h, so keep `HOURS + POST_HOURS ≤ 11`. Unfinished runs
  resume next session; finished problems are skipped.
- **Forced answers** at 512–8,192 tokens run automatically after generation (`POST_HOURS`,
  default 1.2). If they don't fit, add a `kind: budget` group later (see
  `configs/member-b-week1.yaml` for the format).
- **After each session:**
  1. pull and score;
  2. run `.venv/bin/python scripts/ledger_table.py --vs-base` and paste the snapshot into
     `LEDGER.md`;
  3. update T04 and add the Run log line;
  4. commit `results/r1-7b` and `LEDGER.md` together.

**Done when** r1-7b has all six settings on 360 problems with forced answers, scored and in the
ledger. Then write a short comparison in the ledger's findings: 7B against 1.5B on the three
patterns above (accuracy drop, cut-off share, had-it share, forced@4k).

## T05: Qwen3-8B (after T04)

Same two-session structure as T04, in `configs/scale-8b.yaml`, with settings `base`,
`rtn-w4-g128`, `awq-w4-g128`, `awq-w3-g128` and `wanda-50`. Qwen3 at 1.7B was broken at
RTN-W3, so skip `rtn-w3` unless W4 looks healthy. Qwen3's KV cache per token is about 2.5× R1's,
so fewer chains fit at once and it will be slower: plan from its smoke numbers.

## T06: 32k extension for 7B (after T04)

This continues the chains that hit the 8,192 limit up to 32,768 tokens, on the 200 MATH-500
problems, for `base` and the worst-hit setting. Copy the `r1-awq-extend` group in
`configs/r1-awq.yaml` (`kind: extend`, `results: results-32k-ext`, `chunk_size: 20`). It reads
the finished 8k runs, so push it only after those are pulled.

## How to report back

- **Each session:** a ledger Run log line, plus a two-line plain-words summary for your human.
- **Surprises:** if anything contradicts the 1.5B findings, flag it to the lead with the run
  folder and numbers. Examples: AWQ-W3 had-it share near 50%, or no cut-off problem at 7B.
- **Blockers:** if a blocker costs more than a day (NCCL hang, OOM, fp16 overflow), write it
  under Decisions as "blocked: …" and tell the lead. Don't silently change models or settings.
