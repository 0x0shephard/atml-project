# Cross-compression study of the commitment failure (ATML Group 20)

This harness tests whether compressed reasoning models write the correct answer, then talk
themselves out of it. It runs weight quantization, pruning, low-rank truncation and KV-cache
quantization through the same pipeline, and checks whether a training-free hesitation-word
penalty fixes the failure for all of them. See the proposal PDF for the research questions
(RQ1–RQ7).

```
compress (simulated, fp16 weights) → generate (vLLM / HF) → record → score → judge → analyse
```

## Layout

| Path | What |
|---|---|
| `configs/models.yaml` | the two models, sampling, prompt, max 8,192 tokens |
| `configs/settings.yaml` | the 18 versions per model, plus the `base-hf` control |
| `configs/fixes.yaml` | decoding fixes and the budgets 512–4,096 |
| `configs/priority.yaml` | job order if compute runs short; `smoke.yaml` is the 10-problem smoke run |
| `data/problems.jsonl` | the frozen 360 problems: 200 MATH-500, 100 GSM8K, 30 AIME 2024 and 30 AIME 2025 |
| `src/crs/compress/` | RTN fake-quant, Wanda and magnitude pruning, SVD truncation, fake-quant KV cache |
| `src/crs/backends/` | vLLM (weight settings) and HF transformers (KV settings, CPU tests) |
| `src/crs/decoding/` | `logit_bias` penalty, entropy-adaptive vLLM logits processor, budget truncation |
| `src/crs/extract/` | final answer, scan of the reasoning for the correct answer, judge, manual 10% check |
| `src/crs/wordlists/` | hand-written list, mined list (log-odds at high-entropy points), random control |
| `src/crs/analysis/` | paired tests, equal-accuracy-loss matching, trends, regression, figures |
| `kaggle/` | push code and results to Kaggle, run the jobs on T4×2, pull the outputs |

Run outputs go to `results/<model>/<setting>/<fix>/`. Each holds `records.jsonl`,
`arrays/part-*.npz` (token ids, chosen-token logprob and top-20 entropy per token),
`scored.jsonl`, `budget_*.jsonl` and `meta.json`. Runs resume chunk by chunk.

### Outcome labels

| Label | Meaning |
|---|---|
| `correct` | the final answer is right |
| `truncated` | no final answer, because generation hit the token limit; never counted as a commitment failure |
| `commit_fail` | the final answer is wrong or missing, but the correct answer appeared in the reasoning first |
| `never_correct` | the final answer is wrong and the correct answer never appeared |

`outcome_strict` counts only boxed or "the answer is" statements in the reasoning.
`outcome_judged` counts only hits the judge model confirmed.

## Local setup (CPU, for tests and analysis)

```bash
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -e ".[dev]"
```

```bash
.venv/bin/python -m pytest -q
```

```bash
.venv/bin/python -m crs.data build
```

The last command writes `data/problems.jsonl`. It only needs to run once; commit the result.

## Kaggle

1. Install the CLI with `uv tool install kaggle --with six` (Kaggle CLI 2.2.x; it is missing the
   `six` dependency), then sign in once with `kaggle auth login`. `kaggle quota` shows the GPU
   hours left. Sign-in tokens last about 3 hours, and the CLI only refreshes them about
   30 minutes after they expire. Calls in that window fail with "Authentication required", so
   wait or run `kaggle auth login --force`.
2. Run the smoke test (10 problems × 4 settings × 2 fixes, 1,024 tokens):
   ```bash
   kaggle/push.sh <your-kaggle-username> smoke.yaml
   ```
   The kernel asks for **GPU T4 x2** (`machine_shape` in `kernel-metadata.json.tmpl`). It stops
   with a message if it gets an older GPU. Follow it with `kaggle kernels logs -f <user>/crs-run`.
3. When it finishes, pull the outputs:
   ```bash
   kaggle/pull.sh <your-kaggle-username>
   ```
   Then check `results/*/*/*/meta.json`:
   - `records_with_nonfinite_logprobs` should be 0. Otherwise set that model's `dtype` to
     `float32` in `models.yaml`.
   - `tokens_per_s` gives the throughput used to plan the GPU-hours.
4. Measure vLLM speed before the main runs:
   ```bash
   kaggle/push.sh <your-kaggle-username> bench.yaml
   ```
   Each job writes `results/bench/<model>__<attention backend>__c<concurrency>.json`. Put the
   fastest backend in `vllm_attention_backend` in `models.yaml`.
5. Run the main jobs group by group, starting with:
   ```bash
   kaggle/push.sh <your-kaggle-username> priority.yaml P1-main-test
   ```
   Pull after each session and push again. Finished problems are skipped, because the
   results gathered so far are uploaded as the `crs-results` dataset.

## After generation

```bash
python -m crs.run score --runs 'results/*/*/*'
```

```bash
python -m crs.run wordlists --model r1-1.5b
```

Run `wordlists` before the P4 jobs: it needs the `base` and `rtn-w3/w4` chains, and writes
`mined_r1-1.5b.txt` and `random_r1-1.5b.txt`.

```bash
python -m crs.run judge --runs 'results/*/*/*'
```

The judge needs a GPU. Run it on Kaggle, or anywhere vLLM runs.

```bash
python -m crs.run manual export
```

Fill in `human_label` in `reports/manual_check.csv`, then:

```bash
python -m crs.run manual agreement
```

```bash
python -m crs.run report --results results --out reports --judged
```

The report writes:
- **Paired comparisons:** `vs_baseline.csv` (each version vs uncompressed, McNemar and
  bootstrap) and `fix_effects.csv` (each fix vs no fix on the same version).
- **Method comparisons:** `equal_loss.csv` (failure metrics at 5/10/15-point accuracy drops)
  and `severity_trends.csv`.
- **Failure and pruning checks:** `failure_regression.csv`, `budget_curves.csv`,
  `pruning_gain_vs_budget.csv` and `voting.csv`.
- **Figures and summary:** the figures and `summary.md`.

## Design notes and known simplifications

- **Simulated compression.** Compressed values are written back into fp16 weights. Embeddings
  and `lm_head` are not compressed.
- **Wanda calibration.** Activation norms come from one pass through the dense model, not
  Wanda's layer-by-layer pass. Calibration text is the baseline's own reasoning chains,
  falling back to wikitext.
- **KV-cache settings.** These run on the HF backend with KIVI-style fake-quantized K/V.
  Keys are quantized per channel over blocks of 32 tokens, values per token in groups of 32
  channels, and the newest 128 tokens stay in full precision. They are compared with
  `base-hf`, so differences in backend numerics don't read as compression effects.
  Per-token keys, the first version, made int4 output gibberish: Qwen keys have outlier
  channels.
- **Penalty token matching.** The fixed penalty hits every single-token variant of each word.
  Words with no single-token variant are skipped and logged in `meta.json`. Hesitation rates
  are always counted with the hand-written list, whatever the fix.
- **Entropy.** It is estimated from the top-20 logprobs of the raw distribution, the same way
  in both backends. The estimate is a lower bound.
- **Smaller budgets.** These come from truncating the 8,192-token chains and forcing a boxed
  answer greedily. No new chains are sampled.
- **Hand-written word list.** `src/crs/wordlists/handwritten.txt` is a first draft. Align it
  with Lotfi et al.'s list before the main runs.
