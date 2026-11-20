"""Kaggle entry point (pushed as a script kernel by kaggle/push.sh).

Inputs attached as datasets:
  <user>/crs-code     the repo (src/, configs/, data/, pyproject.toml)
  <user>/crs-results  results gathered so far (so finished problems are skipped)

Runs the jobs of JOB_FILE (optionally only GROUPS) on every visible GPU, one job per
GPU at a time, until the jobs are done or TIME_BUDGET_H runs out. Then derives budgets
and scores the runs. Everything is written to /kaggle/working/results.
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import time

JOB_FILE = "priority.yaml"  # set by push.sh
GROUPS: list = []  # set by push.sh; empty = all groups
TIME_BUDGET_H = 11.0  # Kaggle sessions stop at 12 h
VLLM_SPEC = "vllm==0.11.0"  # needs AdapterLogitsProcessor (>= 0.11); T4 works in float16

T0 = time.time()
WORK = "/kaggle/working"
RESULTS = f"{WORK}/results"
LOGS = f"{WORK}/logs"


def sh(cmd: str, **kw) -> int:  # noqa: ANN003
    print("+", cmd, flush=True)
    return subprocess.call(cmd, shell=True, **kw)


def find_input(marker: str) -> str | None:
    hits = glob.glob(f"/kaggle/input/*/{marker}") + glob.glob(f"/kaggle/input/*/*/{marker}")
    return os.path.dirname(hits[0]) if hits else None


def main() -> None:
    os.makedirs(LOGS, exist_ok=True)
    code_src = find_input("pyproject.toml")
    if code_src is None:
        sys.exit("crs-code dataset not attached")
    code = f"{WORK}/crs-code"
    shutil.copytree(code_src, code, dirs_exist_ok=True)
    sh(f"pip install -q '{VLLM_SPEC}' 'math-verify[antlr4_13_2]' statsmodels tabulate")
    sh(f"pip install -q --no-deps -e {code}")

    import torch

    if not torch.cuda.is_available():
        sys.exit("no GPU: set the notebook accelerator to 'GPU T4 x2'")
    cc = torch.cuda.get_device_capability(0)
    if cc < (7, 0):
        sys.exit(f"GPU compute capability {cc} is too old for vLLM (P100?): set the accelerator to 'GPU T4 x2'")
    n_gpu = torch.cuda.device_count()

    prev = find_input("results/KEEP") or None
    if prev:
        shutil.copytree(os.path.join(prev, "results") if os.path.isdir(os.path.join(prev, "results")) else prev,
                        RESULTS, dirs_exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)

    spec = json.loads(subprocess.check_output(
        [sys.executable, "-m", "crs.run", "jobs", "--priority", f"{code}/configs/{JOB_FILE}", "--json"], cwd=code))
    jobs = [j for j in spec["jobs"] if not GROUPS or j["group"] in GROUPS]
    extra = []
    if "limit" in spec:
        extra += ["--limit", str(spec["limit"])]
    if "max_tokens" in spec:
        extra += ["--max-tokens", str(spec["max_tokens"])]
    print(f"{len(jobs)} jobs on {n_gpu} GPU(s)", flush=True)

    env = dict(os.environ, CRS_CONFIG_DIR=f"{code}/configs", CRS_DATA_DIR=f"{code}/data")
    running: dict[int, tuple[subprocess.Popen, dict, str]] = {}
    queue = list(jobs)
    status = []
    while queue or running:
        for gpu, (proc, job, ck) in list(running.items()):
            if proc.poll() is not None:
                status.append({**job, "returncode": proc.returncode})
                shutil.rmtree(ck, ignore_errors=True)
                del running[gpu]
        hours = (time.time() - T0) / 3600
        for gpu in range(n_gpu):
            if gpu in running or not queue or hours > TIME_BUDGET_H:
                continue
            job = queue.pop(0)
            ck = f"/tmp/ckpt/gpu{gpu}"
            name = f"{job['model']}__{job['setting']}__{job['fix']}__n{job['samples']}"
            cmd = [sys.executable, "-m", "crs.run", "gen", "--model", job["model"], "--setting", job["setting"],
                   "--fix", job["fix"], "--samples", str(job["samples"]), "--results", RESULTS, "--ckpt-dir", ck, *extra]
            log = open(f"{LOGS}/{name}.log", "a")
            print(f"[{hours:.2f} h] GPU{gpu}: {name}", flush=True)
            running[gpu] = (subprocess.Popen(cmd, cwd=code, env={**env, "CUDA_VISIBLE_DEVICES": str(gpu)},
                                             stdout=log, stderr=subprocess.STDOUT), job, ck)
        if not running and (not queue or hours > TIME_BUDGET_H):
            break
        time.sleep(20)

    # budgets for the no-fix, single-sample runs, then scoring (cheap)
    for job in [j for j in jobs if j["fix"] == "none" and j["samples"] == 1]:
        if (time.time() - T0) / 3600 > TIME_BUDGET_H + 0.5:
            break
        sh(f"{sys.executable} -m crs.run budget --model {job['model']} --setting {job['setting']} --fix none "
           f"--results {RESULTS} --ckpt-dir /tmp/ckpt/budget >> {LOGS}/budget.log 2>&1", cwd=code, env=env)
        shutil.rmtree("/tmp/ckpt/budget", ignore_errors=True)
    sh(f"{sys.executable} -m crs.run score --runs '{RESULTS}/*/*/*' > {LOGS}/score.log 2>&1", cwd=code, env=env)
    open(f"{RESULTS}/KEEP", "w").close()
    json.dump({"jobs": status, "hours": (time.time() - T0) / 3600, "left": queue}, open(f"{WORK}/status.json", "w"), indent=1)
    print(open(f"{LOGS}/score.log").read())


if __name__ == "__main__":
    main()
