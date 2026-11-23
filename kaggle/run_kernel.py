"""Kaggle entry point (pushed as a script kernel by kaggle/push.sh).

Inputs attached as datasets:
  <user>/crs-code     the repo (src/, configs/, data/, pyproject.toml)
  <user>/crs-results  results gathered so far (so finished problems are skipped)

Runs the jobs of JOB_FILE (optionally only GROUPS) on every visible GPU, one job per
GPU at a time, until the jobs are done or TIME_BUDGET_H runs out (running jobs are stopped
then, so the session ends cleanly). Then forces answers at each thinking budget. Scoring
is left to `crs.run score` locally. Everything is written to /kaggle/working.
"""

import glob
import json
import os
import shutil
import signal
import subprocess
import sys
import time

JOB_FILE = "priority.yaml"  # set by push.sh
GROUPS: list = []  # set by push.sh; empty = all groups
TIME_BUDGET_H = 10.5  # generation stops here (set by push.sh); Kaggle kills the session at 12 h
POST_BUDGET_H = 1.2   # then the forced-answer step gets this long. Scoring runs locally (slow on CPU)
VLLM_SPEC = "vllm==0.11.0"  # needs AdapterLogitsProcessor (>= 0.11); T4 works in float16
# vLLM 0.11 breaks on transformers 5 (Kaggle's default image): tokenizers lost all_special_tokens_extended
TRANSFORMERS_SPEC = "transformers>=4.56,<5"

T0 = time.time()
WORK = "/kaggle/working"
RESULTS = f"{WORK}/results"
LOGS = f"{WORK}/logs"


def sh(cmd: str, **kw) -> int:  # noqa: ANN003
    print("+", cmd, flush=True)
    return subprocess.call(cmd, shell=True, **kw)


def find_input(marker: str) -> str | None:
    """Directory holding `marker` under /kaggle/input. Datasets mount at
    /kaggle/input/datasets/<owner>/<slug>/ (older images: /kaggle/input/<slug>/), so search any depth."""
    hits = sorted(glob.glob(f"/kaggle/input/**/{marker}", recursive=True), key=len)
    return os.path.dirname(hits[0]) if hits else None


def unpack_dirs(root: str, names: list[str]) -> None:
    """`--dir-mode zip` uploads folders as <name>.zip; unpack any that Kaggle left zipped."""
    import zipfile

    for n in names:
        z = os.path.join(root, f"{n}.zip")
        if not os.path.isdir(os.path.join(root, n)) and os.path.isfile(z):
            print(f"unzipping {z}", flush=True)
            zipfile.ZipFile(z).extractall(os.path.join(root, n))


def job_name(job: dict) -> str:
    if job.get("kind") == "bench":
        return f"bench__{job['model']}__{job['attn']}__c{job['concurrency']}"
    if job.get("kind") == "budget":
        return f"budget__{job['model']}__{job['setting']}"
    if job.get("kind") == "extend":
        return f"extend__{job['model']}__{job['setting']}__{job['results']}"
    name = f"{job['model']}__{job['setting']}__{job['fix']}__n{job['samples']}"
    return f"{name}__{job['results']}" if job.get("results") else name


def main() -> None:
    os.makedirs(LOGS, exist_ok=True)
    code_src = find_input("pyproject.toml")
    if code_src is None:
        sys.exit("crs-code dataset not attached")
    code = "/tmp/crs-code"  # outside /kaggle/working, so the code is not part of the output
    shutil.copytree(code_src, code, dirs_exist_ok=True)
    unpack_dirs(code, ["src", "configs", "data"])
    sh(f"pip install -q '{VLLM_SPEC}' '{TRANSFORMERS_SPEC}' 'math-verify[antlr4_13_2]' statsmodels tabulate")
    sh(f"pip install -q --no-deps -e {code}")

    import torch

    if not torch.cuda.is_available():
        sys.exit("no GPU: set the notebook accelerator to 'GPU T4 x2'")
    cc = torch.cuda.get_device_capability(0)
    if cc < (7, 0):
        sys.exit(f"GPU compute capability {cc} is too old for vLLM (P100?): set the accelerator to 'GPU T4 x2'")
    n_gpu = torch.cuda.device_count()

    prev = find_input("crs-results.marker")
    if prev:
        unpack_dirs(prev, ["results"])
        if os.path.isdir(os.path.join(prev, "results")):
            shutil.copytree(os.path.join(prev, "results"), RESULTS, dirs_exist_ok=True)
            print(f"restored earlier results from {prev}", flush=True)
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
    running: dict[int, tuple[subprocess.Popen, dict, str, float]] = {}
    queue = list(jobs)
    status = []
    while queue or running:
        for gpu, (proc, job, ck, started) in list(running.items()):
            limit = job.get("timeout_min")
            if proc.poll() is None and limit and time.time() - started > 60 * limit:
                print(f"GPU{gpu}: {job_name(job)} hit its {limit} min limit, stopping it", flush=True)
                proc.kill()
                proc.wait()
                job = {**job, "timed_out": True}
            elif proc.poll() is None and (time.time() - T0) / 3600 > TIME_BUDGET_H:
                # session nearly over: stop now so the forced-answer step still runs. Finished
                # chunks are on disk, only the one in progress is lost; the job resumes next session.
                print(f"GPU{gpu}: {job_name(job)} stopped at the session time budget", flush=True)
                proc.kill()
                proc.wait()
                job = {**job, "stopped_at_budget": True}
                queue.insert(0, {k: v for k, v in job.items() if k != "stopped_at_budget"})
            if proc.poll() is not None:
                # vLLM runs its engine in a child process that outlives a killed parent and keeps
                # holding GPU memory; in the 2026-11-21 bench run that made the next jobs on the
                # GPU fail. Each job gets its own process group, so clear the whole group.
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                status.append({**job, "returncode": proc.returncode})
                shutil.rmtree(ck, ignore_errors=True)
                del running[gpu]
        hours = (time.time() - T0) / 3600
        for gpu in range(n_gpu):
            if gpu in running or not queue or hours > TIME_BUDGET_H:
                continue
            job = queue.pop(0)
            ck = f"/tmp/ckpt/gpu{gpu}"
            name = job_name(job)
            if job.get("kind") == "bench":
                cmd = [sys.executable, "-m", "crs.run", "bench", "--model", job["model"], "--attn", job["attn"],
                       "--concurrency", str(job["concurrency"]), "--gen-tokens", str(job["gen_tokens"]),
                       "--results", RESULTS, "--ckpt-dir", ck]
            elif job.get("kind") == "extend":
                cmd = [sys.executable, "-m", "crs.run", "extend", "--model", job["model"], "--setting", job["setting"],
                       "--max-tokens", str(job["max_tokens"]), "--from-results", RESULTS,
                       "--results", f"{WORK}/{job['results']}", "--chunk-size", str(job["chunk_size"]),
                       "--ckpt-dir", ck, *(["--problems", f"data/{job['problems']}"] if job.get("problems") else [])]
            elif job.get("kind") == "budget":
                cmd = [sys.executable, "-m", "crs.run", "budget", "--model", job["model"], "--setting", job["setting"],
                       "--fix", "none", "--results", RESULTS, "--ckpt-dir", ck]
            else:
                cmd = [sys.executable, "-m", "crs.run", "gen", "--model", job["model"], "--setting", job["setting"],
                       "--fix", job["fix"], "--samples", str(job["samples"]),
                       "--results", f"{WORK}/{job.get('results', 'results')}", "--ckpt-dir", ck, *extra,
                       *(["--limit", str(job["limit"])] if job.get("limit") else []),
                       *(["--max-tokens", str(job["max_tokens"])] if job.get("max_tokens") else []),
                       *(["--problems", f"data/{job['problems']}"] if job.get("problems") else []),
                       *(["--chunk-size", str(job["chunk_size"])] if job.get("chunk_size") else [])]
            log = open(f"{LOGS}/{name}.log", "a")
            print(f"[{hours:.2f} h] GPU{gpu}: {name}", flush=True)
            running[gpu] = (subprocess.Popen(cmd, cwd=code, env={**env, "CUDA_VISIBLE_DEVICES": str(gpu)},
                                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True),
                            job, ck, time.time())
        if not running and (not queue or hours > TIME_BUDGET_H):
            break
        time.sleep(20)

    def write_status() -> None:
        json.dump({"jobs": status, "hours": (time.time() - T0) / 3600, "left": queue},
                  open(f"{WORK}/status.json", "w"), indent=1)

    write_status()
    # forced answers at each thinking budget (incl. the 8,192 cut-off) for the no-fix,
    # single-sample runs that have chains, first those that finished this session
    seen, todo = set(), []
    # (runs in a separate results folder, e.g. a longer length limit, are skipped, and so are runs
    # whose forced answers already exist, e.g. from a `kind: budget` job this session)
    for job in [j for j in status if j.get("kind", "gen") == "gen"] + [j for j in jobs if j.get("kind", "gen") == "gen"]:
        key = (job["model"], job["setting"])
        rd = f"{RESULTS}/{job['model']}/{job['setting']}/none"
        if job["fix"] == "none" and job["samples"] == 1 and key not in seen and not job.get("results") \
                and os.path.exists(f"{rd}/records.jsonl") and not os.path.exists(f"{rd}/budget_8192.jsonl"):
            seen.add(key)
            todo.append(job)
    for job in todo:
        if (time.time() - T0) / 3600 > TIME_BUDGET_H + POST_BUDGET_H:
            print("out of time for the forced-answer step", flush=True)
            break
        sh(f"{sys.executable} -m crs.run budget --model {job['model']} --setting {job['setting']} --fix none "
           f"--results {RESULTS} --ckpt-dir /tmp/ckpt/budget >> {LOGS}/budget.log 2>&1", cwd=code, env=env)
        shutil.rmtree("/tmp/ckpt/budget", ignore_errors=True)
    open(f"{RESULTS}/KEEP", "w").close()
    write_status()
    print(json.dumps({"hours": (time.time() - T0) / 3600, "left": len(queue)}), flush=True)


if __name__ == "__main__":
    main()
