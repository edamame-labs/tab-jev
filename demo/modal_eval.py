"""Run the evaluation with the local models (Kev + TabICL) on a Modal GPU instead of this machine.

One container starts the Kev server (torch on the GPU) and runs `eval_benchmark.py <dataset> local`
against it; the results file comes back to demo/results/. Kev's weights and the jev answer cache
live on Modal volumes, so a re-run only pays for new calls.

    uv run --with modal==1.5.5 modal run demo/modal_eval.py                          # full run
    uv run --with modal==1.5.5 modal run demo/modal_eval.py --args "--test 200 --labels 64 --seeds 0"   # smoke test
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GPU = os.environ.get("TAB_JEV_GPU", "L4")
KEV_RUN = "jaredpalmer/kev-0.8b"
KEV_PORT = 8009

app = modal.App("tab-jev-eval")
image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git")
    .pip_install(
        "torch>=2.6,<2.9", "transformers>=5.17,<6", "accelerate>=1.15.0", "peft>=0.21", "datasets>=3.0",
        "pydantic>=2.9", "fastapi>=0.115", "uvicorn>=0.30", "typesafe-sdk>=0.6.0",
        "pandas>=2.0", "scikit-learn>=1.7", "tabicl>=2.2.0", "tabulate>=0.10",
    )
    # Kev's fast kernels for the Qwen3.5 hybrid backbone, as in Kev's own modal_app.py
    .pip_install("flash-linear-attention", "triton>=3.7.1")
    .env({"HF_HOME": "/hf", "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1", "TABICL_ESTIMATORS": "8"})
    .add_local_file(ROOT / "pyproject.toml", "/root/tab-jev/pyproject.toml", copy=True)
    .add_local_file(ROOT / "README.md", "/root/tab-jev/README.md", copy=True)
    .add_local_file(ROOT / "LICENSE", "/root/tab-jev/LICENSE", copy=True)
    .add_local_dir(ROOT / "src", "/root/tab-jev/src", copy=True, ignore=["**/__pycache__"])
    .run_commands("pip install /root/tab-jev")
    .add_local_dir(HERE / "kev" / "kev", "/root/kev/kev", ignore=["**/__pycache__"])
    .add_local_file(HERE / "eval_benchmark.py", "/root/tab-jev/demo/eval_benchmark.py")
    .add_local_file(HERE / "data" / "kickstarter" / "train.csv", "/root/tab-jev/demo/data/kickstarter/train.csv")
)
hf_cache = modal.Volume.from_name("kev-hf-cache", create_if_missing=True)
jev_cache = modal.Volume.from_name("tab-jev-cache", create_if_missing=True)


def wait_for(url: str, seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5):
                return
        except OSError:
            time.sleep(5)
    raise TimeoutError(f"{url} did not come up within {seconds:.0f}s")


@app.function(
    image=image,
    gpu=GPU,
    cpu=8,
    memory=32768,
    timeout=8 * 3600,
    volumes={"/hf": hf_cache, "/root/tab-jev/demo/cache": jev_cache},
)
def evaluate(dataset: str, args: list[str]) -> tuple[str, str]:
    server = subprocess.Popen(
        [sys.executable, "-m", "kev.serve", "--run", KEV_RUN, "--port", str(KEV_PORT)],
        cwd="/root/kev",
        env={**os.environ, "PYTHONPATH": "/root/kev"},
    )
    try:
        wait_for(f"http://127.0.0.1:{KEV_PORT}/v1/models", 20 * 60)
        hf_cache.commit()  # keep Kev's weights for the next run
        subprocess.run([sys.executable, "demo/eval_benchmark.py", dataset, "local", *args], cwd="/root/tab-jev", check=True)
    finally:
        server.terminate()
        jev_cache.commit()
    out = Path(f"/root/tab-jev/demo/results/{dataset}_local_tabicl.md")
    return out.name, out.read_text()


@app.local_entrypoint()
def main(dataset: str = "kickstarter", args: str = "") -> None:
    name, text = evaluate.remote(dataset, args.split())
    out = HERE / "results" / name
    out.write_text(text)
    print(f"wrote {out}")
