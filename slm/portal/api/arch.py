from __future__ import annotations

import json
from pathlib import Path

import anyio
from fastapi import APIRouter, HTTPException, Request

from slm.portal.services.hparams import illustrations, model_config_from

# slm.model.introspect pulls in torch (and with it CUDA init); the portal main process stays torch-free
# at import and only loads it inside the one request that needs a graph.

router = APIRouter(prefix="/api/arch")


TRAIN_BENCH_FIELDS = ("seq", "microbatch", "tokens_per_sec", "step_ms")  # a row of scripts/bench_throughput.py


def training_benchmarks(bench_dir: Path = Path("artifacts/bench")) -> list[dict]:
    """The training-throughput benchmarks (scripts/bench_throughput.py: `config` + `results` rows with step timings),
    as {path, config, rows}. The directory also holds other kinds (scripts/bench_inference.py writes prefill/decode
    rows with no model config), which the budget tab's training table cannot show; they are left out here, and an
    unreadable file is skipped rather than failing the listing."""
    out = []
    for p in sorted(bench_dir.glob("*.json")) if bench_dir.exists() else []:
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(d, dict) or not d.get("config"):
            continue
        rows = d.get("results")
        if not isinstance(rows, list) or not rows or not all(isinstance(r, dict) and all(k in r for k in TRAIN_BENCH_FIELDS) for r in rows):
            continue
        out.append({"path": str(p).replace("\\", "/"), "config": str(d["config"]).replace("\\", "/"), "rows": len(rows)})
    return out


def model_file_of(spec: str, runs_root: Path) -> str | None:
    """The model yaml a graph spec rests on (benchmarks are keyed by it): the spec itself, a train config's
    `model_file`, or the `model_file` in a run's stored config (None when the run did not record one)."""
    try:
        if spec.startswith("run:"):
            meta = json.loads((Path(runs_root) / spec[4:] / "run.json").read_text(encoding="utf-8"))
            mf = (meta.get("config") or {}).get("model_file")
        elif spec.startswith("train:"):  # read the key itself: RL configs name a model_file but are not TrainConfigs
            from slm.config import load_yaml

            mf = (load_yaml(spec[6:]) or {}).get("model_file")
        else:
            mf = spec
    except (OSError, KeyError, ValueError):
        return None
    return str(mf).replace("\\", "/") if mf else None


@router.get("/configs")
def configs(request: Request) -> dict:
    s = request.app.state.settings
    models = sorted(str(p).replace("\\", "/") for p in (Path(s.configs_root) / "model").glob("*.yaml"))
    trains = sorted(str(p).replace("\\", "/") for p in (Path(s.configs_root) / "train").glob("*.yaml"))
    runs = [f"run:{n}" for n in request.app.state.runs.names() if (Path(s.runs_root) / n / "run.json").exists()]
    benches = sorted(str(p).replace("\\", "/") for p in Path("artifacts/bench").glob("*.json")) if Path("artifacts/bench").exists() else []
    return {"models": models, "trains": trains, "runs": runs, "benchmarks": benches, "training_benchmarks": training_benchmarks()}


@router.get("/graph")
async def graph(request: Request, config: str, B: int = 8, T: int = 2048, override: list[str] | None = None, expand_layers: int = 1, grad_checkpointing: bool = False, loss_chunk: int = 0) -> dict:
    from slm.model.introspect import build_graph, flops_per_token, kv_cache_shape, memory_budget

    try:
        mcfg, label = model_config_from(config, request.app.state.settings.runs_root, override)
    except (FileNotFoundError, KeyError) as e:
        raise HTTPException(404, f"config not found: {e}") from None
    g = await anyio.to_thread.run_sync(lambda: build_graph(mcfg, expand_layers))
    n = g["totals"]["params"]
    n_ne = g["totals"]["params_non_embedding"]
    g["label"] = label
    g["model_file"] = model_file_of(config, request.app.state.settings.runs_root)
    g["B"], g["T"] = B, T
    g["flops"] = {"per_token_train": flops_per_token(mcfg, T, n_ne), "per_token_train_causal": flops_per_token(mcfg, T, n_ne, causal=True)}
    g["memory"] = memory_budget(mcfg, n, B, T, grad_checkpointing=grad_checkpointing, loss_chunk=loss_chunk)
    g["kv_cache"] = kv_cache_shape(mcfg, 1, mcfg.max_seq_len)
    return g


@router.get("/hparams")
async def hparams(request: Request, config: str) -> dict:
    if not Path(config).exists():
        raise HTTPException(404, "train config not found")
    return await anyio.to_thread.run_sync(lambda: illustrations(config))


@router.get("/benchmark")
def benchmark(request: Request, path: str) -> dict:
    p = Path(path)
    if not p.exists() or p.suffix != ".json" or "bench" not in str(p):
        raise HTTPException(404, "benchmark not found")
    return json.loads(p.read_text(encoding="utf-8"))
