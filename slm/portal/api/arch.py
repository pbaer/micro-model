from __future__ import annotations

import json
from pathlib import Path

import anyio
from fastapi import APIRouter, HTTPException, Request

from slm.portal.services.hparams import illustrations, model_config_from

# slm.model.introspect pulls in torch (and with it CUDA init); the portal main process stays torch-free
# at import and only loads it inside the one request that needs a graph.

router = APIRouter(prefix="/api/arch")


@router.get("/configs")
def configs(request: Request) -> dict:
    s = request.app.state.settings
    models = sorted(str(p).replace("\\", "/") for p in (Path(s.configs_root) / "model").glob("*.yaml"))
    trains = sorted(str(p).replace("\\", "/") for p in (Path(s.configs_root) / "train").glob("*.yaml"))
    runs = [f"run:{n}" for n in request.app.state.runs.names() if (Path(s.runs_root) / n / "run.json").exists()]
    benches = sorted(str(p).replace("\\", "/") for p in Path("artifacts/bench").glob("*.json")) if Path("artifacts/bench").exists() else []
    return {"models": models, "trains": trains, "runs": runs, "benchmarks": benches}


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
