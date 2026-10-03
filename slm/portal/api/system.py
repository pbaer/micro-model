from __future__ import annotations

import sys
from pathlib import Path

from fastapi import APIRouter, Request

from slm.utils.gpu import query_gpu

router = APIRouter(prefix="/api")


def gpu_info() -> dict:
    return query_gpu()


@router.get("/meta")
def meta(request: Request) -> dict:
    s = request.app.state.settings
    return {
        "runs_root": str(Path(s.runs_root).resolve()), "data_root": str(s.data_root), "configs_root": str(Path(s.configs_root).resolve()),
        "python": sys.version.split()[0], "gpu_policy": s.gpu_policy,
        "pages": [
            {"id": "home", "label": "Overview"}, {"id": "evals", "label": "Evals"}, {"id": "data", "label": "Data"},
            {"id": "tokenizer", "label": "Tokenizer"}, {"id": "inference", "label": "Inference"}, {"id": "arena", "label": "Arena"},
            {"id": "arch", "label": "Architecture"},
        ],
        "stages": [{"name": "pretrain", "label": "Pretraining"}],
    }


@router.get("/system/gpu")
def gpu(request: Request) -> dict:
    info = gpu_info()
    info["training_live"] = request.app.state.runs.live_runs()
    return info
