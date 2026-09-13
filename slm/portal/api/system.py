from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from fastapi import APIRouter, Request

router = APIRouter(prefix="/api")


def gpu_info() -> dict:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu", "--format=csv,noheader,nounits"],
            text=True, timeout=5, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ).strip().splitlines()[0]
        name, used, total, util, power, temp = [x.strip() for x in out.split(",")]
        return {"available": True, "name": name, "used_gib": float(used) / 1024, "total_gib": float(total) / 1024,
                "util": float(util), "power_w": float(power) if power not in ("[N/A]", "") else None, "temp_c": float(temp) if temp not in ("[N/A]", "") else None}
    except Exception as e:  # noqa: BLE001
        return {"available": False, "error": str(e)[:200]}


@router.get("/meta")
def meta(request: Request) -> dict:
    s = request.app.state.settings
    return {
        "runs_root": str(Path(s.runs_root).resolve()), "data_root": str(s.data_root), "configs_root": str(Path(s.configs_root).resolve()),
        "python": sys.version.split()[0], "gpu_policy": s.gpu_policy,
        "pages": [
            {"id": "home", "label": "Home"}, {"id": "runs", "label": "Runs"}, {"id": "data", "label": "Data"},
            {"id": "tokenizer", "label": "Tokenizer"}, {"id": "model", "label": "Model"}, {"id": "arch", "label": "Architecture"},
        ],
        "stages": [{"name": "pretrain", "label": "Pretraining"}],
    }


@router.get("/system/gpu")
def gpu(request: Request) -> dict:
    info = gpu_info()
    info["training_live"] = request.app.state.runs.live_runs()
    return info
