from __future__ import annotations

import anyio
from fastapi import APIRouter, HTTPException, Request

router = APIRouter(prefix="/api")


@router.get("/evals")
async def evals(request: Request) -> dict:
    """Every evaluated checkpoint x every benchmark / homebrew eval (services/evals.py), with per-column colour positions."""
    return await anyio.to_thread.run_sync(request.app.state.evals.table)


@router.get("/evals/detail")
async def eval_detail(request: Request, run: str, checkpoint: str, key: str, file: str | None = None, filter: str = "all",
                      q: str = "", offset: int = 0, limit: int = 100) -> dict:
    """Everything stored for one (checkpoint, column) cell: summary numbers, aggregate tables, per-item rows (paged by
    offset/limit, filtered by verdict `filter` = all|pass|fail|other and text `q`), the source files and the run's
    config (services/eval_detail.py). `file` picks another file that measured the same cell."""
    try:
        return await anyio.to_thread.run_sync(lambda: request.app.state.evals.detail(
            run, checkpoint, key, file=file, filter=filter, q=q, offset=offset, limit=limit))
    except KeyError as e:
        raise HTTPException(404, e.args[0] if e.args else "not found") from None
