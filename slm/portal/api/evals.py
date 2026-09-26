from __future__ import annotations

import anyio
from fastapi import APIRouter, Request

router = APIRouter(prefix="/api")


@router.get("/evals")
async def evals(request: Request) -> dict:
    """Every evaluated checkpoint x every benchmark / homebrew eval (services/evals.py), with per-column colour positions."""
    return await anyio.to_thread.run_sync(request.app.state.evals.table)
