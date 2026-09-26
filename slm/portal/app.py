from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from slm.portal.api import arch as arch_api
from slm.portal.api import data as data_api
from slm.portal.api import evals as evals_api
from slm.portal.api import model as model_api
from slm.portal.api import runs as runs_api
from slm.portal.api import system as system_api
from slm.portal.api import tokenizer as tokenizer_api
from slm.portal.services.datasets import DataCatalog
from slm.portal.services.evals import EvalIndex
from slm.portal.services.runs import RunIndex
from slm.portal.services.tokenizer import TokenizerRegistry
from slm.portal.services.worker import WorkerClient
from slm.portal.settings import PortalSettings

STATIC = Path(__file__).parent / "static"


class NoCacheStatic(StaticFiles):
    """Static assets are edited constantly during development; never let the browser cache them."""

    def file_response(self, *args, **kwargs):  # type: ignore[override]
        resp = super().file_response(*args, **kwargs)
        resp.headers["Cache-Control"] = "no-cache, must-revalidate"
        return resp


@asynccontextmanager
async def _lifespan(app: FastAPI):
    async def idle_loop():
        while True:
            await asyncio.sleep(30)
            app.state.worker.maybe_idle_stop()

    task = asyncio.create_task(idle_loop())
    try:
        yield
    finally:
        task.cancel()
        app.state.worker.stop()


def create_app(settings: PortalSettings | None = None) -> FastAPI:
    settings = settings or PortalSettings()
    app = FastAPI(title="slm command center", docs_url="/api/docs", redoc_url=None, lifespan=_lifespan)
    app.state.settings = settings
    app.state.runs = RunIndex(settings.runs_root)
    app.state.evals = EvalIndex(settings.runs_root, app.state.runs)
    app.state.data = DataCatalog(settings.data_root, settings.cache_dir)
    app.state.tokenizers = TokenizerRegistry(Path(settings.data_root) / "tokenizer")
    app.include_router(system_api.router)
    app.include_router(runs_api.router)
    app.include_router(evals_api.router)
    app.include_router(data_api.router)
    app.include_router(tokenizer_api.router)
    app.include_router(arch_api.router)
    app.include_router(model_api.router)
    app.state.worker = WorkerClient(Path(settings.data_root) / "tokenizer", settings.worker_idle_timeout_s)
    app.state.streams = {}

    app.mount("/static", NoCacheStatic(directory=STATIC), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    return app
