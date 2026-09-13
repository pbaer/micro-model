from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from slm.portal.api import arch as arch_api
from slm.portal.api import data as data_api
from slm.portal.api import runs as runs_api
from slm.portal.api import system as system_api
from slm.portal.api import tokenizer as tokenizer_api
from slm.portal.services.datasets import DataCatalog
from slm.portal.services.runs import RunIndex
from slm.portal.services.tokenizer import TokenizerRegistry
from slm.portal.settings import PortalSettings

STATIC = Path(__file__).parent / "static"


def create_app(settings: PortalSettings | None = None) -> FastAPI:
    settings = settings or PortalSettings()
    app = FastAPI(title="slm command center", docs_url="/api/docs", redoc_url=None)
    app.state.settings = settings
    app.state.runs = RunIndex(settings.runs_root)
    app.state.data = DataCatalog(settings.data_root, settings.cache_dir)
    app.state.tokenizers = TokenizerRegistry(Path(settings.data_root) / "tokenizer")
    app.include_router(system_api.router)
    app.include_router(runs_api.router)
    app.include_router(data_api.router)
    app.include_router(tokenizer_api.router)
    app.include_router(arch_api.router)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    return app
