"""应用入口：uvicorn app.main:app --reload"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api import routes_chat, routes_documents, routes_system
from app.config import Settings, get_settings
from app.container import Container, build_container


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    resolved_settings = settings or get_settings()
    injected = container

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.container = injected or build_container(resolved_settings)
        try:
            yield
        finally:
            app.state.container.store.close()

    app = FastAPI(
        title=resolved_settings.app_name,
        version=__version__,
        description="企业知识库智能问答：混合检索 + RAG + 引用溯源",
        lifespan=lifespan,
    )
    app.state.settings = resolved_settings

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(routes_system.router)
    app.include_router(routes_documents.router)
    app.include_router(routes_chat.router)

    @app.get("/")
    async def index() -> dict:
        return {
            "name": resolved_settings.app_name,
            "version": __version__,
            "docs": "/docs",
            "health": "/api/health",
        }

    return app


app = create_app()
