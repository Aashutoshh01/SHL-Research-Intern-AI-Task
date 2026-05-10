"""TalentRoute AI — FastAPI application entry point.

Production-style application factory with proper startup/shutdown
lifecycle, CORS configuration, and route registration.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.core.logging import setup_logging, get_logger
from app.api.routes.health import router as health_router
from app.api.routes.chat import router as chat_router
from app.services.retrieval import initialize_retrieval

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle manager.

    Handles startup initialization (embedding model, FAISS index)
    and cleanup on shutdown.
    """
    settings = get_settings()
    setup_logging(settings.log_level)

    logger.info(
        "application_starting",
        llm_model=settings.llm_model,
        embedding_model=settings.embedding_model,
    )

    # Load embedding model and FAISS index at startup
    initialize_retrieval()
    logger.info("application_ready")

    yield

    logger.info("application_shutting_down")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance.
    """
    settings = get_settings()

    app = FastAPI(
        title="TalentRoute AI",
        description=(
            "Conversational AI assessment recommender system powered by "
            "the SHL product catalog. Supports multi-turn stateless "
            "conversations for assessment selection, comparison, and refinement."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routes
    app.include_router(health_router)
    app.include_router(chat_router)

    from fastapi.responses import RedirectResponse

    @app.get("/", include_in_schema=False)
    async def root_redirect():
        return RedirectResponse(url="/docs")

    return app


# Application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )
