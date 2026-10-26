"""FastAPI application. Run with `dfs serve` or `uvicorn dfs.api.main:app`."""

import logging
from contextlib import asynccontextmanager
from typing import Annotated, Literal

import psycopg
from fastapi import Depends, FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from dfs.api import schemas
from dfs.api.repository import PostgresRepository, Repository
from dfs.api.services import BadRequestError, NoRunError, NotFoundError, Service
from dfs.config import get_settings

log = logging.getLogger("dfs.api")

ERROR_RESPONSES = {
    400: {"model": schemas.ErrorResponse},
    404: {"model": schemas.ErrorResponse},
    503: {"model": schemas.ErrorResponse},
}


def create_app(repo: Repository | None = None) -> FastAPI:
    """App factory. Tests pass a fake repository; production builds a read-only Postgres one."""
    settings = get_settings()
    owned: PostgresRepository | None = None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal owned
        if repo is None:
            dsn = settings.database_url_readonly
            if not dsn:
                raise RuntimeError("DATABASE_URL_READONLY is not set")
            owned = PostgresRepository(dsn)
            owned.open()
            app.state.repo = owned
        yield
        if owned is not None:
            owned.close()

    app = FastAPI(
        title="Demand Forecasting & Stockout Simulation API",
        version="0.1.0",
        description="Serves precomputed forecasts and metrics, and runs the stockout simulator.",
        lifespan=lifespan,
    )
    if repo is not None:
        app.state.repo = repo
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
    )

    def get_service(request: Request) -> Service:
        return Service(request.app.state.repo)

    Svc = Annotated[Service, Depends(get_service)]

    @app.exception_handler(NotFoundError)
    async def _nf(_: Request, exc: NotFoundError):
        return JSONResponse({"detail": str(exc)}, status_code=404)

    @app.exception_handler(BadRequestError)
    async def _br(_: Request, exc: BadRequestError):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.exception_handler(NoRunError)
    async def _nr(_: Request, exc: NoRunError):
        return JSONResponse({"detail": str(exc)}, status_code=503)

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception):
        # Log the exception type only: messages can embed connection details.
        log.error("unhandled error: %s", type(exc).__name__)
        if isinstance(exc, psycopg.OperationalError):
            return JSONResponse({"detail": "Database unavailable"}, status_code=503)
        return JSONResponse({"detail": "Internal server error"}, status_code=500)

    @app.get("/health", response_model=schemas.HealthResponse, tags=["system"])
    def health(svc: Svc):
        return svc.health()

    @app.get("/series", response_model=schemas.SeriesListResponse, responses=ERROR_RESPONSES, tags=["data"])
    def series(
        svc: Svc,
        store: Annotated[str | None, Query(max_length=16)] = None,
        category: Annotated[str | None, Query(max_length=32)] = None,
        search: Annotated[str | None, Query(max_length=64)] = None,
        limit: Annotated[int, Query(ge=1, le=500)] = 100,
        offset: Annotated[int, Query(ge=0)] = 0,
    ):
        return svc.list_series(store, category, search, limit, offset)

    @app.get("/forecast", response_model=schemas.ForecastResponse, responses=ERROR_RESPONSES, tags=["data"])
    def forecast(
        svc: Svc,
        item: Annotated[str, Query(min_length=1, max_length=64)],
        store: Annotated[str, Query(min_length=1, max_length=16)],
        model: Annotated[list[str] | None, Query(description="Repeat for several models; default all")] = None,
        fold: Annotated[int, Query(ge=0, le=10, description="0 = future, 1-3 = validation")] = 0,
        history_days: Annotated[int, Query(ge=0, le=730)] = 90,
    ):
        return svc.forecast(item, store, model, fold, history_days)

    @app.get("/metrics", response_model=schemas.MetricsResponse, responses=ERROR_RESPONSES, tags=["metrics"])
    def metrics(
        svc: Svc,
        segment_type: Literal["overall", "store", "category", "dept", "weekday", "horizon", "series"] = "overall",
        model: Annotated[str | None, Query(max_length=32)] = None,
        fold: Annotated[int | None, Query(ge=0, le=10, description="0 = pooled folds")] = None,
    ):
        return svc.metrics(segment_type, model, fold)

    @app.get("/errors/top", response_model=schemas.TopErrorsResponse, responses=ERROR_RESPONSES, tags=["metrics"])
    def errors_top(
        svc: Svc,
        n: Annotated[int, Query(ge=1, le=100)] = 10,
        model: Annotated[str | None, Query(max_length=32)] = None,
        sort_by: Literal["wape", "abs_error"] = "wape",
        min_volume: Annotated[float, Query(ge=0, description="Min total validation demand")] = 84.0,
    ):
        return svc.top_errors(n, model, sort_by, min_volume)

    @app.post("/simulate", response_model=schemas.SimulateResponse, responses=ERROR_RESPONSES, tags=["simulation"])
    def simulate(req: schemas.SimulateRequest, svc: Svc):
        return svc.simulate(req)

    @app.get(
        "/simulation/backtest",
        response_model=schemas.BacktestResponse,
        responses=ERROR_RESPONSES,
        tags=["simulation"],
    )
    def simulation_backtest(svc: Svc):
        return svc.backtest()

    return app


app = create_app()
