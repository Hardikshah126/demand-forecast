"""API request validation and response contracts (fake repository, no database)."""

import pytest
from fastapi.testclient import TestClient

from dfs.api import schemas
from dfs.api.main import create_app
from tests.conftest import FakeRepository

ITEM, STORE = "FOODS_3_090", "CA_1"


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert schemas.HealthResponse(**r.json()).latest_run_id == 7


def test_health_degraded_when_db_down():
    with TestClient(create_app(FakeRepository(healthy=False))) as c:
        body = c.get("/health").json()
    assert body == {"status": "degraded", "database": "unavailable", "latest_run_id": None}


def test_series_listing_and_pagination(client):
    r = client.get("/series", params={"limit": 1})
    assert r.status_code == 200
    body = schemas.SeriesListResponse(**r.json())
    assert body.total == 2 and len(body.items) == 1 and body.limit == 1
    assert len(client.get("/series", params={"store": "TX_1"}).json()["items"]) == 1


@pytest.mark.parametrize("params", [{"limit": 0}, {"limit": 501}, {"offset": -1}])
def test_series_rejects_bad_pagination(client, params):
    assert client.get("/series", params=params).status_code == 422


def test_forecast_contract(client):
    r = client.get("/forecast", params={"item": ITEM, "store": STORE, "model": ["xgboost", "naive"]})
    assert r.status_code == 200
    body = schemas.ForecastResponse(**r.json())
    assert set(body.models) == {"xgboost", "naive"}
    pts = body.models["xgboost"]
    assert len(pts) == 28 and pts[0].horizon == 1
    assert pts[0].lower_80 <= pts[0].y_pred <= pts[0].upper_80
    assert body.history and body.history[-1].date < body.horizon_start
    assert "xgboost" in body.series_metrics


def test_forecast_defaults_to_all_models_and_validation_fold_has_actuals(client):
    body = client.get("/forecast", params={"item": ITEM, "store": STORE, "fold": 2}).json()
    assert len(body["models"]) == 4
    assert body["models"]["naive"][0]["actual"] == 5


def test_forecast_errors(client):
    assert client.get("/forecast", params={"item": "NOPE", "store": STORE}).status_code == 404
    bad_model = client.get("/forecast", params={"item": ITEM, "store": STORE, "model": "prophet"})
    assert bad_model.status_code == 400 and "Unknown model" in bad_model.json()["detail"]
    assert client.get("/forecast", params={"item": ITEM, "store": STORE, "fold": 9}).status_code == 400
    assert client.get("/forecast", params={"item": ITEM}).status_code == 422


def test_metrics_contract_and_filters(client):
    body = client.get("/metrics").json()
    parsed = schemas.MetricsResponse(**body)
    assert parsed.run.best_model == "xgboost" and parsed.segment_type == "overall"
    assert {r.model for r in parsed.rows} == set(FakeRepository.MODELS)
    only = client.get("/metrics", params={"model": "naive", "fold": 0}).json()["rows"]
    assert len(only) == 1 and only[0]["model"] == "naive"
    assert client.get("/metrics", params={"segment_type": "bogus"}).status_code == 422
    assert client.get("/metrics", params={"model": "bogus"}).status_code == 400


def test_top_errors_direction_and_limits(client):
    body = client.get("/errors/top", params={"n": 2}).json()
    parsed = schemas.TopErrorsResponse(**body)
    assert [i.direction for i in parsed.items] == ["overforecast", "underforecast"]
    assert len(client.get("/errors/top", params={"n": 1}).json()["items"]) == 1
    for bad in (0, 101):
        assert client.get("/errors/top", params={"n": bad}).status_code == 422


def _sim(client, **kw):
    return client.post("/simulate", json={"item": ITEM, "store": STORE, **kw})


def test_simulate_contract(client):
    r = _sim(client, demand_change_pct=10, lead_time_days=5, service_level=0.9, initial_cover_days=10)
    assert r.status_code == 200, r.text
    body = schemas.SimulateResponse(**r.json())
    assert body.n_paths == 10_000 and len(body.dates) == 28
    assert 0 <= body.recommended.stockout_probability <= 1
    assert body.reorder_point == pytest.approx(body.expected_lead_time_demand + body.safety_stock)
    assert body.assumptions and set(body.trajectory_percentiles) == {"p10", "p50", "p90"}


def test_simulate_is_deterministic_per_seed(client):
    assert _sim(client, seed=5).json() == _sim(client, seed=5).json()


@pytest.mark.parametrize(
    "payload",
    [
        {"demand_change_pct": -31}, {"demand_change_pct": 51}, {"lead_time_days": 0},
        {"lead_time_days": 29}, {"service_level": 0.5}, {"service_level": 0.999},
        {"initial_cover_days": -1}, {"n_paths": 10}, {"unexpected": 1},
    ],
)  # fmt: skip
def test_simulate_validation_errors(client, payload):
    assert _sim(client, **payload).status_code == 422


def test_simulate_unknown_series_and_model(client):
    assert client.post("/simulate", json={"item": "NOPE", "store": STORE}).status_code == 404
    assert _sim(client, model="nope").status_code == 400


def test_backtest_contract(client):
    body = client.get("/simulation/backtest").json()
    parsed = schemas.BacktestResponse(**body)
    assert set(parsed.by_policy) == {"recommended", "no_safety_stock"}
    assert 0 <= parsed.by_policy["recommended"].realized_stockout_frequency <= 1


def test_503_when_no_run_exists():
    with TestClient(create_app(FakeRepository(with_run=False))) as c:
        r = c.get("/metrics")
    assert r.status_code == 503 and "dfs train" in r.json()["detail"]


def test_cors_allows_configured_origin_only(client):
    ok = client.get("/health", headers={"Origin": "http://localhost:3000"})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:3000"
    other = client.get("/health", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in other.headers


def test_responses_never_contain_credentials(client):
    text = "".join(
        client.get(p).text for p in ("/health", "/metrics", "/series")
    ).lower()
    assert "password" not in text and "postgresql://" not in text
